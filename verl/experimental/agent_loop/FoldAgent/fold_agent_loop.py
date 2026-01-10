# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fold Agent Loop implementation adapted to VERL's AgentLoopBase interface.

The Fold Agent extends the ReAct agent with branching capabilities:
- Main agent can spawn branch agents for sub-tasks
- Branch agents inherit context and return results to main
- Supports session summarization for long contexts
- Process reward for training signal shaping
- mask_rollout flag for controlling gradient computation
- Returns multiple AgentLoopOutput (one per agent trajectory)
"""

import asyncio
import copy
import logging
import os
import random
import re
import time
from typing import Any, Optional, Union
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

# Import existing FoldAgent components
from .agents.prompts import (
    create_chat,
    BRANCH_MESSAGE_SEARCH,
    BRANCH_MESSAGE,
    SUMMARY_PROMPT_CODE,
    SUMMARY_PROMPT_SEARCH,
)
from .agents.utils import select_env, truncate_prompt, truncate_text, is_weird
from .agents.verifier import judge_scope

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def extract_fn_call(text):
    if text is None:
        return None
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}


def extract_summary(text: str) -> Optional[str]:
    """Extract summary from <summary> tags."""
    matches = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def clean_response(response: str) -> str:
    """Clean response by removing return function markers."""
    if '<function=return>' in response:
        response = response.split('<function=return>')[-1]
    else:
        response = re.split(r'<\[[^\]]+\]>', response)[-1]
    return response


def print_chat(chat: list[dict]) -> str:
    """Convert chat to printable string format."""
    chat_str = ""
    for turn in chat:
        if is_weird(str(turn)):
            chat_str += '# ' + turn['role'] + ' **CJK**\n\n' + turn['content'] + "\n\n---\n\n"
        else:
            chat_str += '# ' + turn['role'] + '\n\n' + turn['content'] + "\n\n---\n\n"
    return chat_str


class FoldAgentContext:
    """
    Manages conversation state for the Fold agent loop.

    Similar to ReactAgentContext but with support for:
    - Multiple agent instances (main + branches)
    - Process reward assignment
    - Info cache for reward shaping
    """

    def __init__(self, chat: list[dict], tokenizer, config, prompt_turn: int = 2):
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_turn = prompt_turn
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.context_uid = str(uuid4())

        # Truncate prompt if needed
        self.chat = copy.deepcopy(chat)
        self.chat = truncate_prompt(self.chat, self.prompt_length, tokenizer, prompt_turn)

        # Token tracking for each turn
        self.chat_ids: list[list[int]] = [self._get_turn_tokens(i) for i in range(len(self.chat))]
        self.log_probs: list[list[float]] = [[0.0] * len(turn) for turn in self.chat_ids]
        self.token_mask: list[list[bool]] = [[False] * len(turn) for turn in self.chat_ids]
        self.additional_info: list[Optional[dict]] = [None] * len(self.chat_ids)

        # Cache generation prompt tokens
        self._generation_prompt: Optional[list[int]] = None

        # Metrics and info cache
        self.metrics: dict[str, Any] = {}
        self.info_cache: dict[str, Any] = {}

    def _get_turn_tokens(self, i: int) -> list[int]:
        """Get tokens for turn i (excluding previous turns)."""
        tokens = self.tokenizer.apply_chat_template(
            self.chat[:i + 1], add_generation_prompt=False, tokenize=True
        )
        if i > 0:
            prev_tokens = self.tokenizer.apply_chat_template(
                self.chat[:i], add_generation_prompt=False, tokenize=True
            )
            return tokens[len(prev_tokens):]
        return tokens

    def get_generation_prompt(self) -> list[int]:
        """Get the generation prompt tokens."""
        if self._generation_prompt is None:
            tokens_without = self.tokenizer.apply_chat_template(
                self.chat, add_generation_prompt=False, tokenize=True
            )
            tokens_with = self.tokenizer.apply_chat_template(
                self.chat, add_generation_prompt=True, tokenize=True
            )
            self._generation_prompt = tokens_with[len(tokens_without):]
        return self._generation_prompt

    def get_prompt_ids(self) -> list[int]:
        """Get all token IDs up to and including generation prompt."""
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def context(self, turn_cut: Optional[int] = None) -> list[int]:
        """Get context tokens, optionally cutting at a specific turn."""
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def messages(self) -> list[dict]:
        """Return the current conversation messages."""
        return self.chat

    def append_user(self, content: str):
        """Append a user message (environment observation)."""
        turn = {'role': 'user', 'content': content}
        self.chat.append(turn)
        self._generation_prompt = None

        turn_tokens = self._get_turn_tokens(len(self.chat) - 1)
        self.chat_ids.append(turn_tokens)
        self.log_probs.append([0.0] * len(turn_tokens))
        self.token_mask.append([False] * len(turn_tokens))
        self.additional_info.append(None)

    def append_assistant(self, content: str, response_ids: list[int], response_log_probs: Optional[list[float]]):
        """Append an assistant message (LLM response)."""
        turn = {'role': 'assistant', 'content': content}
        self.chat.append(turn)
        self._generation_prompt = None

        gen_prompt = self.get_generation_prompt()
        turn_tokens = gen_prompt + response_ids

        if len(response_ids) == 0 or response_ids[-1] != self.tokenizer.eos_token_id:
            turn_tokens.append(self.tokenizer.eos_token_id)
            if response_log_probs:
                response_log_probs = list(response_log_probs) + [0.0]

        self.chat_ids.append(turn_tokens)

        if response_log_probs:
            self.log_probs.append([0.0] * len(gen_prompt) + list(response_log_probs))
        else:
            self.log_probs.append([0.0] * len(turn_tokens))

        self.token_mask.append([False] * len(gen_prompt) + [True] * len(response_ids))
        if len(response_ids) == 0 or response_ids[-1] != self.tokenizer.eos_token_id:
            self.token_mask[-1].append(False)

        self.additional_info.append(None)

    def rollback(self, k: int = 1):
        """Rollback the last k turns."""
        self.chat = self.chat[:-k]
        self.chat_ids = self.chat_ids[:-k]
        self.log_probs = self.log_probs[:-k]
        self.token_mask = self.token_mask[:-k]
        self.additional_info = self.additional_info[:-k]
        self._generation_prompt = None

    def set_process_reward(self, turn, reward: float):
        """Set process reward for specific turns."""
        if isinstance(turn, str) and turn.lower() == 'all':
            turn = list(range(len(self.chat)))
        if not isinstance(turn, list):
            turn = [turn]
        for i in turn:
            if i <= 0 or i > len(self.chat) - 1:
                continue
            if self.additional_info[i] is None:
                self.additional_info[i] = {}
            self.additional_info[i]['process_reward'] = reward

    def set_cache(self, key: str, value: Any):
        """Set a value in the info cache."""
        self.info_cache[key] = value

    def get_response_data(self) -> tuple[list[int], list[int], list[float]]:
        """Get response tokens, mask, and log probs for VERL output."""
        response_ids = sum(self.chat_ids[self.prompt_turn:], [])
        response_mask = [1 if m else 0 for turn in self.token_mask[self.prompt_turn:] for m in turn]
        response_logprobs = sum(self.log_probs[self.prompt_turn:], [])
        return response_ids, response_mask, response_logprobs

    def get_current_response_length(self) -> int:
        """Get current total response length."""
        return len(sum(self.chat_ids[self.prompt_turn:], []))


async def run_action(env, response: str) -> Optional[str]:
    """Execute an action in the environment and return the observation."""
    try:
        env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)

        if 'action' in env_return:
            action = env_return['action']
            if action == 'finish':
                return None

        observation = env_return.get('observation', 'Empty')
        return observation

    except asyncio.TimeoutError:
        return 'Action timed out after 120 seconds'
    except Exception as e:
        return f"Error: {e}"

# TODO@Miao[Done]: handle the output of process_reward_mask

@register("fold_agent")
class FoldAgentLoop(AgentLoopBase):
    """
    Fold Agent Loop adapted to VERL's AgentLoopBase interface.

    The Fold Agent extends ReAct with hierarchical branching:
    - Main agent orchestrates the overall task
    - Branch agents handle sub-tasks with inherited context
    - Supports session summarization for long contexts
    - Process reward for training signal shaping
    - mask_rollout flag for controlling gradient computation

    Returns:
        list[AgentLoopOutput]: One output per agent (main + all branches)
        During evaluation, only returns main agent's output.
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        """One-time class-level initialization."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing FoldAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        # Get plugin config
        cls.plugin_config = getattr(config.actor_rollout_ref.rollout, 'plugin', None)
        cls.max_turn = getattr(cls.plugin_config, 'max_turn', 64) if cls.plugin_config else 64
        cls.max_session = getattr(cls.plugin_config, 'max_session', 5) if cls.plugin_config else 5
        cls.session_timeout = getattr(cls.plugin_config, 'session_timeout', 90 * 60) if cls.plugin_config else 5400
        cls.enable_summary = getattr(cls.plugin_config, 'enable_summary', False) if cls.plugin_config else False
        cls.branch_len = getattr(cls.plugin_config, 'branch_len', None) if cls.plugin_config else None
        cls.process_reward = getattr(cls.plugin_config, 'process_reward', None) if cls.plugin_config else None
        cls.retry_cjk = getattr(cls.plugin_config, 'retry_cjk', 0) if cls.plugin_config else 0
        cls.must_finish = getattr(cls.plugin_config, 'must_finish', None) if cls.plugin_config else None
        cls.max_traj = getattr(cls.plugin_config, 'max_traj', None) if cls.plugin_config else None

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        """
        Run the Fold agent loop.

        Returns:
            list[AgentLoopOutput]: One output per agent trajectory.
            During training: returns all agents (main + branches)
            During evaluation: returns only main agent
        """
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}
        request_id = uuid4().hex
        uid = kwargs.get('uid', request_id)
        gen_uid = kwargs.get('gen_uid', None)
        print("uid:", uid)
        print("gen_uid:", gen_uid)

        # Check if training or evaluation
        is_train = kwargs.get('is_train', True)

        # Get ability and select environment
        ability = kwargs.get('ability', 'LocalSearch')
        if isinstance(ability, list):
            ability = ability[0] if ability else 'LocalSearch'

        # Initialize environment
        EnvClass = select_env(ability, self.config.actor_rollout_ref.rollout)
        env = EnvClass(self.config.actor_rollout_ref.rollout, self.tokenizer, ability)

        try:
            class ItemProxy:
                def __init__(self, kwargs):
                    self.non_tensor_batch = {
                        'extra_info': [kwargs.get('extra_info', {})],
                        'ability': [kwargs.get('ability', 'LocalSearch')],
                        'uid': [kwargs.get('uid', request_id)],
                        'reward_model': [kwargs.get('reward_model', 'default')],
                    }
                    self.meta_info = kwargs.get('meta_info', {})

            item_proxy = ItemProxy(kwargs)
            await env.init_env(item_proxy)
        except Exception as e:
            logger.error(f"Error during environment init: {e}")

        # Get initial conversation
        try:
            user_prompt, agent_config = await env.get_data(item_proxy, None)
        except Exception as e:
            logger.error(f"Error getting data from env: {e}")
            user_prompt = list(kwargs.get('raw_prompt', []))
            agent_config = {'max_turn': self.max_turn, 'meta_info': {}}

        # Create chat using the environment's problem statement
        workflow = kwargs.get('workflow', 'search')
        if hasattr(env, 'instance_info') and 'problem_statement' in env.instance_info:
            user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item_proxy)
        else:
            user_prompt = list(kwargs.get('raw_prompt', []))

        # Select prompts based on workflow
        branch_prompt = BRANCH_MESSAGE_SEARCH if 'search' in workflow else BRANCH_MESSAGE
        summary_prompt = SUMMARY_PROMPT_SEARCH if 'search' in workflow else SUMMARY_PROMPT_CODE

        max_turn = agent_config.get('max_turn', self.max_turn)
        prompt_turn = len(user_prompt)

        # Initialize main agent and tracking structures
        agents: dict[str, FoldAgentContext] = {}
        agents['main'] = FoldAgentContext(
            chat=user_prompt,
            tokenizer=self.tokenizer,
            config=self.config,
            prompt_turn=prompt_turn
        )

        branches: list[str] = []
        branch_tasks: dict[str, str] = {}
        branch_returns: dict[str, str] = {}
        session_message: list[dict] = []

        init_len = len(agents['main'].context())
        current = 'main'
        session_start_time = time.time()
        iteration = 0

        # mask_rollout: If True, no gradient update on this trajectory
        mask_rollout = True

        # Main agent loop
        while iteration < max_turn:

            if time.time() - session_start_time > self.session_timeout:
                logger.info('Session Timeout')
                break

            iteration += 1

            # Check for session summarization (context getting full)
            if self.enable_summary and agents[current].get_current_response_length() > self.response_length * 0.95:

                if len(agents) >= self.max_session:
                    logger.info(f'Session OOC after {len(agents)} sessions')
                    break

                # Summarize current session
                agents[current].rollback(k=2)
                agents[current].append_user(summary_prompt)
                session_message.append({'role': 'user', 'content': summary_prompt})

                summary_response = await self._generate_step(
                    agents[current], sampling_params, request_id, metrics
                )
                session_message.append({'role': 'assistant', 'content': summary_response})

                if summary_response is None:
                    break

                summary = extract_summary(summary_response) or summary_response
                next_session_prompt = (
                    f"For this question, you have already made the following progress in previous session, "
                    f"summarized as follow:\n\n{summary}\n\nNow continue work on it."
                )

                # Create new session
                current = current + '+'
                agents[current] = FoldAgentContext(
                    chat=user_prompt,
                    tokenizer=self.tokenizer,
                    config=self.config,
                    prompt_turn=prompt_turn
                )
                agents[current].append_user(next_session_prompt)
                session_message.append({'role': 'user', 'content': next_session_prompt})

            # Generate main agent response
            response = await self._generate_step(
                agents[current], sampling_params, request_id, metrics
            )
            if response is None:
                break

            session_message.append({'role': 'assistant', 'content': response})

            # Check for branch or regular action
            fn_call = extract_fn_call(response)

            if fn_call is not None and fn_call['function'] == 'branch':
                # Handle branch creation
                if len(branches) + 1 > self.max_session:
                    observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
                else:
                    description = fn_call['arguments'].get('description', 'Agent')
                    message_to_branch = fn_call['arguments'].get('prompt', 'Empty prompt')
                    logger.info(f'[BRANCH] {description}')

                    agent_name = f"#{len(branches)}-" + description.replace(' ', '_')
                    branches.append(agent_name)
                    branch_tasks[agent_name] = message_to_branch

                    # Create branch agent with inherited context
                    history = agents[current].messages()
                    agents[agent_name] = FoldAgentContext(
                        chat=history,
                        tokenizer=self.tokenizer,
                        config=self.config,
                        prompt_turn=prompt_turn
                    )

                    branch_prompt_formatted = branch_prompt.format(message=message_to_branch)
                    agents[agent_name].append_user(branch_prompt_formatted)

                    # Run branch agent loop
                    branch_result = await self._run_branch(
                        agents[agent_name],
                        env,
                        sampling_params,
                        request_id,
                        metrics,
                        max_turn=max_turn - iteration,
                        max_tokens=self.branch_len,
                        timeout=self.session_timeout - (time.time() - session_start_time),
                        description=description,
                    )

                    iteration += branch_result['iteration']
                    last_response = branch_result['last_response']

                    # Add branch messages to session
                    session_message.extend(agents[agent_name].messages()[len(history):])

                    # Extract branch return message
                    fn_call = extract_fn_call(last_response)
                    branch_message = None
                    if fn_call is not None and fn_call['function'] in ('return', 'finish'):
                        if 'message' in fn_call['arguments']:
                            branch_message = fn_call['arguments'].get('message', 'Empty message')
                            branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'

                    if branch_message is None:
                        branch_message = f'Branch has finished its task. The last message was:\n\n{clean_response(last_response)}'

                    observation = branch_message
                    branch_returns[agent_name] = observation
            else:
                # Execute regular action
                with simple_timer("tool_calls", metrics):
                    observation = await run_action(env, response)

                if observation is None:
                    # Agent called 'finish' - this is a proper termination
                    mask_rollout = False
                    break

            # Ensure proper role alternation
            if agents[current].chat[-1]['role'] == 'user':
                logger.warning('[ROLE ERROR] Appending empty assistant turn')
                agents[current].chat.append({'role': 'assistant', 'content': str(response)})
                agents[current]._generation_prompt = None
                turn_tokens = agents[current]._get_turn_tokens(len(agents[current].chat) - 1)
                agents[current].chat_ids.append(turn_tokens)
                agents[current].log_probs.append([0.0] * len(turn_tokens))
                agents[current].token_mask.append([False] * len(turn_tokens))
                agents[current].additional_info.append(None)

            # Truncate observation if process reward enabled
            if self.process_reward:
                observation = truncate_text(observation, max_lines=100, merge_repeat=True, merge_num=4)

            agents[current].append_user(observation)
            session_message.append({'role': 'user', 'content': observation})

        # Record session time in env stats
        env.stats['session_time'] = time.time() - session_start_time

        # Get reward
        print("TASK FINISHED, STARTING REWARD")
        reward_score = None
        score = ('', 0)
        reward_dict = {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}
        try:
            score_msg, reward, reward_dict = await asyncio.wait_for(
                env.get_reward(item_proxy, agents['main'].messages(), None),
                timeout=600
            )
            score = (score_msg, reward)
            reward_score = reward
            logger.info(f"Reward: {reward}")
        except Exception as e:
            logger.error(f"Error getting reward: {e}")

        # Update env stats
        # Notice: the main_len and main turn is only valid when summary is disabled
        env.stats['get_final_score'] = score[1]
        env.stats['traj_num'] = len(agents)
        env.stats['main_len'] = min(len(agents['main'].context()) - init_len, self.response_length)
        env.stats['total_token'] = len(self.tokenizer.encode(print_chat(user_prompt + session_message)))
        env.stats['main_turn'] = len(agents['main'].messages())
        env.stats['is_branch'] = int(len(agents) > 1)
        env.stats['branch_success'] = int(int(len(agents) > 1) * score[1])
        env.stats['use_all_branch'] = int(len(branches) + 1 > self.max_session)

        # Determine mask_rollout based on finish status and reward
        is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
        if is_finish:
            mask_rollout = False
        if score[1] > 0:
            mask_rollout = False

        # If must_finish is set and agent didn't finish, zero out score
        if self.must_finish and not is_finish:
            score = ('', 0)
            reward_score = 0

        # If process_reward is enabled, always allow gradient updates
        if self.process_reward and is_train:
            mask_rollout = False
            # Apply process reward logic (CJK check, scope check, etc.)
            await self._apply_process_rewards(
                agents, branches, branch_tasks, branch_returns, env, score, is_finish
            )

        # Build outputs for all agents
        outputs: list[AgentLoopOutput] = []
        session_message_str = print_chat(session_message)

        # During eval, only return main agent; during train, return all agents
        agent_names_to_output = list(agents.keys()) if is_train else ['main']

        for name in agent_names_to_output:
            agent_ctx = agents[name]
            prompt_ids = sum(agent_ctx.chat_ids[:agent_ctx.prompt_turn], [])
            response_ids, response_mask, response_logprobs = agent_ctx.get_response_data()

            process_reward_values = sum(
                [
                    [info.get('process_reward', 0) if isinstance(info, dict) else 0] * len(turn_ids)
                    for turn_ids, info in zip(agent_ctx.chat_ids, agent_ctx.additional_info)
                ][agent_ctx.prompt_turn:],
                []
            )
            process_reward_mask = [int(v) for v in process_reward_values[:self.response_length]]
            process_reward_mask = [p * m for p, m in zip(process_reward_mask, response_mask[:self.response_length])]

            if self.process_reward and 'flat' in self.process_reward and 'reward' in agent_ctx.info_cache:
                agent_reward = agent_ctx.info_cache['reward']
            else:
                agent_reward = reward_score

            output = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[:self.response_length],
                response_mask=response_mask[:self.response_length],
                response_logprobs=response_logprobs[:self.response_length] if response_logprobs else None,
                multi_modal_data={},
                reward_score=agent_reward,
                num_turns=len(agent_ctx.messages()),
                metrics=AgentLoopMetrics(**metrics),
                extra_fields={
                    'messages': agent_ctx.messages(),
                    'env_stats': copy.deepcopy(env.stats) if hasattr(env, 'stats') else {},
                    'num_branches': len(branches),
                    'branch_names': branches,
                    'mask_rollout': mask_rollout,
                    'is_finish': is_finish,
                    'agent_name': name,
                    'message_str': session_message_str,
                    'meta_info': f"N: {len(agents)} | {name}",
                    'process_reward_mask': process_reward_mask,
                    'uid': uid,
                    'gen_uid': gen_uid,
                }
            )
            outputs.append(output)

        # Limit number of trajectories if max_traj is set
        if self.max_traj is not None and len(outputs) > self.max_traj:
            # Always keep main (index 0), randomly sample the rest
            idx = [0] + sorted(random.sample(range(1, len(outputs)), k=self.max_traj - 1))
            outputs = [outputs[i] for i in idx]

        return outputs

    async def _apply_process_rewards(
        self,
        agents: dict[str, FoldAgentContext],
        branches: list[str],
        branch_tasks: dict[str, str],
        branch_returns: dict[str, str],
        env,
        score: tuple,
        is_finish: bool,
    ):
        """Apply process reward logic to agents."""
        process_reward = self.process_reward
        if not process_reward:
            return

        # CJK check
        if 'cjk' in process_reward:
            env.stats['is_cjk'] = 0
            for name in agents:
                for i, turn in enumerate(agents[name].chat):
                    if is_weird(str(turn)):
                        logger.warning('[CJK ERROR]')
                        env.stats['is_cjk'] = 1
                        agents[name].set_process_reward(i, -1)
                        if 'flat' in process_reward:
                            agents[name].set_cache('reward', 0)

        if score[1] > 0:
            # Success case
            init_len = len(agents['main'].context(turn_cut=agents['main'].prompt_turn))

            # Check main agent context length
            if len(agents['main'].context()) - init_len > self.response_length * 0.5:
                bad_turn = [
                    i for i, turn in enumerate(agents['main'].messages())
                    if '<function=branch>' not in str(turn) and '<function=finish>' not in str(turn)
                ]
                agents['main'].set_process_reward(bad_turn, -1)

            # Penalize if no branching was used
            if len(agents) == 1:
                agents['main'].set_process_reward('all', -1)
                if 'flat' in process_reward:
                    agents['main'].set_cache('reward', 0)

            # Scope check for branches
            if 'scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    if name not in agents:
                        continue
                    assigned_task = branch_tasks.get(name, '')
                    return_message = branch_returns.get(name, '')
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:
                        logger.info(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agents[name].set_process_reward(list(range(len(agents[name].chat) - 1)), -0.2)
                        if 'flat' in process_reward:
                            agents[name].set_cache('reward', 1 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agents[name].set_process_reward(list(range(len(agents[name].chat) - 1)), 0.2)
                        if 'flat' in process_reward:
                            agents[name].set_cache('reward', 1 + 0.2)

            # Tool call error check
            for name in branches:
                if name not in agents:
                    continue
                for i, turn in enumerate(agents[name].chat):
                    ERR_MARKERS = (
                        'Failed to validate tool call',
                        'Failed to parse tool call',
                        'You are in branch mode and cannot branch task or finish the task.',
                        'No function call was detected in the model response',
                        '[Error] The "search" function requires a "query" argument',
                        '[Error] The "open_page" function requires either a "docid" or a "url".',
                        '[Error] The function',
                    )
                    if any(m in str(turn) for m in ERR_MARKERS):
                        agents[name].set_process_reward(i - 1, -1)
        else:
            # Failure case
            if 'drop_fail' in process_reward:
                if not is_finish:
                    for name in list(branches):
                        if name not in agents:
                            continue
                        if 'cjk' in process_reward:
                            should_drop = True
                            for i, turn in enumerate(agents[name].chat):
                                if is_weird(str(turn)):
                                    agents[name].set_process_reward(i, -2)
                                    if 'flat' in process_reward:
                                        agents[name].set_cache('reward', -1)
                                    should_drop = False
                            if should_drop:
                                del agents[name]
                        else:
                            del agents[name]

            # Scope check with reward
            if 'reward_scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    if name not in agents:
                        continue
                    assigned_task = branch_tasks.get(name, '')
                    return_message = branch_returns.get(name, '')
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:
                        logger.info(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agents[name].set_process_reward(list(range(len(agents[name].chat) - 1)), -0.2)
                        if 'flat' in process_reward:
                            agents[name].set_cache('reward', 0 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agents[name].set_process_reward(list(range(len(agents[name].chat) - 1)), 0.2)
                        if 'flat' in process_reward:
                            agents[name].set_cache('reward', 0 + 0.2)

    async def _generate_step(
        self,
        agent_ctx: FoldAgentContext,
        sampling_params: dict[str, Any],
        request_id: str,
        metrics: dict[str, float],
    ) -> Optional[str]:
        """Generate one step of LLM response."""
        prompt_ids = agent_ctx.get_prompt_ids()

        max_total_len = self.prompt_length + self.response_length
        current_len = len(prompt_ids) + agent_ctx.get_current_response_length()
        max_new_tokens = max_total_len - current_len

        if max_new_tokens < 10:
            logger.info(f"max_new_tokens {max_new_tokens} too small, stopping")
            return None

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
            )

        if output is None or len(output.token_ids) == 0:
            return None

        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        )

        # Retry if weird output
        if self.retry_cjk > 0 and is_weird(response_text):
            for _ in range(int(self.retry_cjk)):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    image_data=None,
                )
                if output is None:
                    return None
                response_text = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                )
                if not is_weird(response_text):
                    break

        agent_ctx.append_assistant(
            content=response_text,
            response_ids=output.token_ids,
            response_log_probs=output.log_probs
        )

        return response_text

    async def _run_branch(
        self,
        agent_ctx: FoldAgentContext,
        env,
        sampling_params: dict[str, Any],
        request_id: str,
        metrics: dict[str, float],
        max_turn: int,
        max_tokens: Optional[int],
        timeout: float,
        description: str,
    ) -> dict[str, Any]:
        """Run a branch agent loop."""
        branch_start = time.time()
        iteration = 0
        init_len = len(agent_ctx.context(turn_cut=agent_ctx.prompt_turn))
        last_response = None

        max_response_tokens = max_tokens if max_tokens else self.response_length - 512

        while iteration < max_turn:
            if time.time() - branch_start > timeout:
                logger.info('[BRANCH] Timeout')
                break

            if len(agent_ctx.context()) - init_len > max_response_tokens:
                # Context full, force summary/return
                agent_ctx.append_user(
                    "The context limit has been exceeded for the branch. Please finish the sub task directly "
                    "and clearly state the progress made and the pending jobs of the sub task. "
                    "Only summarize the sub task progress, using the return tool."
                )
                response = await self._generate_step(agent_ctx, sampling_params, request_id, metrics)
                if response:
                    last_response = response
                break

            iteration += 1

            response = await self._generate_step(agent_ctx, sampling_params, request_id, metrics)
            if response is None:
                break

            last_response = response

            # Check for return/finish
            if '<function=return>' in response:
                break

            # Check for invalid branch actions
            if '<function=finish>' in response or '<function=branch>' in response:
                observation = (
                    "You are in branch mode and cannot branch task or finish the task. "
                    "Use the `return` tool to go back to the main agent."
                )
            else:
                with simple_timer("tool_calls", metrics):
                    observation = await run_action(env, response)

                if observation is None:
                    break

            observation += f"\n* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you complete the assigned sub task, use return tool to return, do not perform action beyond the assigned sub task."
            agent_ctx.append_user(observation)

        # record assistant turns in this branch
        return {
            'last_response': last_response or '',
            'iteration': iteration,
        }
    