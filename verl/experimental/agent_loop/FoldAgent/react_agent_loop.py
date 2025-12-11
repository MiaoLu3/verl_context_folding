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
ReAct Agent Loop implementation adapted to VERL's AgentLoopBase interface.

This module provides a VERL-compatible wrapper around the existing ReAct agent
implementation, keeping the original tool parsing and environment logic intact.
"""

import asyncio
import copy
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

# Import existing FoldAgent components
from .agents.prompts import create_chat
from .agents.utils import select_env, truncate_prompt, is_weird
from .envs.local_search import extract_fn_call

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReactAgentContext:
    """
    Manages conversation state for the ReAct agent loop.

    Adapted from the original AgentContext in agents/utils.py to work with
    VERL's server_manager.generate() interface.
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

        # Cache generation prompt tokens
        self._generation_prompt: Optional[list[int]] = None

        # Metrics
        self.metrics: dict[str, Any] = {}

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
        """Get the generation prompt tokens (added after the last message)."""
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

    def get_prompt_ids_len(self) -> int:
        """Get length of prompt tokens (first prompt_turn turns)."""
        return len(sum(self.chat_ids[:self.prompt_turn], []))

    def messages(self) -> list[dict]:
        """Return the current conversation messages."""
        return self.chat

    def append_user(self, content: str):
        """Append a user message (environment observation)."""
        turn = {'role': 'user', 'content': content}
        self.chat.append(turn)
        self._generation_prompt = None  # Reset cache

        turn_tokens = self._get_turn_tokens(len(self.chat) - 1)
        self.chat_ids.append(turn_tokens)
        self.log_probs.append([0.0] * len(turn_tokens))
        self.token_mask.append([False] * len(turn_tokens))  # User turns are masked out

    def append_assistant(self, content: str, response_ids: list[int], response_log_probs: Optional[list[float]]):
        """Append an assistant message (LLM response)."""
        turn = {'role': 'assistant', 'content': content}
        self.chat.append(turn)
        self._generation_prompt = None  # Reset cache

        # Include generation prompt + response tokens
        gen_prompt = self.get_generation_prompt()
        turn_tokens = gen_prompt + response_ids

        # Add EOS if not present
        if len(response_ids) == 0 or response_ids[-1] != self.tokenizer.eos_token_id:
            turn_tokens.append(self.tokenizer.eos_token_id)
            if response_log_probs:
                response_log_probs = list(response_log_probs) + [0.0]

        self.chat_ids.append(turn_tokens)

        # Log probs: 0 for generation prompt, actual values for response
        if response_log_probs:
            self.log_probs.append([0.0] * len(gen_prompt) + list(response_log_probs))
        else:
            self.log_probs.append([0.0] * len(turn_tokens))

        # Mask: False for generation prompt, True for LLM-generated response
        self.token_mask.append([False] * len(gen_prompt) + [True] * len(response_ids))
        if len(response_ids) == 0 or response_ids[-1] != self.tokenizer.eos_token_id:
            self.token_mask[-1].append(False)  # EOS token not generated by model

    def get_response_data(self) -> tuple[list[int], list[int], list[float]]:
        """
        Get response tokens, mask, and log probs for VERL output.

        Returns:
            Tuple of (response_ids, response_mask, response_logprobs)
        """
        # Combine all turns after prompt_turn
        response_ids = sum(self.chat_ids[self.prompt_turn:], [])
        response_mask = [1 if m else 0 for turn in self.token_mask[self.prompt_turn:] for m in turn]
        response_logprobs = sum(self.log_probs[self.prompt_turn:], [])

        return response_ids, response_mask, response_logprobs

    def get_current_response_length(self) -> int:
        """Get current total response length (for checking limits)."""
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


@register("react_agent")
class ReactAgentLoop(AgentLoopBase):
    """
    ReAct Agent Loop adapted to VERL's AgentLoopBase interface.

    This agent loop implements the ReAct (Reasoning + Acting) pattern:
    1. Generate a response from the LLM
    2. Parse tool calls from the response
    3. Execute tools in the environment
    4. Append observation and repeat

    The loop terminates when:
    - The agent calls 'finish'
    - Max turns reached
    - Response length limit reached
    - LLM returns None
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        """One-time class-level initialization."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing ReactAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        # Get plugin config for agent-specific settings
        cls.plugin_config = getattr(config.actor_rollout_ref.rollout, 'plugin', None)
        cls.max_turn = getattr(cls.plugin_config, 'max_turn', 64) if cls.plugin_config else 64
        cls.retry_cjk = getattr(cls.plugin_config, 'retry_cjk', 0) if cls.plugin_config else 0
        cls.turn_max_new_tokens = getattr(cls.plugin_config, 'turn_max_new_tokens', -1) if cls.plugin_config else -1

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run the ReAct agent loop.

        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, etc.)
            **kwargs: Dataset fields including 'raw_prompt', 'extra_info', 'ability', etc.

        Returns:
            AgentLoopOutput with prompt_ids, response_ids, response_mask, etc.
        """
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}
        request_id = uuid4().hex

        # Get ability and select environment
        ability = kwargs.get('ability', 'LocalSearch')
        if isinstance(ability, list):
            ability = ability[0] if ability else 'LocalSearch'

        # Initialize environment
        EnvClass = select_env(ability, self.config.actor_rollout_ref.rollout)
        env = EnvClass(self.config.actor_rollout_ref.rollout, self.tokenizer, ability)

        try:
            # Create a minimal item-like object for env.init_env
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

        # Get initial conversation and config from environment
        try:
            user_prompt, agent_config = await env.get_data(item_proxy, None)
        except Exception as e:
            logger.error(f"Error getting data from env: {e}")
            user_prompt = list(kwargs.get('raw_prompt', []))
            agent_config = {'max_turn': self.max_turn, 'meta_info': {}}

        # Create chat using the environment's problem statement
        workflow = kwargs.get('extra_info', {}).get('workflow', 'search')
        if hasattr(env, 'instance_info') and 'problem_statement' in env.instance_info:
            user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item_proxy)
        else:
            user_prompt = list(kwargs.get('raw_prompt', []))

        max_turn = agent_config.get('max_turn', self.max_turn)
        prompt_turn = len(user_prompt)

        # Initialize agent context
        agent_ctx = ReactAgentContext(
            chat=user_prompt,
            tokenizer=self.tokenizer,
            config=self.config,
            prompt_turn=prompt_turn
        )

        # Main agent loop
        iteration = 0
        while iteration < max_turn:
            iteration += 1

            # Check response length limit
            if agent_ctx.get_current_response_length() >= self.response_length:
                logger.info(f"Response length limit reached: {agent_ctx.get_current_response_length()}")
                break

            # Generate LLM response
            response_text = await self._generate_step(
                agent_ctx, sampling_params, request_id, metrics
            )

            if response_text is None:
                break

            # Execute action in environment
            with simple_timer("tool_calls", metrics):
                observation = await run_action(env, response_text)

            if observation is None:
                # Agent called 'finish' or similar termination
                break

            # Append observation as user message
            agent_ctx.append_user(observation)

        # Get reward from environment
        reward_score = None
        try:
            score_msg, reward, reward_dict = await asyncio.wait_for(
                env.get_reward(item_proxy, agent_ctx.messages(), None),
                timeout=600
            )
            reward_score = reward
            logger.info(f"Reward: {reward}")
        except Exception as e:
            logger.error(f"Error getting reward: {e}")

        # Build output
        prompt_ids = sum(agent_ctx.chat_ids[:prompt_turn], [])
        response_ids, response_mask, response_logprobs = agent_ctx.get_response_data()

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            response_logprobs=response_logprobs[:self.response_length] if response_logprobs else None,
            multi_modal_data={},
            reward_score=reward_score,
            num_turns=len(agent_ctx.messages()),
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                'messages': agent_ctx.messages(),
                'env_stats': getattr(env, 'stats', {}),
            }
        )

    async def _generate_step(
        self,
        agent_ctx: ReactAgentContext,
        sampling_params: dict[str, Any],
        request_id: str,
        metrics: dict[str, float]
    ) -> Optional[str]:
        """
        Generate one step of LLM response.

        Args:
            agent_ctx: Agent context with conversation state
            sampling_params: LLM sampling parameters
            request_id: Request ID for sticky session
            metrics: Metrics dict to update

        Returns:
            Generated response text, or None if generation should stop
        """
        prompt_ids = agent_ctx.get_prompt_ids()

        # Calculate max tokens for this turn
        max_total_len = self.prompt_length + self.response_length
        current_len = len(prompt_ids) + agent_ctx.get_current_response_length()
        max_new_tokens = max_total_len - current_len

        # Apply per-turn token limit if configured
        if self.turn_max_new_tokens > 0:
            max_new_tokens = min(max_new_tokens, self.turn_max_new_tokens)

        if max_new_tokens < 10:
            logger.info(f"max_new_tokens {max_new_tokens} too small, stopping")
            return None

        # Update sampling params with max tokens
        turn_sampling_params = {**sampling_params}
        # Note: The server_manager.generate() uses the response_length from config,
        # but we can limit via the prompt_ids length

        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=turn_sampling_params,
                image_data=None,
            )

        if output is None or len(output.token_ids) == 0:
            return None

        # Decode response
        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        )

        # Retry if weird output (CJK/repetition check)
        if self.retry_cjk > 0 and is_weird(response_text):
            for _ in range(int(self.retry_cjk)):
                output = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_sampling_params,
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

        # Append assistant response to context
        agent_ctx.append_assistant(
            content=response_text,
            response_ids=output.token_ids,
            response_log_probs=output.log_probs
        )

        return response_text
