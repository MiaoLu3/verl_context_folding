# Context-Folding: Scaling Long-Horizon LLM Agents

>**Warning⚠️:** This repository is under active development. The code has been tested to run correctly, but full validation of training results is still in progress.

This repository contains the RL training implementation of **Context-Folding**, a framework that empowers LLM agents to actively manage their working context for long-horizon tasks. Built on top of [VeRL](https://github.com/volcengine/verl).



**Paper:** [Scaling Long-Horizon LLM Agent via Context-Folding](https://arxiv.org/abs/2510.11967)

**Project Page:** [https://context-folding.github.io/](https://context-folding.github.io/)

## Abstract

Large language model (LLM) agents are fundamentally constrained by context length on long-horizon tasks. We introduce **Context-Folding**, a framework that empowers agents to actively manage their working context. An agent can procedurally **branch** into a sub-trajectory to handle a subtask and then **fold** it upon completion, collapsing the intermediate steps while retaining a concise summary of the outcome. To make this behavior learnable, we develop an end-to-end reinforcement learning framework **FoldGRPO** with specific process rewards to encourage effective task decomposition and context management. On complex long-horizon tasks like Deep Research, our folding agent matches or outperforms the ReAct baselines while using an active context **10x smaller** and significantly outperforms models that rely on summarization-based context management.

## Key Results

| Model | Peak Length | Max #Token | BrowseComp-Plus | SWE-Bench Verified |
|-------|-------------|------------|-----------------|-------------------|
| GPT-5 | 327K | 327K | 79.3% | 71.8% |
| DeepSeek-V3.1 | 327K | 327K | 61.3% | 61.0% |
| ReAct Agent (Seed-OSS-36B) | 327K | 327K | 47.8% | 55.2% |
| Summary Agent + RL | 32K | 32K x 10 | 52.7% | 55.0% |
| **Folding Agent + FoldGRPO (Ours)** | **32K** | **32K x 10** | **62.0%** | **58.0%** |

Our agent achieves **90%+ context compression** while maintaining or improving task performance.

## How Context-Folding Works

Context-Folding enables agents to actively manage their context through two special actions:

1. **`branch(description, prompt)`**: Create a temporary sub-trajectory to handle a localized subtask. The agent operates in a separate working context while inheriting the main context prefix.

2. **`return(message)`**: Summarize the outcome and rejoin the main thread. The intermediate steps within the branch are "folded" (removed from context), leaving only a concise summary.

```
Main Thread:  [a1, o1] → [branch] → [a5, o8] → [a9, o9] → [finish]
                           ↓           ↑
Branch 1:            [a2, o2, a3, o3, a4, o4] → [return]
                                    (folded)
```

This plan-execution framework allows the agent to:
- **Planning State**: High-level reasoning in the main thread, task decomposition, branch initiation
- **Execution State**: Focused work within a branch on assigned subtasks

## FoldGRPO: End-to-End RL for Context-Folding

FoldGRPO augments standard GRPO with:

1. **Dynamic Folded LLM Contexts**: Maintains compact working context during training by folding rollout history
2. **Dense Process Rewards**: Token-level rewards to guide context folding behavior
   - **Unfolded Token Penalty**: Discourages token-heavy operations in the main context
   - **Out-of-Scope Penalty**: Encourages agents to stay focused within sub-tasks
   - **Failure Penalty**: Penalizes failed tool calls

## Repository Structure

```
verl_context_folding/
├── verl/
│   ├── experimental/
│   │   └── agent_loop/
│   │       └── FoldAgent/           # Context-Folding agent implementation
│   │           ├── agents/
│   │           │   ├── prompts.py   # System prompts and chat templates
│   │           │   ├── tool_spec.py # Tool definitions (search, branch, return, etc.)
│   │           │   ├── utils.py     # Utility functions
│   │           │   └── verifier.py  # Scope verification for branch agents
│   │           ├── envs/
│   │           │   ├── local_search.py   # Local search environment
│   │           │   ├── search_server.py  # Search server for BrowseComp-Plus
│   │           │   └── README.md         # Search server documentation
│   │           └── fold_agent_loop.py    # Main agent loop implementation
│   └── trainer/
│       └── ppo/                     # FoldGRPO training implementation
├── bcp_qwen3_8b.sh                  # Example training script
└── README.md
```

## Environment Setup

### 1. Install VeRL

Follow the [verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html).

### 2. (BrowseComp-Plus) Install Dependencies and Start Search Server

For the deep research task, install dependencies and start the search server before training:

```bash
# For the search environment (BrowseComp-Plus)
pip install fastapi uvicorn httpx huggingface_hub numpy pydantic
```

```bash
cd verl/experimental/agent_loop/FoldAgent/envs
python search_server.py --host 0.0.0.0 --port 8000
```

The server will:
1. Download corpus from HuggingFace (`Tevatron/browsecomp-plus-corpus`)
2. Download pre-computed embeddings (`miaolu3/browsecomp-plus`)
3. Load Qwen3-Embedding-8B model on available GPUs

See [`verl/experimental/agent_loop/FoldAgent/envs/README.md`](verl/experimental/agent_loop/FoldAgent/envs/README.md) for detailed configuration options.

### 3. Environment Variables

```bash
# Required: URL of the local search server (for BrowseComp-Plus)
export LOCAL_SEARCH_URL="http://localhost:8000"

# Optional: For LLM-based answer grading
export OPENAI_API_KEY="your-api-key"
```

## Running the Experiment

### Data Preparation

For the deep research task, ensure that the training and testing files are in the root directory: `bcp_train.parquet` and `bcp_test.parquet`.

### Training with FoldGRPO

Example using the provided script:

```bash
bash bcp_qwen3_8b.sh
```

## Key Configuration Options

### Agent Loop Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent.default_agent_loop` | - | Set to `fold_agent` to use Context-Folding |
| `plugin.workflow` | `search` | Workflow type: `search_branch` for folding agent |
| `plugin.max_turn` | 100 | Maximum turns per rollout |
| `plugin.max_session` | 10 | Maximum number of branches |
| `plugin.branch_len` | 32768 | Maximum tokens per branch |
| `plugin.session_timeout` | 5400 | Session timeout in seconds |

### Process Reward Settings (FoldGRPO)

| Parameter | Description |
|-----------|-------------|
| `plugin.process_reward='[flat,scope]'` | Enable process rewards |
| `flat` | Flat reward assignment to branches |
| `scope` | Out-of-scope penalty for branches |
| `cjk` | CJK character penalty |
| `drop_fail` | Drop failed trajectories |

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout.prompt_length` | 4096 | Maximum prompt length |
| `rollout.response_length` | 32768 | Maximum response length (32K active context) |
| `rollout.n` | 32 | Number of rollout samples |
| `actor.ppo_mini_batch_size` | 8 | PPO mini batch size |
| `trainer.total_training_steps` | 100 | Total training steps |

## Tools Available to the Agent

### Search Environment (BrowseComp-Plus)

- **`search(query, topk)`**: Semantic search over corpus, returns top-k documents
- **`open_page(docid/url)`**: Fetch full document content
- **`finish(answer, explanation)`**: Submit final answer

### Context-Folding Tools

- **`branch(description, prompt)`**: Create sub-agent for focused subtask
- **`return(message)`**: Return results to main agent, fold intermediate steps

## Benchmark: BrowseComp-Plus (Deep Research)

- 680 training instances, 150 evaluation instances
- Uses Qwen3-Embed-8B as retriever
- LLM-based answer grading

## Citation

```bibtex
@article{sun2025scaling,
  title={Scaling Long-Horizon LLM Agent via Context-Folding},
  author={Sun, Weiwei and Lu, Miao and Ling, Zhan and Liu, Kang and Yao, Xuesong and Yang, Yiming and Chen, Jiecao},
  journal={arXiv preprint arXiv:2510.11967},
  year={2025}
}
```

## Acknowledgments

This project is built on [verl](https://github.com/volcengine/verl), a flexible and efficient RL training library for LLMs by ByteDance Seed team.

```bibtex
@article{sheng2024hybridflow,
  title={HybridFlow: A Flexible and Efficient RLHF Framework},
  author={Sheng, Guangming and Zhang, Chi and Ye, Zilingfeng and Wu, Xibin and Zhang, Wang and Zhang, Ru and Peng, Yanghua and Lin, Haibin and Wu, Chuan},
  journal={arXiv preprint arXiv:2409.19256},
  year={2024}
}
```

## Contact

- Weiwei Sun: sunnweiwei@gmail.com
- Miao Lu: lumiao.work@gmail.com
