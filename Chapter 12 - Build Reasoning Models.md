# Chapter 12: Build Reasoning Models

This chapter explores reinforcement learning for LLMs, the DeepSeek R1 paper, and implementing GRPO (Group Relative Policy Optimization) to train reasoning models.

---

## Overview

LLMs are excellent at generating fluent text but traditionally struggle with complex reasoning tasks. **Open R1** is a community project that uses reinforcement learning to make LLMs "think" and reason.

```text
                    ┌─────────────────────────────────────────────┐
                    │           Open R1 Training Pipeline          │
                    └─────────────────────────────────────────────┘
                                         │
     ┌──────────────┬────────────────────┼────────────────────┬──────────────┐
     ▼              ▼                    ▼                    ▼              ▼
┌─────────┐   ┌───────────┐   ┌───────────────────┐   ┌───────────┐   ┌───────────┐
│   RL    │   │ DeepSeek  │   │   GRPO Training   │   │  Reward   │   │  Deploy   │
│ Basics  │   │ R1 Paper  │   │   with TRL        │   │  Design   │   │  Model    │
└─────────┘   └───────────┘   └───────────────────┘   └───────────┘   └───────────┘
```

### Reasoning Format

Models trained with Open R1 generate structured outputs:

```text
<think>I need to add the number of apples and oranges to get the total.</think>
5
```

This allows separating the **reasoning process** from the **final answer**.

---

## 1. Reinforcement Learning Basics

Think of RL like training a dog - rewarding good behavior and discouraging bad behavior through feedback.

### Key Concepts

| Concept         | Dog Training           | LLM Training                          |
| --------------- | ---------------------- | ------------------------------------- |
| **Agent**       | The dog                | The LLM                               |
| **Environment** | Your house + you       | Users or simulated scenarios          |
| **Action**      | Sit, stand, bark       | Generate words/responses              |
| **Reward**      | Treats and praise      | Helpfulness/correctness scores        |
| **Policy**      | Dog's learned behavior | Model's strategy for choosing actions |

### Why RL for LLMs?

Pre-trained models can:

- ✅ Generate fluent text
- ✅ Predict next tokens well

But they may still:

- ❌ Produce incorrect information
- ❌ Generate harmful content
- ❌ Miss user intent

**RL helps align LLMs to be helpful, harmless, and honest.**

---

## 2. RLHF (Reinforcement Learning from Human Feedback)

### The Process

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1. Collect      │ -> │ 2. Train Reward │ -> │ 3. Fine-tune    │
│ Human Prefs     │    │    Model        │    │    LLM with RL  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

1. **Collect Preferences**: Humans compare LLM responses and indicate which is better
2. **Train Reward Model**: Learn to predict human preferences
3. **Fine-tune LLM**: Use reward model to guide LLM training

### Popular RLHF Techniques

| Method   | Description                                          | Complexity |
| -------- | ---------------------------------------------------- | ---------- |
| **PPO**  | Policy gradient with separate reward model           | High       |
| **DPO**  | Direct preference optimization without reward model  | Medium     |
| **GRPO** | Group-based comparison with flexible rewards         | Medium     |

---

## 3. Why GRPO?

**GRPO (Group Relative Policy Optimization)** is the focus of this chapter because:

| Advantage                  | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| **Flexible Rewards**       | Works with any reward function, not just preference data  |
| **No Reward Model Needed** | Can use rule-based or function-based rewards              |
| **Stable Training**        | Group-based normalization provides stable gradients       |
| **Simple Implementation**  | Easier to implement than PPO                              |

Example reward sources:

- Length functions (shorter/longer responses)
- Math solvers (correctness verification)
- Format checkers (proper structure)

---

## 4. Understanding DeepSeek R1

### The "Aha Moment"

DeepSeek R1 discovered that models can self-correct during reasoning:

1. **Initial Attempt**: Makes first try at solving
2. **Recognition**: Identifies potential errors
3. **Self-Correction**: Adjusts approach
4. **Explanation**: Explains why new approach is better

This emerged naturally from RL training without explicit programming!

### Training Process

DeepSeek R1 uses a 4-phase training process:

| Phase                 | Purpose                  | Innovation                           |
| --------------------- | ------------------------ | ------------------------------------ |
| 1. Cold Start         | Build quality foundation | Small, high-quality dataset          |
| 2. Reasoning RL       | Develop core reasoning   | Rule-based rewards, verifiable tasks |
| 3. Rejection Sampling | Quality control          | LLM as quality judge                 |
| 4. Diverse RL         | Broad alignment          | Hybrid reward approach               |

**Two resulting models:**

- **DeepSeek-R1-Zero**: Pure RL training
- **DeepSeek-R1**: RL + supervised fine-tuning

---

## 5. GRPO Algorithm

### How It Works

```text
                    For each prompt
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐       ┌─────────────────┐
     │ Generate 4-16   │       │ Group-based     │
     │ completions     │  -->  │ comparison      │
     └─────────────────┘       └─────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
           ┌─────────────────┐                   ┌─────────────────┐
           │ Compute rewards │                   │ Normalize within│
           │ for each        │                   │ group           │
           └─────────────────┘                   └─────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │ Update policy   │
                              │ toward better   │
                              │ completions     │
                              └─────────────────┘
```

### Group Advantage Formula

```python
advantage = (reward - mean(group_rewards)) / std(group_rewards)
```

This normalization is like "grading on a curve" - comparing solutions relative to their peers.

### Algorithm Pseudocode

```text
Input:
  - initial_policy: Starting model
  - reward_function: Evaluates outputs
  - training_prompts: Training data
  - group_size: Completions per prompt (4-16)

For each training iteration:
    1. Snapshot current policy as reference
    2. For each prompt:
        a. Generate group_size completions
        b. Compute rewards for each
        c. Normalize: advantage = (reward - mean) / std
        d. Update policy to favor high-advantage completions
        e. Apply KL penalty to prevent drastic changes
    
Output: Optimized policy
```

---

## 6. Implementing GRPO in TRL

### Installation

```bash
pip install trl peft transformers datasets accelerate bitsandbytes
```

### Dataset Format

Your dataset needs a `prompt` column:

```python
from datasets import load_dataset

# Load a dataset with prompts
dataset = load_dataset("trl-lib/tldr", split="train")
```

### Reward Functions

Reward functions determine what the model learns. They receive completions and return scores.

**Length-based reward:**

```python
def reward_length(completions, **kwargs):
    """Reward completions close to ideal length."""
    ideal_length = 50
    return [-abs(ideal_length - len(c)) for c in completions]
```

**Format-based reward:**

```python
import re

def reward_format(completions, **kwargs):
    """Reward proper think-then-answer format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [1.0 if re.search(pattern, c, re.DOTALL) else 0.0 for c in completions]
```

**Correctness reward (for verifiable tasks):**

```python
def reward_correctness(completions, answers, **kwargs):
    """Reward correct answers for math problems."""
    rewards = []
    for completion, correct in zip(completions, answers):
        try:
            extracted = extract_answer(completion)  # Your parsing logic
            rewards.append(1.0 if extracted == correct else 0.0)
        except:
            rewards.append(0.0)
    return rewards
```

### Basic GRPO Training

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Load dataset
dataset = load_dataset("trl-lib/tldr", split="train")

# Define reward function
def reward_unique_chars(completions, **kwargs):
    """Reward completions with more unique characters."""
    return [len(set(c)) for c in completions]

# Configure training
training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_generations=8,              # Group size
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
)

# Initialize trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_unique_chars,
    args=training_args,
    train_dataset=dataset,
)

# Train
trainer.train()
```

### GRPO with LoRA

For memory-efficient training on larger models:

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Dataset
dataset = load_dataset("mlabonne/smoltldr", split="train")

# LoRA config
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)

# Training config
training_args = GRPOConfig(
    output_dir="./grpo_lora",
    num_generations=8,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    max_prompt_length=512,
    max_completion_length=96,
    num_train_epochs=1,
    bf16=True,
    logging_steps=1,
)

# Reward function
ideal_length = 50
def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(c)) for c in completions]

# Initialize trainer with LoRA
trainer = GRPOTrainer(
    model="HuggingFaceTB/SmolLM-135M-Instruct",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()

# Save merged model
merged = trainer.model.merge_and_unload()
merged.push_to_hub("your-username/SmolGRPO")
```

### Configuration Parameters

| Parameter                     | Description                        | Recommended         |
| ----------------------------- | ---------------------------------- | ------------------- |
| `num_generations`             | Completions per prompt (group size)| 4-16                |
| `per_device_train_batch_size` | Batch size per GPU                 | 4-8                 |
| `gradient_accumulation_steps` | Steps before weight update         | 2-4                 |
| `learning_rate`               | Learning rate                      | 1e-5 to 5e-5        |
| `max_prompt_length`           | Max input tokens                   | 512                 |
| `max_completion_length`       | Max output tokens                  | 96-256              |
| `use_vllm`                    | Enable vLLM for faster generation  | True (if supported) |

---

## 7. Monitoring Training

### Key Metrics

| Metric       | Description                    | Expected Behavior                |
| ------------ | ------------------------------ | -------------------------------- |
| `reward`     | Average reward                 | Should increase                  |
| `reward_std` | Reward standard deviation      | May decrease as model converges  |
| `kl`         | KL divergence from reference   | Should stay bounded              |
| `loss`       | Policy loss                    | May increase (this is normal!)   |

> 💡 **Note**: GRPO loss increasing is expected! It reflects the KL divergence as the model diverges from its initial policy while learning.

### Tips for Success

1. **Memory Management**: Reduce `per_device_train_batch_size` if OOM
2. **Speed**: Enable `use_vllm=True` for faster generation
3. **Group Size**: Start with 8, increase for complex reasoning tasks
4. **Reward Design**: Ensure rewards are well-calibrated and distinguishable

---

## 8. GRPO Limitations

| Limitation            | Description                                       |
| --------------------- | ------------------------------------------------- |
| **Generation Cost**   | Multiple completions (4-16) per prompt            |
| **Batch Constraints** | Group processing can limit effective batch size   |
| **Reward Design**     | Quality depends heavily on reward function        |
| **KL Tuning**         | Finding right KL penalty balance requires tuning  |

---

## Quick Reference

### Complete GRPO Training Script

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
import re

# 1. Load dataset
dataset = load_dataset("your-dataset", split="train")

# 2. Define reward functions
def format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    return [1.0 if re.search(pattern, c, re.DOTALL) else 0.0 for c in completions]

def length_reward(completions, **kwargs):
    return [-abs(100 - len(c)) for c in completions]

# 3. Configure LoRA (optional)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)

# 4. Configure training
training_args = GRPOConfig(
    output_dir="./grpo_model",
    num_generations=8,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    max_prompt_length=512,
    max_completion_length=128,
    bf16=True,
    logging_steps=10,
    gradient_checkpointing=True,
)

# 5. Initialize and train
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[format_reward, length_reward],  # Multiple rewards
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()

# 6. Save
trainer.model.save_pretrained("./grpo_adapter")
```

### Key Commands

```bash
# Install dependencies
pip install trl peft transformers datasets accelerate bitsandbytes

# With flash attention (optional, for speed)
pip install flash-attn --no-build-isolation
```

### Resources

| Resource               | URL                                                                       |
| ---------------------- | ------------------------------------------------------------------------- |
| **TRL Documentation**  | [huggingface.co/docs/trl](https://huggingface.co/docs/trl)                |
| **GRPO Trainer Docs**  | [GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)       |
| **Open R1 Project**    | [github.com/huggingface/open-r1](https://github.com/huggingface/open-r1)  |
| **DeepSeek R1 Paper**  | [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)              |
| **Hugging Chat (R1)**  | [huggingface.co/chat](https://huggingface.co/chat)                        |
| **PEFT Documentation** | [huggingface.co/docs/peft](https://huggingface.co/docs/peft)              |
