# Chapter 11: Fine-tune Large Language Models

This chapter covers adapting pre-trained language models to specific tasks using Supervised Fine-Tuning (SFT), including chat templates, efficient training with LoRA, and model evaluation.

---

## Overview

Most LLMs used in production (like ChatGPT) have undergone **Supervised Fine-Tuning** to make them more helpful and aligned with human preferences.

```text
                    ┌─────────────────────────────────────────────┐
                    │         Supervised Fine-Tuning Pipeline      │
                    └─────────────────────────────────────────────┘
                                         │
     ┌──────────────┬────────────────────┼────────────────────┬──────────────┐
     ▼              ▼                    ▼                    ▼              ▼
┌─────────┐   ┌───────────┐   ┌───────────────────┐   ┌───────────┐   ┌───────────┐
│  Chat   │   │  Dataset  │   │  Configure SFT /  │   │  Train    │   │  Evaluate │
│Templates│   │  Prepare  │   │  LoRA Adapter     │   │  Model    │   │  Model    │
└─────────┘   └───────────┘   └───────────────────┘   └───────────┘   └───────────┘
```

---

## 1. Chat Templates

Chat templates structure interactions between users and AI models, ensuring consistent and contextually appropriate responses.

### Base Models vs Instruct Models

| Model Type         | Description                                | Example                  |
| ------------------ | ------------------------------------------ | ------------------------ |
| **Base Model**     | Trained on raw text to predict next token  | `SmolLM2-135M`           |
| **Instruct Model** | Fine-tuned to follow instructions          | `SmolLM2-135M-Instruct`  |

> 💡 Instruct models use a specific chat template format. Always verify the correct template from the model's tokenizer configuration.

### Common Template Formats

Different models use different templates. Here's a sample conversation:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
```

**ChatML Format** (SmolLM2, Qwen 2):

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?<|im_end|>
<|im_start|>user
What's the weather?<|im_end|>
<|im_start|>assistant
```

**Mistral Format**:

```text
<s>[INST] You are a helpful assistant. [/INST] Hi! How can I help you today?</s> [INST] Hello! [/INST]
```

### Applying Chat Templates

The `transformers` library handles template formatting automatically:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

# Apply the model's chat template
formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted_chat)
```

### Advanced Features

Chat templates can handle:

- **Tool Use** - Interacting with external APIs
- **Multimodal Inputs** - Images, audio, etc.
- **Function Calling** - Structured function execution

```python
# Multimodal example
messages = [
    {"role": "system", "content": "You are a helpful vision assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image_url": "https://example.com/image.jpg"},
        ],
    },
]
```

---

## 2. Supervised Fine-Tuning with TRL

### When to Use SFT

Consider SFT only if:

- Prompting existing models is insufficient
- You need precise output formatting
- You require domain-specific knowledge
- The cost of a smaller fine-tuned model beats using a larger general model

### Dataset Preparation

SFT requires input-output pairs. Use datasets with `messages` field for chat-style training:

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceTB/smoltalk", "all")
# Dataset has 'messages' field with role/content structure
```

### Training Configuration

Key parameters for `SFTConfig`:

| Parameter                       | Description              | Recommended   |
| ------------------------------- | ------------------------ | ------------- |
| `num_train_epochs`              | Total training passes    | 1-3           |
| `per_device_train_batch_size`   | Batch size per GPU       | 2-8           |
| `gradient_accumulation_steps`   | Steps before update      | 4-16          |
| `learning_rate`                 | Weight update size       | 2e-5 to 2e-4  |
| `warmup_ratio`                  | LR warmup portion        | 0.03-0.1      |
| `max_length`                    | Maximum sequence length  | 512-2048      |

### Implementation

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.extras.chat_template import setup_chat_format

# Load model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Setup chat format
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Load dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# Configure training
training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=50,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

# Train
trainer.train()

# Save model
trainer.save_model("./fine_tuned_model")
```

### Dataset Packing

Packing combines multiple short examples into one sequence for efficiency:

```python
# Enable packing in SFTConfig
training_args = SFTConfig(
    output_dir="./sft_output",
    packing=True,  # Enable packing
    eval_packing=False,  # Disable for evaluation
)
```

Custom formatting function for specific dataset structures:

```python
def formatting_func(example):
    return f"### Question: {example['question']}\n### Answer: {example['answer']}"

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)
```

### Monitoring Training

**Loss Patterns:**

1. **Initial Sharp Drop** - Rapid adaptation
2. **Gradual Stabilization** - Fine-tuning progress
3. **Convergence** - Training completion

**Warning Signs:**

| Issue        | Sign                           | Solution                |
| ------------ | ------------------------------ | ----------------------- |
| Overfitting  | Val loss ↑ while train loss ↓  | Reduce epochs, add data |
| Underfitting | No loss improvement            | Increase LR, check data |
| Memorization | Extremely low loss             | More diverse data       |

---

## 3. LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning technique that adds small trainable matrices instead of updating all model weights.

### Key Benefits

| Benefit                  | Description                          |
| ------------------------ | ------------------------------------ |
| **Memory Efficient**     | Only adapter params in GPU memory    |
| **Fast Training**        | 10,000x fewer trainable parameters   |
| **No Latency Overhead**  | Adapters merge with base model       |
| **Multiple Adapters**    | Switch between tasks easily          |

### LoRA Configuration

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=8,                      # Rank dimension (4-32)
    lora_alpha=16,            # Scaling factor (typically 2x rank)
    lora_dropout=0.05,        # Dropout for regularization
    bias="none",              # Bias training type
    target_modules="all-linear",  # Which modules to adapt
    task_type="CAUSAL_LM",    # Task type
)
```

### SFT with LoRA

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Load model with 4-bit quantization (QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    load_in_4bit=True,  # QLoRA: 4-bit quantization
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# Training config
training_args = SFTConfig(
    output_dir="./Llama-3.1-8B-LoRA",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,  # Higher LR for LoRA
    gradient_checkpointing=True,
    max_length=2048,
    logging_steps=10,
)

# Load dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# Train with LoRA
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,  # Pass LoRA config here
    processing_class=tokenizer,
)
trainer.train()

# Save only the adapter (lightweight)
trainer.model.save_pretrained("./Llama-3.1-8B-LoRA-adapter")
```

### Loading and Using LoRA Adapters

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

# Load config and base model
config = PeftConfig.from_pretrained("your-username/model-lora")
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

# Load adapter
model = PeftModel.from_pretrained(base_model, "your-username/model-lora")

# Switch adapters
model.load_adapter("path/to/another-adapter", adapter_name="task2")
model.set_adapter("task2")
```

### Merging LoRA Adapters

Merge adapters back into base model for deployment:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# Merge and unload
merged_model = peft_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("path/to/merged_model")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
tokenizer.save_pretrained("path/to/merged_model")
```

---

## 4. Evaluation

### Automatic Benchmarks

| Benchmark        | Type              | Description                             |
| ---------------- | ----------------- | --------------------------------------- |
| **MMLU**         | General Knowledge | 57 subjects from science to humanities  |
| **TruthfulQA**   | Truthfulness      | Tests for common misconceptions         |
| **BBH**          | Reasoning         | Complex logical thinking tasks          |
| **GSM8K**        | Math              | Grade school math problems              |
| **HumanEval**    | Coding            | 164 Python programming problems         |
| **HELM**         | Holistic          | Comprehensive multi-aspect evaluation   |
| **Alpaca Eval**  | Chat              | GPT-4 judged instruction following      |

### Alternative Approaches

| Method              | Description                         |
| ------------------- | ----------------------------------- |
| **LLM-as-Judge**    | Use one LLM to evaluate another     |
| **Chatbot Arena**   | Crowdsourced head-to-head battles   |
| **Custom Suites**   | Domain-specific test sets           |

### Evaluation with LightEval

```bash
pip install lighteval
```

Task format: `{suite}|{task}|{num_few_shot}|{auto_reduce}`

```bash
lighteval accelerate \
    "pretrained=your-model-name" \
    "mmlu|anatomy|0|0" \
    "mmlu|high_school_biology|0|0" \
    "mmlu|professional_medicine|0|0" \
    --max_samples 40 \
    --batch_size 1 \
    --output_path "./results"
```

Example output:

```text
| Task                                  |Metric| Value |Stderr|
|---------------------------------------|------|------:|-----:|
| mmlu:anatomy                          | acc  | 0.4500|0.1141|
| mmlu:high_school_biology              | acc  | 0.1500|0.0819|
```

### Custom Evaluation Strategy

1. **Baseline** - Run standard benchmarks for comparison
2. **Domain-specific** - Create evaluation sets for your use case
3. **Multi-layered approach**:
   - Automated metrics (quick feedback)
   - Human evaluation (nuanced understanding)
   - Domain expert review (specialized validation)
   - A/B testing (real-world performance)

---

## Quick Reference

### Complete SFT + LoRA Workflow

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# 1. Load model (with quantization for memory efficiency)
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-360M",
    load_in_4bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")

# 2. Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# 3. Configure training
training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    logging_steps=10,
)

# 4. Load dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")

# 5. Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)
trainer.train()

# 6. Save adapter
trainer.model.save_pretrained("./my-lora-adapter")
tokenizer.save_pretrained("./my-lora-adapter")
```

### Key Commands

```bash
# Install dependencies
pip install transformers trl peft datasets accelerate bitsandbytes

# Run SFT with LoRA via CLI
python -m trl.scripts.sft \
    --model_name_or_path HuggingFaceTB/SmolLM2-360M \
    --dataset_name HuggingFaceTB/smoltalk \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16 \
    --output_dir ./output \
    --push_to_hub

# Evaluate with LightEval
lighteval accelerate "pretrained=your-model" "mmlu|anatomy|0|0"
```

### Resources

| Resource                 | URL                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| **TRL Documentation**    | [huggingface.co/docs/trl](https://huggingface.co/docs/trl)                                       |
| **PEFT Documentation**   | [huggingface.co/docs/peft](https://huggingface.co/docs/peft)                                     |
| **Chat Templates Guide** | [transformers chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating) |
| **LightEval**            | [github.com/huggingface/lighteval](https://github.com/huggingface/lighteval)                     |
| **LoRA Paper**           | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)                                     |
| **Chatbot Arena**        | [lmarena.ai](https://lmarena.ai/)                                                                |
