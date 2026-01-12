# Chapter 3: Fine-tuning

This chapter covers how to fine-tune pretrained models on your own datasets.

---

## Overview

Fine-tuning adapts a pretrained model to a specific task using your data.

```text
Pretrained Model → [Fine-tuning on Your Data] → Task-Specific Model
```

**What you'll learn:**

- Load and preprocess datasets from the Hub
- Fine-tune using the `Trainer` API
- Implement custom training loops with PyTorch
- Use `🤗 Accelerate` for distributed training

---

## Processing the Data

### Loading Datasets from the Hub

```python
from datasets import load_dataset

# Load MRPC (paraphrase detection dataset)
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
```

Output:

```python
DatasetDict({
    train: Dataset({features: ['sentence1', 'sentence2', 'label', 'idx'], num_rows: 3668})
    validation: Dataset({features: [...], num_rows: 408})
    test: Dataset({features: [...], num_rows: 1725})
})
```

Access samples:

```python
raw_datasets["train"][0]

# {'idx': 0, 
#  'label': 1, 
#  'sentence1': 'Amrozi accused his brother...', 
#  'sentence2': 'Referring to him as only...'}
```

---

### Tokenizing Sentence Pairs

For tasks with sentence pairs, pass both sentences to the tokenizer:

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Tokenize a single pair
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
```

Output:

```python
{
    'input_ids': [101, 2023, 2003, ..., 102, 2023, 2003, ..., 102],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```

| Key             | Purpose                                 |
| --------------- | --------------------------------------- |
| `input_ids`     | Token IDs for both sentences            |
| `token_type_ids`| 0 for first sentence, 1 for second      |
| `attention_mask`| Which tokens to attend to               |

Token structure: `[CLS] sentence1 [SEP] sentence2 [SEP]`

---

### Using Dataset.map()

Apply tokenization to the entire dataset efficiently:

```python
def tokenize_function(example):
    return tokenizer(
        example["sentence1"], 
        example["sentence2"], 
        truncation=True
    )

# Apply to all splits at once (batched=True for speed)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

> **Note:** We skip `padding` here. Dynamic padding is more efficient!

---

### Dynamic Padding

Pad each batch to its maximum length (not the entire dataset's maximum):

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Test it
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})

# {'attention_mask': torch.Size([8, 67]),
#  'input_ids': torch.Size([8, 67]),
#  'token_type_ids': torch.Size([8, 67]),
#  'labels': torch.Size([8])}
```

---

## Fine-tuning with the Trainer API

### Basic Setup

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)

# Load and tokenize data
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model (num_labels=2 for binary classification)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```

### Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",        # Evaluate every epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)
```

### Define Trainer and Train

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

trainer.train()
```

---

### Adding Evaluation Metrics

```python
import numpy as np
import evaluate

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Create trainer with metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,  # Add this
)
```

---

### Advanced Training Features

| Feature                     | How to Enable                    |
| --------------------------- | -------------------------------- |
| **Mixed Precision**         | `fp16=True`                      |
| **Gradient Accumulation**   | `gradient_accumulation_steps=4`  |
| **Learning Rate Scheduler** | `lr_scheduler_type="cosine"`     |
| **Push to Hub**             | `push_to_hub=True`               |

Example with advanced options:

```python
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    fp16=True,                          # Mixed precision
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,      # Effective batch = 16
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
)
```

---

## Custom Training Loop

### Prepare Data for PyTorch

```python
from torch.utils.data import DataLoader

# Prepare dataset
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Create DataLoaders
train_dataloader = DataLoader(
    tokenized_datasets["train"], 
    shuffle=True, 
    batch_size=8, 
    collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], 
    batch_size=8, 
    collate_fn=data_collator
)
```

---

### Training Loop

```python
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# Setup
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Move to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

---

### Evaluation Loop

```python
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
# {'accuracy': 0.843, 'f1': 0.890}
```

---

## Distributed Training with 🤗 Accelerate

**Accelerate** simplifies multi-GPU/TPU training with minimal code changes:

```python
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# Initialize Accelerator
accelerator = Accelerator()

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

# Prepare for distributed training (handles device placement)
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, 
                             num_warmup_steps=0, 
                             num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    for batch in train_dl:
        outputs = model(**batch)
        loss = outputs.loss
        
        accelerator.backward(loss)  # Use accelerator instead of loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

**Key changes with Accelerate:**

- No need for `.to(device)` - Accelerator handles it
- Use `accelerator.backward(loss)` instead of `loss.backward()`
- Use `accelerator.prepare()` to wrap objects

**Running distributed training:**

```bash
# Configure your setup
accelerate config

# Launch training
accelerate launch train.py
```

---

## Understanding Learning Curves

### Loss Curves

| Pattern             | Meaning                                    | Action                           |
| ------------------- | ------------------------------------------ | -------------------------------- |
| Decreasing smoothly | Healthy training                           | Continue training                |
| Stuck high          | Model not learning                         | Check learning rate, data        |
| Diverging (going up)| Learning rate too high                     | Lower learning rate              |
| Train ↓, Val ↑      | Overfitting                                | Add regularization, early stop   |

### Accuracy Curves

- Should generally increase over time
- Often "steppy" (discrete jumps) rather than smooth
- May plateau when model confidence improves without changing predictions

### Healthy vs Unhealthy Training

**Healthy Training:**

- Training loss decreases
- Validation loss follows training loss
- Gap between train/val stays small

**Overfitting:**

- Training loss keeps decreasing
- Validation loss starts increasing
- Growing gap between train/val

**Underfitting:**

- Both losses remain high
- Model fails to learn patterns

---

## Quick Reference

### Complete Fine-tuning with Trainer

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
import evaluate
import numpy as np

# 1. Load data
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Tokenize
def tokenize_fn(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

tokenized = dataset.map(tokenize_fn, batched=True)

# 3. Define metrics
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 4. Train
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

trainer = Trainer(
    model=model,
    args=TrainingArguments("./output", eval_strategy="epoch", num_train_epochs=3),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Key Classes

| Class                           | Purpose                              |
| ------------------------------- | ------------------------------------ |
| `load_dataset()`                | Load datasets from Hub               |
| `Dataset.map()`                 | Apply preprocessing                  |
| `DataCollatorWithPadding`       | Dynamic batch padding                |
| `TrainingArguments`             | Training configuration               |
| `Trainer`                       | High-level training API              |
| `evaluate.load()`               | Load evaluation metrics              |
| `Accelerator`                   | Distributed training                 |

### Common TrainingArguments

| Argument                       | Default | Purpose                      |
| ------------------------------ | ------- | ---------------------------- |
| `output_dir`                   | Required| Save directory               |
| `num_train_epochs`             | 3       | Number of epochs             |
| `per_device_train_batch_size`  | 8       | Batch size per GPU           |
| `learning_rate`                | 5e-5    | Initial learning rate        |
| `eval_strategy`                | "no"    | "epoch", "steps", or "no"    |
| `fp16`                         | False   | Mixed precision training     |
| `gradient_accumulation_steps`  | 1       | Effective batch multiplier   |
| `push_to_hub`                  | False   | Auto-push to Hub             |
