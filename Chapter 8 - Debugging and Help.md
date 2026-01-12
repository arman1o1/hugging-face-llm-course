# Chapter 8: Debugging and Getting Help

This chapter covers debugging techniques and how to effectively get help when things go wrong.

---

## Overview

Chapter 8 teaches you how to troubleshoot problems and engage with the community:

```text
                    ┌─────────────────────────┐
                    │    Debugging Journey    │
                    └─────────────────────────┘
                               │
    ┌──────────────┬───────────┼───────────┬──────────────┐
    ▼              ▼           ▼           ▼              ▼
┌────────┐   ┌──────────┐  ┌───────┐  ┌────────┐   ┌──────────┐
│ Error  │   │ Pipeline │  │Forum  │  │Training│   │ GitHub   │
│ Basics │   │ Debugging│  │ Help  │  │Pipeline│   │ Issues   │
└────────┘   └──────────┘  └───────┘  └────────┘   └──────────┘
```

**What you'll learn:**

- Reading and understanding Python tracebacks
- Debugging the `pipeline` and model forward pass
- Asking effective questions on forums
- Debugging training pipelines systematically
- Writing good GitHub issues

> **Key Insight:** These debugging skills apply to most open source projects, not just Hugging Face!

---

## 1. Understanding Python Tracebacks

When an error occurs, Python displays a **traceback** (stack trace). Read it **from bottom to top**.

### Anatomy of a Traceback

```text
Traceback (most recent call last):
  File "script.py", line 10, in <module>
    result = process_data(data)
  File "script.py", line 5, in process_data
    return model(**inputs)
  ...
OSError: Can't load config for 'model-name'  ← Start here!
```

| Section | What It Tells You |
| ------- | ----------------- |
| **Last line** | Exception type + error message |
| **Arrow lines (`-->`)** | Exact line causing the error |
| **Middle frames** | Call sequence that led to error |

### Debug Strategy

1. **Read the last line first** - it contains the exception type and message
2. **Work your way up** - find where in your code the error originated
3. **Search online** - copy the error message to Google/Stack Overflow
4. **Use `huggingface_hub`** - tools to inspect repos on the Hub

---

## 2. Debugging the Pipeline

### Common Pipeline Errors

#### Example: Model not found

```python
from transformers import pipeline

# Typo in model name: "distillbert" should be "distilbert"
reader = pipeline("question-answering", model="distillbert-base-uncased")
```

```text
OSError: Can't load config for 'distillbert-base-uncased'.
Make sure that:
- 'distillbert-base-uncased' is a correct model identifier...
```

**Fix:** Check for typos in model names!

### Inspecting Repository Contents

```python
from huggingface_hub import list_repo_files

# Check what files exist in a repo
files = list_repo_files(repo_id="username/model-name")
print(files)
# ['README.md', 'pytorch_model.bin', 'vocab.txt', ...]
```

> **Missing `config.json`?** Download from the base model and push it:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("distilbert-base-uncased")
config.push_to_hub("your-model-repo", commit_message="Add config.json")
```

---

## 3. Debugging the Forward Pass

### Common Issue: Wrong Input Type

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Without return_tensors - returns Python lists (wrong!)
inputs = tokenizer("Hello world")
outputs = model(**inputs)  # AttributeError!
```

```text
AttributeError: 'list' object has no attribute 'size'
```

**Fix:** Always specify `return_tensors`:

```python
# Correct: Returns PyTorch tensors
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)  # Works!
```

### Input Shape Issues

```python
# Too many tokens for model's max_length
long_text = "..." * 600  # 600+ tokens
inputs = tokenizer(long_text, return_tensors="pt")
# IndexError: index out of range in self
```

**Fix:** Use truncation:

```python
inputs = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=512)
```

---

## 4. Asking for Help on Forums

The [Hugging Face Forums](https://discuss.huggingface.co) are a great resource. Write effective posts:

### Good Forum Post Checklist

| Element | Bad Example | Good Example |
| ------- | ----------- | ------------ |
| **Title** | "Help me please!" | "IndexError in AutoModel forward pass with long text" |
| **Code** | Plain text, no formatting | Wrapped in triple backticks |
| **Traceback** | Only last line | Full traceback, formatted |
| **Reproducibility** | References local files | Self-contained example |
| **Tone** | Demanding, tags many people | Polite, professional |

### Format Code Properly

````text
Use triple backticks with language:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
# Your code here
```
````

### Provide Reproducible Examples

```python
# Bad: References your local data
inputs = tokenizer(my_dataset[0]["text"])  # Others can't run this

# Good: Self-contained example
inputs = tokenizer("This is a sample text for testing.", return_tensors="pt")
```

---

## 5. Debugging the Training Pipeline

### The Training Pipeline Flow

```text
Dataset → DataLoader → Model Forward → Loss → Backward → Optimizer
    ↓          ↓            ↓           ↓        ↓          ↓
 Check 1    Check 2     Check 3      Check 4  Check 5    Check 6
```

### Systematic Debugging Steps

#### Step 1: Check Your Data

```python
# Verify training data exists and is correct
print(trainer.train_dataset[0])

# Common mistake: using raw_datasets instead of tokenized_datasets
trainer = Trainer(
    model=model,
    train_dataset=tokenized_datasets["train"],  # NOT raw_datasets!
    ...
)
```

**Always decode and verify:**

```python
# Check that tokenization worked correctly
sample = trainer.train_dataset[0]
print(tokenizer.decode(sample["input_ids"]))
# Should show: '[CLS] your text here [SEP]'
```

#### Step 2: Check Data Collation

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Test collator with a small batch
samples = [trainer.train_dataset[i] for i in range(4)]
batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})
```

**Collation errors usually mean:**

- Missing padding/truncation
- Inconsistent sequence lengths
- Wrong data types

#### Step 3: Test Model Forward Pass

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = {k: v.to(device) for k, v in batch.items()}

# Test forward pass
outputs = trainer.model.to(device)(**batch)
print(f"Loss: {outputs.loss.item()}")
```

#### Step 4: Test Optimization Step

```python
# Test backward pass
loss = outputs.loss
loss.backward()

# Test optimizer step
trainer.create_optimizer()
trainer.optimizer.step()
```

---

## 6. Handling CUDA Out of Memory

```text
RuntimeError: CUDA out of memory
```

### Quick Fixes

| Solution | How to Apply |
| -------- | ------------ |
| **Reduce batch size** | `per_device_train_batch_size=4` → `2` |
| **Use gradient accumulation** | `gradient_accumulation_steps=4` |
| **Use smaller model** | `bert-base` → `distilbert-base` |
| **Enable mixed precision** | `fp16=True` |
| **Clear GPU memory** | Restart kernel, `torch.cuda.empty_cache()` |

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,       # Reduce if OOM
    gradient_accumulation_steps=4,       # Effective batch = 16
    fp16=True,                           # Half precision
)
```

---

## 7. Debugging Silent Errors

When training runs but the model doesn't learn:

### Sanity Checks

1. **Verify labels are correct:**

   ```python
   # Check label distribution
   from collections import Counter
   labels = [ex["label"] for ex in trainer.train_dataset]
   print(Counter(labels))
   # If one label dominates, model may just predict that
   ```

2. **Check expected loss for random predictions:**

   ```python
   import math
   num_classes = 3
   random_loss = -math.log(1 / num_classes)
   print(f"Random baseline loss: {random_loss:.2f}")  # ~1.1 for 3 classes
   ```

3. **Verify decoded predictions make sense:**

   ```python
   with torch.no_grad():
       outputs = model(**batch)
       preds = torch.argmax(outputs.logits, dim=-1)
       print(f"Predictions: {preds}")
       print(f"Labels: {batch['labels']}")
   ```

### The Overfit Test

**Can your model memorize a single batch?** If not, something is fundamentally wrong.

```python
# Get one batch
for batch in trainer.get_train_dataloader():
    break

batch = {k: v.to(device) for k, v in batch.items()}
trainer.create_optimizer()

# Train only on this batch for 20 steps
for _ in range(20):
    outputs = trainer.model(**batch)
    loss = outputs.loss
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    print(f"Loss: {loss.item():.4f}")

# Check accuracy (should be ~100%)
with torch.no_grad():
    outputs = trainer.model(**batch)
    preds = torch.argmax(outputs.logits, dim=-1)
    accuracy = (preds == batch["labels"]).float().mean()
    print(f"Accuracy: {accuracy:.2%}")  # Should be ~100%
```

> ⚠️ **After the overfit test**, recreate your model before training on full data!

---

## 8. Writing Good GitHub Issues

When you've found a real bug, report it effectively:

### Creating Minimal Reproducible Examples

```python
# Bad: Depends on your local data
from my_project import load_my_data
data = load_my_data()  # Others can't run this

# Good: Self-contained
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Minimal code that reproduces the issue
text = "A" * 600  # Simple way to create long text
inputs = tokenizer(text, return_tensors="pt")
```

### Include Environment Information

```bash
transformers-cli env
```

Output:

```text
- transformers version: 4.36.0
- Platform: Linux-5.15.0-x86_64
- Python version: 3.10.12
- PyTorch version (GPU?): 2.1.0 (True)
- Using GPU in script?: Yes
```

### Issue Template Checklist

- [ ] Descriptive title
- [ ] Environment info (`transformers-cli env`)
- [ ] Minimal reproducible example
- [ ] Full traceback (formatted)
- [ ] Expected vs actual behavior
- [ ] What you've already tried

### Tagging Guidelines

- Tag only people who last modified relevant code
- Never tag more than 3 people
- Be polite - maintainers are volunteers!

---

## Quick Reference

### Debugging Workflow

```text
1. Read traceback (bottom to top)
        ↓
2. Identify error type and location
        ↓
3. Search online for similar issues
        ↓
4. Try minimal reproducible example
        ↓
5. Ask on forums (with good post)
        ↓
6. File GitHub issue (if confirmed bug)
```

### Common Errors and Fixes

| Error | Likely Cause | Quick Fix |
| ----- | ------------ | --------- |
| `OSError: Can't load config` | Wrong model ID / missing file | Check model name, verify repo contents |
| `'list' has no attribute 'size'` | Missing `return_tensors` | Add `return_tensors="pt"` |
| `IndexError: index out of range` | Input too long | Add `truncation=True, max_length=512` |
| `CUDA out of memory` | Batch too large | Reduce batch size, use `fp16=True` |
| `input_ids missing` | Wrong dataset passed | Use `tokenized_datasets` not `raw_datasets` |
| `expected sequence of length X` | Padding issue | Use `DataCollatorWithPadding` |

### Useful Commands

```python
# Check repository files
from huggingface_hub import list_repo_files
list_repo_files("model-id")

# Get environment info
# In terminal: transformers-cli env
# In notebook: !transformers-cli env

# Decode tokenized input
tokenizer.decode(inputs["input_ids"][0])

# Check GPU memory
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Resources

| Resource | URL |
| -------- | --- |
| **HF Forums** | [discuss.huggingface.co](https://discuss.huggingface.co) |
| **Transformers Issues** | [github.com/huggingface/transformers/issues](https://github.com/huggingface/transformers/issues) |
| **Stack Overflow** | Search with error message |
| **Code of Conduct** | [Transformers Code of Conduct](https://github.com/huggingface/transformers/blob/master/CODE_OF_CONDUCT.md) |
