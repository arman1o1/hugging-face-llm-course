# Chapter 2: Using Transformers

This chapter covers how to use models and tokenizers behind the `pipeline()` function.

---

## Behind the Pipeline

The `pipeline()` function has 3 steps:

```text
Text → [Tokenizer] → Numbers → [Model] → Logits → [Post-process] → Predictions
```

---

## Step 1: Tokenizer (Preprocessing)

Convert text to numbers the model understands:

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

Output:

```python
{
    'input_ids': tensor([[101, 1045, ...], [101, 1045, ...]]),
    'attention_mask': tensor([[1, 1, ...], [1, 1, ...]])
}
```

| Key              | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `input_ids`      | Token IDs (numbers representing tokens)  |
| `attention_mask` | Which tokens to pay attention to (1=yes) |

---

## Step 2: Model (Inference)

Pass tokenized inputs through the model:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

print(outputs.logits)
# tensor([[-1.5607,  1.6123],
#         [ 4.1692, -3.3464]])
```

---

## Step 3: Post-processing

Convert logits to probabilities with softmax:

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# tensor([[0.0402, 0.9598],   # 96% POSITIVE
#         [0.9995, 0.0005]])  # 99% NEGATIVE

# Get label names
print(model.config.id2label)
# {0: 'NEGATIVE', 1: 'POSITIVE'}
```

---

## Models

### Loading Models

```python
from transformers import AutoModel

# Auto class guesses the right model type
model = AutoModel.from_pretrained("bert-base-cased")

# Or use specific class if you know the model type
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
```

### Model Output Shapes

```python
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
# torch.Size([batch_size, sequence_length, hidden_size])
# Example: torch.Size([2, 16, 768])
```

### Model Heads

Different tasks need different heads:

| Class                                | Task                |
| ------------------------------------ | ------------------- |
| `AutoModel`                          | Hidden states only  |
| `AutoModelForSequenceClassification` | Text classification |
| `AutoModelForTokenClassification`    | NER, POS tagging    |
| `AutoModelForQuestionAnswering`      | Q&A                 |
| `AutoModelForCausalLM`               | Text generation     |
| `AutoModelForMaskedLM`               | Fill-mask           |

### Saving and Loading

```python
# Save locally
model.save_pretrained("my_model_directory")
tokenizer.save_pretrained("my_model_directory")

# Load from local
model = AutoModel.from_pretrained("my_model_directory")

# Push to Hugging Face Hub
model.push_to_hub("my-awesome-model")

# Load from Hub
model = AutoModel.from_pretrained("username/my-awesome-model")
```

---

## Tokenizers

### Tokenization Types

| Type       | How it works                     | Pros/Cons                    |
| ---------- | -------------------------------- | ---------------------------- |
| Word-based | Split by spaces                  | Simple but large vocabulary  |
| Character  | Split into characters            | Small vocab but less meaning |
| Subword    | Split into meaningful sub-pieces | Best of both worlds ✓        |

### Subword Example

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer.tokenize("tokenization")
print(tokens)
# ['token', '##ization']  ← Split into subwords
```

### Encoding Process

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Full encoding (recommended)
encoded = tokenizer("Using a Transformer is simple")
print(encoded)
# {'input_ids': [101, 7993, ...], 'attention_mask': [1, 1, ...]}

# Step by step (for understanding)
tokens = tokenizer.tokenize("Using a Transformer is simple")
# ['Using', 'a', 'Trans', '##former', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)
# [7993, 170, 11303, 1200, 2443, 3014]
```

### Decoding

```python
decoded = tokenizer.decode([7993, 170, 11303, 1200, 2443, 3014])
print(decoded)
# 'Using a Transformer is simple'
```

---

## Handling Multiple Sequences

### Batching Requirement

Models expect batched input (even for single sentences):

```python
# Wrong - will fail
input_ids = torch.tensor(ids)
model(input_ids)  # Error!

# Correct - add batch dimension
input_ids = torch.tensor([ids])  # Note the extra []
model(input_ids)  # Works!
```

### Padding

Sequences must have same length in a batch:

```python
# Different lengths can't form a tensor
batched_ids = [
    [200, 200, 200],
    [200, 200]  # Shorter - needs padding
]

# With padding
batched_ids = [
    [200, 200, 200],
    [200, 200, 0]  # 0 is padding token
]
```

### Attention Mask

Tell model to ignore padding tokens:

```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
attention_mask = [
    [1, 1, 1],  # Attend to all
    [1, 1, 0],  # Ignore last (padding)
]

outputs = model(
    torch.tensor(batched_ids),
    attention_mask=torch.tensor(attention_mask)
)
```

---

## Special Tokens

Tokenizers add special tokens automatically:

```python
sequence = "Hello world"

# Using tokenizer directly
model_inputs = tokenizer(sequence)
print(tokenizer.decode(model_inputs["input_ids"]))
# "[CLS] Hello world [SEP]"

# Manual tokenization (no special tokens)
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokenizer.decode(ids))
# "Hello world"
```

| Token   | Purpose              | Model       |
| ------- | -------------------- | ----------- |
| `[CLS]` | Start of sequence    | BERT        |
| `[SEP]` | End/separator        | BERT        |
| `<s>`   | Start of sequence    | GPT/Llama   |
| `</s>`  | End of sequence      | GPT/Llama   |
| `[PAD]` | Padding              | Most models |

---

## Putting It All Together

Complete workflow:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Tokenize (handles padding, truncation, tensors)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!"
]
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# Inference
output = model(**tokens)

# Get predictions
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
for i, pred in enumerate(predictions):
    label = model.config.id2label[pred.argmax().item()]
    confidence = pred.max().item()
    print(f"'{sequences[i]}' → {label} ({confidence:.2%})")
```

Output:

```text
'I've been waiting for a HuggingFace course my whole life.' → POSITIVE (95.98%)
'So have I!' → POSITIVE (99.46%)
```

---

## Quick Reference

### Tokenizer Arguments

| Argument              | Purpose                  |
| --------------------- | ------------------------ |
| `padding=True`        | Pad to same length       |
| `truncation=True`     | Truncate if too long     |
| `return_tensors="pt"` | Return PyTorch tensors   |
| `max_length=512`      | Set max sequence length  |

### Common Auto Classes

```python
from transformers import (
    AutoTokenizer,                    # Load any tokenizer
    AutoModel,                        # Base model (hidden states)
    AutoModelForSequenceClassification,  # Classification
    AutoModelForTokenClassification,     # NER
    AutoModelForQuestionAnswering,       # Q&A
    AutoModelForCausalLM,                # Text generation
    AutoModelForMaskedLM,                # Fill-mask
)
```
