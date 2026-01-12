# Chapter 5: The Datasets Library

This chapter covers the 🤗 Datasets library for loading, processing, and sharing datasets.

---

## Overview

The 🤗 Datasets library handles everything from local files to massive multi-gigabyte datasets.

```text
Your Data → [🤗 Datasets] → Ready for Training
              ↓
    • Memory Mapping (huge datasets)
    • Streaming (infinite datasets)
    • Easy transformations
```

**What you'll learn:**

- Load datasets from various sources (Hub, local, remote)
- Slice, dice, and transform datasets
- Handle big data with memory mapping and streaming
- Create and share your own datasets
- Build semantic search with FAISS

---

## Loading Datasets

### From the Hub

```python
from datasets import load_dataset

# Load dataset from Hub
dataset = load_dataset("imdb")

# Load specific split
train_dataset = load_dataset("imdb", split="train")
```

### From Local/Remote Files

🤗 Datasets supports many formats:

| Format  | How to Load                                              |
| ------- | -------------------------------------------------------- |
| CSV     | `load_dataset("csv", data_files="file.csv")`             |
| JSON    | `load_dataset("json", data_files="file.json")`           |
| Text    | `load_dataset("text", data_files="file.txt")`            |
| Parquet | `load_dataset("parquet", data_files="file.parquet")`     |

**Loading local JSON:**

```python
from datasets import load_dataset

# Single file
dataset = load_dataset("json", data_files="data.json")

# Multiple splits
data_files = {
    "train": "train.json",
    "test": "test.json"
}
dataset = load_dataset("json", data_files=data_files)

# Nested JSON (with field parameter)
dataset = load_dataset("json", data_files="data.json", field="data")
```

**Loading from URL:**

```python
url = "https://example.com/path/to/"
data_files = {
    "train": url + "train.json.gz",
    "test": url + "test.json.gz"
}
dataset = load_dataset("json", data_files=data_files)
```

> **Tip:** 🤗 Datasets automatically decompresses `.gz`, `.zip`, and `.tar` files!

---

## Slicing and Dicing Data

### Exploring Your Dataset

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train")

# View structure
print(dataset)
# Dataset({features: ['text', 'label'], num_rows: 25000})

# Access single example
print(dataset[0])

# Access multiple examples (returns dict of lists)
print(dataset[:3])

# Access specific column
print(dataset["text"][:3])
```

### Key Methods

| Method              | Description                        |
| ------------------- | ---------------------------------- |
| `shuffle()`         | Randomly shuffle dataset           |
| `select()`          | Select specific indices            |
| `filter()`          | Filter by condition                |
| `sort()`            | Sort by column                     |
| `unique()`          | Get unique values in column        |
| `train_test_split()`| Split into train/validation        |

**Shuffle and Select:**

```python
# Get random sample of 1000 examples
sample = dataset.shuffle(seed=42).select(range(1000))
```

**Filter:**

```python
# Keep only positive reviews
positive = dataset.filter(lambda x: x["label"] == 1)

# Filter with multiprocessing
long_reviews = dataset.filter(
    lambda x: len(x["text"]) > 200,
    num_proc=4
)
```

**Sort:**

```python
# Sort by a column
sorted_dataset = dataset.sort("rating")

# Descending order
sorted_dataset = dataset.sort("rating", reverse=True)
```

**Create Validation Split:**

```python
# Split training into train + validation
split_dataset = dataset.train_test_split(train_size=0.8, seed=42)

# Rename "test" to "validation"
split_dataset["validation"] = split_dataset.pop("test")
```

---

## Transforming Data with map()

The `map()` method is your Swiss Army knife for data transformation.

### Basic Usage

```python
# Add a new column
def compute_length(example):
    return {"text_length": len(example["text"])}

dataset = dataset.map(compute_length)
```

### Modifying Existing Column

```python
import html

# Clean HTML entities
dataset = dataset.map(lambda x: {"text": html.unescape(x["text"])})
```

### Batched Processing (Faster!)

```python
# Process in batches (default batch_size=1000)
dataset = dataset.map(
    lambda batch: {"text": [html.unescape(t) for t in batch["text"]]},
    batched=True
)
```

### Tokenization Example

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Batched tokenization is ~30x faster!
tokenized = dataset.map(tokenize_function, batched=True)
```

> **Performance Tip:** Always use `batched=True` with fast tokenizers for maximum speed.

---

## Converting to/from Pandas

```python
# Convert to Pandas DataFrame
dataset.set_format("pandas")
df = dataset[:]  # Now a DataFrame

# Analyze with Pandas
value_counts = df["label"].value_counts()

# Convert back to Dataset
dataset.reset_format()

# Create Dataset from DataFrame
from datasets import Dataset
new_dataset = Dataset.from_pandas(df)
```

---

## Saving Datasets

| Method              | Format       | Use Case                     |
| ------------------- | ------------ | ---------------------------- |
| `save_to_disk()`    | Arrow        | Fast reload, preserves types |
| `to_csv()`          | CSV          | Universal compatibility      |
| `to_json()`         | JSON Lines   | Web APIs, line-by-line       |

```python
# Save to Arrow format (recommended)
dataset.save_to_disk("my_dataset")

# Load back
from datasets import load_from_disk
dataset = load_from_disk("my_dataset")

# Save to JSON Lines
dataset.to_json("data.jsonl")

# Save each split separately
for split, data in dataset.items():
    data.to_json(f"data-{split}.jsonl")
```

---

## Handling Big Data

### Memory Mapping

🤗 Datasets uses Apache Arrow and memory-mapped files to handle datasets larger than RAM.

```python
# This works even for 20GB+ datasets!
dataset = load_dataset("pubmed", split="train")

# Memory used is minimal - data is memory-mapped
import psutil
print(f"RAM: {psutil.Process().memory_info().rss / 1024**2:.0f} MB")
```

**Why it works:**

- Data stays on disk, mapped to virtual memory
- Only accessed portions are loaded into RAM
- Multiple processes can share the same data

### Streaming for Huge Datasets

When datasets are too large for your disk, use streaming:

```python
from datasets import load_dataset

# Stream instead of download
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

# Returns IterableDataset
print(next(iter(dataset)))
```

**Streaming Operations:**

```python
# Chain operations efficiently
dataset = (
    load_dataset("allenai/c4", "en", split="train", streaming=True)
    .shuffle(seed=42, buffer_size=10_000)
    .skip(1000)       # Skip first 1000
    .take(5000)       # Take next 5000
    .map(lambda x: {"length": len(x["text"])})
    .filter(lambda x: x["length"] > 100)
)

# Iterate through results
for i, example in enumerate(dataset):
    if i >= 10:
        break
    print(example)
```

**Combine Multiple Streams:**

```python
from datasets import interleave_datasets

# Interleave examples from multiple datasets
combined = interleave_datasets([dataset1, dataset2])
```

---

## Creating Your Own Dataset

### From Python Objects

```python
from datasets import Dataset

# From dictionary
data = {
    "text": ["Hello world", "Goodbye world"],
    "label": [1, 0]
}
dataset = Dataset.from_dict(data)

# From list of dicts
data = [
    {"text": "Hello", "label": 1},
    {"text": "Goodbye", "label": 0}
]
dataset = Dataset.from_list(data)
```

### Using APIs to Create Dataset

```python
import requests
from datasets import Dataset

# Fetch data from API
response = requests.get("https://api.example.com/data")
data = response.json()

# Convert to dataset
issues_dataset = Dataset.from_dict(data)
```

### Upload to Hub

```python
from huggingface_hub import notebook_login

# Login first
notebook_login()

# Push dataset to Hub
dataset.push_to_hub("my-username/my-dataset")

# Now anyone can load it!
# load_dataset("my-username/my-dataset")
```

---

## Semantic Search with FAISS

FAISS (Facebook AI Similarity Search) enables fast similarity search over embeddings.

### Step 1: Create Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded = tokenizer(
        text_list, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    return cls_pooling(output)
```

### Step 2: Add Embeddings to Dataset

```python
# Add embeddings column
def add_embeddings(example):
    embedding = get_embeddings([example["text"]])
    return {"embeddings": embedding.cpu().numpy()[0]}

dataset = dataset.map(add_embeddings)
```

### Step 3: Create FAISS Index

```python
# Add FAISS index
dataset.add_faiss_index(column="embeddings")
```

### Step 4: Search

```python
# Search for similar items
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().numpy()

# Find 5 most similar examples
scores, samples = dataset.get_nearest_examples(
    "embeddings", 
    question_embedding, 
    k=5
)

# View results
import pandas as pd
results = pd.DataFrame(samples)
results["scores"] = scores
print(results.sort_values("scores", ascending=False))
```

---

## Quick Reference

### Loading Data

```python
from datasets import load_dataset, load_from_disk

# From Hub
dataset = load_dataset("dataset_name")

# From files
dataset = load_dataset("csv", data_files="file.csv")
dataset = load_dataset("json", data_files={"train": "train.json"})

# From disk (Arrow format)
dataset = load_from_disk("./saved_dataset")

# Streaming
dataset = load_dataset("huge_dataset", streaming=True)
```

### Key Dataset Methods

| Method                      | Purpose                              |
| --------------------------- | ------------------------------------ |
| `map(fn, batched=True)`     | Transform examples                   |
| `filter(fn)`                | Keep matching examples               |
| `shuffle(seed=42)`          | Randomize order                      |
| `select(indices)`           | Select by index                      |
| `sort(column)`              | Sort by column                       |
| `train_test_split()`        | Create train/validation splits       |
| `set_format("pandas")`      | Convert output to pandas             |
| `add_faiss_index(column)`   | Enable similarity search             |
| `get_nearest_examples()`    | Find similar items                   |

### Saving Data

```python
# Arrow format (best for reloading)
dataset.save_to_disk("./dataset")

# Other formats
dataset.to_csv("data.csv")
dataset.to_json("data.jsonl")

# Push to Hub
dataset.push_to_hub("username/dataset-name")
```

### Streaming (IterableDataset)

```python
dataset = load_dataset("name", streaming=True)

# Available methods
dataset.shuffle(buffer_size=10000)
dataset.skip(n)
dataset.take(n)
dataset.map(fn)
dataset.filter(fn)
```
