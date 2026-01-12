# Chapter 4: The Hugging Face Hub

This chapter covers how to discover, use, and share models on the Hugging Face Hub.

---

## Overview

The [Hugging Face Hub](https://huggingface.co/) is a central platform for discovering, using, and sharing ML models and datasets.

```text
You → [Hub] ← Community
     Models | Datasets | Spaces
```

**What you'll learn:**

- Navigate and use models from the Hub
- Share your own models using multiple methods
- Create comprehensive model cards

---

## Using Pretrained Models

### Finding Models on the Hub

The Hub hosts thousands of models for various tasks: NLP, Vision, Audio, and more.

**Filtering options:**

| Filter     | Examples                              |
| ---------- | ------------------------------------- |
| Task       | text-classification, fill-mask, etc.  |
| Language   | English, French, Chinese              |
| Library    | Transformers, spaCy, timm             |
| License    | MIT, Apache-2.0, cc-by-4.0            |

---

### Using Models with Pipeline

The simplest way to use any model:

```python
from transformers import pipeline

# French mask-filling model
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")

print(results[0])
# {'sequence': 'Le camembert est délicieux :)', 'score': 0.49, 'token_str': 'délicieux'}
```

> **Important:** Always match the model to the task! A `fill-mask` model won't work for `text-classification`.

---

### Using Auto Classes (Recommended)

Auto classes are architecture-agnostic and more flexible:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# These work with any compatible model
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

**Why use Auto classes?**

- Easy to switch between models
- No need to know the exact architecture
- Cleaner, more maintainable code

---

## Sharing Pretrained Models

There are three ways to share models to the Hub:

| Method              | Best For                         |
| ------------------- | -------------------------------- |
| `push_to_hub` API   | Integrated with Trainer/models   |
| `huggingface_hub`   | Programmatic control             |
| Web Interface       | Quick uploads, manual management |

---

### Authentication

Before uploading, authenticate with the Hub:

```python
# In a notebook
from huggingface_hub import notebook_login
notebook_login()
```

```bash
# In terminal
huggingface-cli login
```

---

### Method 1: push_to_hub API

#### With Trainer (Easiest)

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="bert-finetuned-mrpc",
    push_to_hub=True,  # Auto-upload on save
    save_strategy="epoch",
)

# After training
trainer.push_to_hub()  # Final upload with model card
```

#### Directly on Models/Tokenizers

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("camembert-base")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

# Fine-tune or modify...

# Push to Hub
model.push_to_hub("my-awesome-model")
tokenizer.push_to_hub("my-awesome-model")
```

**Push to an organization:**

```python
model.push_to_hub("my-awesome-model", organization="my-org")
```

---

### Method 2: huggingface_hub Library

#### Create a Repository

```python
from huggingface_hub import create_repo

# Create in your namespace
create_repo("dummy-model")

# Create in an organization
create_repo("dummy-model", organization="my-org")

# Create as private
create_repo("dummy-model", private=True)

# Create different repo types
create_repo("my-dataset", repo_type="dataset")
create_repo("my-space", repo_type="space")
```

#### Upload Files

```python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="./config.json",
    path_in_repo="config.json",
    repo_id="username/dummy-model",
)
```

> **Note:** `upload_file` has a 5GB limit. For larger files, use git-lfs.

#### Upload Entire Folder

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="./my_model",
    repo_id="username/my-model",
)
```

---

### Method 3: Git-Based Approach

For maximum control, use git directly:

```bash
# Initialize git-lfs
git lfs install

# Clone your repo
git clone https://huggingface.co/username/my-model
cd my-model
```

Save model files:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("camembert-base")
tokenizer = AutoTokenizer.from_pretrained("camembert-base")

model.save_pretrained("./my-model")
tokenizer.save_pretrained("./my-model")
```

Push to Hub:

```bash
git add .
git commit -m "Add model and tokenizer files"
git push
```

> **Note:** Large files (like `.bin`, `.h5`) are automatically tracked by git-lfs.

---

## Building a Model Card

The model card (`README.md`) is crucial for documentation and discoverability.

### Why Model Cards Matter

- **Reproducibility**: Others can understand how the model was trained
- **Transparency**: Document biases, limitations, and intended uses
- **Discoverability**: Metadata helps users find your model

---

### Model Card Structure

```markdown
---
language: en
license: mit
datasets:
  - glue
tags:
  - text-classification
metrics:
  - accuracy
---

# Model Name

Brief description of the model.

## Model Description

Architecture, version, and general information.

## Intended Uses & Limitations

- **Intended Use**: What it's designed for
- **Limitations**: What it's NOT good at
- **Biases**: Known biases from training data

## How to Use

\`\`\`python
from transformers import pipeline
classifier = pipeline("text-classification", model="username/my-model")
\`\`\`

## Training Data

Describe the dataset(s) used.

## Training Procedure

- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5

## Evaluation Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | 0.89  |
| F1       | 0.87  |
```

---

### Model Card Metadata

The YAML header enables Hub filtering:

```yaml
---
language: fr
license: mit
datasets:
  - oscar
tags:
  - french
  - text-generation
pipeline_tag: text-generation
---
```

| Field          | Description                        |
| -------------- | ---------------------------------- |
| `language`     | Model language(s)                  |
| `license`      | License type                       |
| `datasets`     | Training datasets                  |
| `tags`         | Searchable keywords                |
| `pipeline_tag` | Default pipeline task              |
| `metrics`      | Evaluation metrics used            |

---

## Quick Reference

### Loading Models

```python
from transformers import pipeline, AutoTokenizer, AutoModel

# Quick way (pipeline)
classifier = pipeline("text-classification", model="model-name")

# Full control (Auto classes)
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModel.from_pretrained("model-name")
```

### Sharing Models

```python
from huggingface_hub import create_repo, upload_file, upload_folder

# Create repo
create_repo("my-model")

# Upload single file
upload_file("config.json", path_in_repo="config.json", repo_id="user/my-model")

# Upload folder
upload_folder(folder_path="./model", repo_id="user/my-model")

# Or use push_to_hub on model objects
model.push_to_hub("my-model")
tokenizer.push_to_hub("my-model")
```

### Key huggingface_hub Functions

| Function               | Purpose                          |
| ---------------------- | -------------------------------- |
| `create_repo()`        | Create a new repository          |
| `upload_file()`        | Upload a single file             |
| `upload_folder()`      | Upload entire folder             |
| `delete_repo()`        | Delete a repository              |
| `list_models()`        | List available models            |
| `notebook_login()`     | Authenticate in notebooks        |

### Model Card Checklist

- [ ] Model description (architecture, version)
- [ ] Intended uses and limitations
- [ ] Code example for usage
- [ ] Training data description
- [ ] Training procedure details
- [ ] Evaluation metrics and results
- [ ] YAML metadata for discoverability
