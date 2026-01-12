# Chapter 10: Introduction to Argilla

This chapter covers how to use Argilla to annotate and curate high-quality datasets for training and evaluating ML models.

---

## Overview

Argilla is a data annotation and curation tool that helps you build high-quality datasets efficiently.

```text
                    ┌─────────────────────────┐
                    │     Argilla Workflow    │
                    └─────────────────────────┘
                               │
    ┌──────────────┬───────────┼───────────┬──────────────┐
    ▼              ▼           ▼           ▼              ▼
┌────────┐   ┌──────────┐  ┌───────┐  ┌────────┐   ┌──────────┐
│  Setup │   │Configure │  │Upload │  │Annotate│   │ Export   │
│Instance│   │ Dataset  │  │Records│  │  Data  │   │ to Hub   │
└────────┘   └──────────┘  └───────┘  └────────┘   └──────────┘
```

**Why use Argilla?**

- Turn unstructured data into structured NLP datasets
- Curate low-quality datasets into high-quality ones
- Gather human feedback for LLMs and multimodal models
- Collaborate with experts or crowdsource annotations

---

## 1. Setting Up Argilla

### Deploy the Argilla UI

The easiest way is through **Hugging Face Spaces**:

1. Go to the [Argilla Space template](https://huggingface.co/new-space?template=argilla%2Fargilla-template-space)
2. Follow the form to create your Space
3. Enable **Persistent storage** in Settings (to preserve data on restarts)

> ⚠️ **Important:** Enable persistent storage to avoid losing data when the Space restarts.

### Install the Python SDK

```bash
pip install argilla
```

### Connect to Your Instance

```python
import argilla as rg

# For public Spaces
client = rg.Argilla(
    api_url="https://<your-username>-<space-name>.hf.space",
    api_key="your-api-key"  # Find in My Settings in Argilla UI
)

# For private Spaces (add HF token)
client = rg.Argilla(
    api_url="https://<your-username>-<space-name>.hf.space",
    api_key="your-api-key",
    headers={"Authorization": f"Bearer {HF_TOKEN}"}
)

# Verify connection
client.me  # Should return your user info
```

**Required credentials:**

| Credential | Where to Find |
| ---------- | ------------- |
| API URL | Space → ⋮ → Embed this Space → Direct URL |
| API Key | Argilla UI → My Settings |
| HF Token | Only for private Spaces (huggingface.co/settings/tokens) |

---

## 2. Loading a Dataset to Argilla

### Connect and Load Data

```python
import argilla as rg
from datasets import load_dataset

# Connect to Argilla
client = rg.Argilla(api_url="...", api_key="...")

# Load data from Hugging Face Hub
data = load_dataset("SetFit/ag_news", split="train")
print(data.features)
# {'text': Value(dtype='string'), 'label': Value(dtype='int64'), 'label_text': Value(dtype='string')}
```

### Configure Dataset Settings

Settings define **what you'll annotate** (fields) and **how you'll annotate** (questions):

```python
settings = rg.Settings(
    guidelines="Classify news articles and identify named entities.",
    fields=[
        rg.TextField(name="text")  # The content to annotate
    ],
    questions=[
        # Text Classification
        rg.LabelQuestion(
            name="label",
            title="Classify the text:",
            labels=data.unique("label_text")  # Use existing labels
        ),
        # Named Entity Recognition
        rg.SpanQuestion(
            name="entities",
            title="Highlight all entities in the text:",
            labels=["PERSON", "ORG", "LOC", "EVENT"],
            field="text"  # Which field contains the text
        ),
    ],
)
```

### Question Types

| Question Type | Use Case |
| ------------- | -------- |
| `LabelQuestion` | Single-label classification |
| `MultiLabelQuestion` | Multi-label classification |
| `SpanQuestion` | Token/span annotation (NER) |
| `TextQuestion` | Free-text responses |
| `RatingQuestion` | Numeric ratings |
| `RankingQuestion` | Ordering/ranking items |

### Create and Upload Dataset

```python
# Create the dataset
dataset = rg.Dataset(
    name="ag_news",
    settings=settings,
    workspace="default"  # Optional: specify workspace
)
dataset.create()

# Upload records with field mapping
dataset.records.log(
    data,
    mapping={"label_text": "label"}  # Map dataset column → question name
)
```

> 💡 **Tip:** Mapping existing labels as pre-annotations speeds up annotation work.

---

## 3. Annotating Your Dataset

### Writing Annotation Guidelines

Before annotating, write clear guidelines:

1. Go to **Dataset Settings** in the Argilla UI
2. Modify **guidelines** and **question descriptions**
3. Document edge cases and label definitions

### Task Distribution

Configure how work is distributed among annotators:

| Setting | Description |
| ------- | ----------- |
| Minimum submitted | Records marked complete after N responses (default: 1) |
| Overlap | Higher values enable inter-annotator agreement analysis |

> 💡 For solo annotation, keep minimum submitted at 1.

### Annotation Workflow

1. Open your dataset in the Argilla UI
2. Review each record and provide responses:
   - For labels: Select the appropriate option
   - For spans: Highlight text and assign labels
3. Choose an action:
   - **Submit** - Complete the annotation
   - **Save as draft** - Return later
   - **Discard** - Remove from dataset

### Adding Team Members

- **HF Spaces:** Team members can log in via Hugging Face OAuth
- **Self-hosted:** Create users following the [Argilla user guide](https://docs.argilla.io/latest/how_to_guides/user/)

---

## 4. Using Your Annotated Dataset

### Load the Dataset

```python
import argilla as rg

client = rg.Argilla(api_url="...", api_key="...")
dataset = client.datasets(name="ag_news")

# Access records
for record in dataset.records:
    print(record.fields["text"])
```

### Filter Records

Get only completed annotations:

```python
# Filter by status
status_filter = rg.Query(
    filter=rg.Filter([("status", "==", "completed")])
)
filtered_records = dataset.records(status_filter)
```

**Record statuses:**

| Status | Meaning |
| ------ | ------- |
| `completed` | Met minimum submitted responses |
| `pending` | Still needs annotations |

**Response statuses:**

| Status | Meaning |
| ------ | ------- |
| `submitted` | Final answer submitted |
| `draft` | Saved but not submitted |
| `discarded` | Marked as invalid/skipped |

### Export to Hugging Face Hub

```python
# Export only completed records as HF Dataset
filtered_records.to_datasets().push_to_hub("your-username/ag_news_annotated")

# Or export full Argilla dataset (preserves settings)
dataset.to_hub(repo_id="your-username/ag_news_annotated")
```

### Import from Hub

Others can import your full Argilla dataset:

```python
# Import with settings preserved
dataset = rg.Dataset.from_hub(repo_id="your-username/ag_news_annotated")
```

---

## Quick Reference

### Complete Workflow

```python
import argilla as rg
from datasets import load_dataset

# 1. Connect
client = rg.Argilla(api_url="...", api_key="...")

# 2. Configure
settings = rg.Settings(
    guidelines="Your annotation guidelines here.",
    fields=[rg.TextField(name="text")],
    questions=[
        rg.LabelQuestion(name="label", labels=["pos", "neg"])
    ],
)

# 3. Create dataset
dataset = rg.Dataset(name="my_dataset", settings=settings)
dataset.create()

# 4. Upload records
data = load_dataset("your-dataset", split="train")
dataset.records.log(data)

# 5. Annotate in UI...

# 6. Export
dataset = client.datasets(name="my_dataset")
dataset.records.to_datasets().push_to_hub("user/dataset")
```

### Common Patterns

```python
# Text Classification Setup
settings = rg.Settings(
    fields=[rg.TextField(name="text")],
    questions=[rg.LabelQuestion(name="sentiment", labels=["positive", "negative", "neutral"])],
)

# NER Setup
settings = rg.Settings(
    fields=[rg.TextField(name="text")],
    questions=[rg.SpanQuestion(name="entities", labels=["PER", "ORG", "LOC"], field="text")],
)

# Multi-label Classification
settings = rg.Settings(
    fields=[rg.TextField(name="text")],
    questions=[rg.MultiLabelQuestion(name="topics", labels=["tech", "sports", "politics"])],
)

# Rating/Feedback
settings = rg.Settings(
    fields=[rg.TextField(name="response")],
    questions=[rg.RatingQuestion(name="quality", values=[1, 2, 3, 4, 5])],
)
```

### Resources

| Resource | URL |
| -------- | --- |
| **Argilla Docs** | [docs.argilla.io](https://docs.argilla.io/latest/) |
| **Argilla Demo** | [demo.argilla.io](https://demo.argilla.io) |
| **Tutorials** | [docs.argilla.io/tutorials](https://docs.argilla.io/latest/tutorials/) |
| **HF Spaces Template** | [argilla-template-space](https://huggingface.co/new-space?template=argilla%2Fargilla-template-space) |
