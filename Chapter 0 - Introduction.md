# Chapter 0: Introduction

This chapter covers setting up your environment for the Hugging Face LLM Course.

---

## Setup Options

### Option 1: Google Colab (Easiest)

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Create a new notebook
3. Run:

```python
!pip install transformers[sentencepiece]
```

### Option 2: Local Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install packages
pip install transformers[sentencepiece]
```

---

## Required Packages

```bash
pip install transformers[sentencepiece] torch datasets
```

---

## Hugging Face Account

1. Create account at [huggingface.co/join](https://huggingface.co/join)
2. Generate token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Login in code:

```python
from huggingface_hub import login
login()
```

---

## Verify Setup

```python
import transformers
print(f"Transformers: {transformers.__version__}")
```

---

## Quick Example

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love Hugging Face!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```
