# Chapter 1: Introduction to NLP and Transformers

This chapter covers the fundamentals of NLP, LLMs, and the Transformer architecture.

---

## What is NLP?

**Natural Language Processing (NLP)** enables computers to understand human language.

Common NLP tasks:

- **Text Classification** - Sentiment analysis, spam detection
- **Named Entity Recognition (NER)** - Identifying people, places, organizations
- **Text Generation** - Auto-completing text
- **Question Answering** - Extracting answers from context
- **Translation** - Converting text between languages
- **Summarization** - Condensing long text

---

## What are LLMs?

**Large Language Models** are massive neural networks trained on huge amounts of text.

Key characteristics:

| Feature             | Description                                          |
| ------------------- | ---------------------------------------------------- |
| Scale               | Billions of parameters                               |
| General             | Can perform multiple tasks without specific training |
| In-context learning | Learn from examples in the prompt                    |

Limitations:

- Hallucinations (confident but wrong)
- No true world understanding
- Can reproduce biases
- Limited context window
- Resource intensive

---

## The Pipeline Function

The simplest way to use transformers:

```python
from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
classifier("I love this course!")
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

---

## Available Pipelines

### Text

```python
# Text Generation
generator = pipeline("text-generation")
generator("Once upon a time", max_length=30)

# Zero-shot Classification (no training needed)
classifier = pipeline("zero-shot-classification")
classifier(
    "This is a tech tutorial",
    candidate_labels=["education", "politics", "business"]
)

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
ner("Elon Musk works at Tesla in California")
# Output: [{'entity_group': 'PER', 'word': 'Elon Musk'}, 
#          {'entity_group': 'ORG', 'word': 'Tesla'},
#          {'entity_group': 'LOC', 'word': 'California'}]

# Question Answering
qa = pipeline("question-answering")
qa(question="Where does Elon work?", 
   context="Elon Musk works at Tesla in California")
# Output: {'answer': 'Tesla'}

# Summarization
summarizer = pipeline("summarization")
summarizer("Long article text here...", max_length=50)

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translator("Hello, how are you?")
# Output: [{'translation_text': 'Bonjour, comment allez-vous?'}]

# Fill Mask
unmasker = pipeline("fill-mask")
unmasker("Paris is the <mask> of France.")
# Output: [{'token_str': 'capital', ...}]
```

### Other Modalities

```python
# Image Classification
pipeline("image-classification")

# Speech to Text
pipeline("automatic-speech-recognition")

# Text to Speech
pipeline("text-to-speech")
```

---

## How Transformers Work

### Key Concepts

**Self-supervised learning**: Model learns patterns from unlabeled text

- No human labeling needed
- Trained on massive text corpora

**Transfer learning**: Use pretrained model → Fine-tune for your task

- Faster training
- Less data needed
- Better results

### Attention Mechanism

The core innovation - model learns which words to focus on:

```text
"The cat sat on the mat because it was tired"
                                  ↑
                    "it" attends to "cat" (not "mat")
```

---

## Transformer Architectures

### 1. Encoder-only (BERT-like)

- Sees all words at once (bidirectional)
- Best for: **understanding** tasks

| Use Case       | Example            |
| -------------- | ------------------ |
| Classification | Sentiment analysis |
| NER            | Entity extraction  |
| Q&A            | Extracting answers |

Models: BERT, DistilBERT, ModernBERT

### 2. Decoder-only (GPT-like)

- Sees only previous words (autoregressive)
- Best for: **generation** tasks

| Use Case        | Example                 |
| --------------- | ----------------------- |
| Text generation | Chatbots, story writing |
| Code generation | Copilot                 |

Models: GPT, Llama, Gemma, SmolLM

### 3. Encoder-Decoder (T5-like)

- Encoder understands input, decoder generates output
- Best for: **transformation** tasks

| Use Case      | Example           |
| ------------- | ----------------- |
| Translation   | English → French  |
| Summarization | Article → Summary |

Models: T5, BART

---

## Quick Reference: Choosing an Architecture

```text
Need to understand text?     → Encoder (BERT)
Need to generate text?       → Decoder (GPT/Llama)
Need to transform text?      → Encoder-Decoder (T5)
```

---

## Architecture vs Checkpoint

| Term         | Meaning          | Example           |
| ------------ | ---------------- | ----------------- |
| Architecture | Model structure  | BERT              |
| Checkpoint   | Trained weights  | `bert-base-cased` |
| Model        | General term     | Can mean either   |

```python
# Using a specific checkpoint
from transformers import pipeline

# Architecture: BERT, Checkpoint: bert-base-cased
classifier = pipeline("sentiment-analysis", model="bert-base-uncased")
```

---

## Key Transformer Models Timeline

| Year | Model       | Type     | Notes                        |
| ---- | ----------- | -------- | ---------------------------- |
| 2017 | Transformer | Original | "Attention is All You Need"  |
| 2018 | GPT         | Decoder  | First pretrained transformer |
| 2018 | BERT        | Encoder  | Bidirectional understanding  |
| 2019 | T5          | Enc-Dec  | Text-to-text framework       |
| 2020 | GPT-3       | Decoder  | Zero-shot learning           |
| 2023 | Llama       | Decoder  | Open weights                 |
| 2023 | Mistral     | Decoder  | Efficient 7B model           |
| 2024 | Gemma 2     | Decoder  | Lightweight, efficient       |
