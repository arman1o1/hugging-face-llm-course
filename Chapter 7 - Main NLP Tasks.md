# Chapter 7: Main NLP Tasks

This chapter covers the essential NLP tasks that form the foundation of modern Large Language Models.

---

## Overview

Chapter 7 brings together everything from previous chapters to tackle real-world NLP tasks:

```text
                    ┌─────────────────────────┐
                    │     Chapter 7 Tasks     │
                    └─────────────────────────┘
                               │
    ┌──────────────┬───────────┼───────────┬──────────────┐
    ▼              ▼           ▼           ▼              ▼
┌────────┐   ┌──────────┐  ┌───────┐  ┌────────┐   ┌──────────┐
│  NER   │   │   MLM    │  │Trans- │  │Summary │   │    QA    │
│Token   │   │ Domain   │  │lation │  │-ization│   │ Extract  │
│ Class  │   │Adaptation│  │       │  │        │   │ Answer   │
└────────┘   └──────────┘  └───────┘  └────────┘   └──────────┘
```

**What you'll learn:**

- Token Classification (NER, POS tagging)
- Masked Language Modeling (domain adaptation)
- Translation (sequence-to-sequence)
- Summarization
- Causal Language Model training from scratch
- Extractive Question Answering

> **Key Insight:** Each section can be read independently. Choose **Trainer API** for simplicity or **🤗 Accelerate** for full customization.

---

## 1. Token Classification

Token classification assigns a label to each token in a sequence. Common applications:

| Task | Description | Labels |
| ---- | ----------- | ------ |
| **NER** | Named Entity Recognition | PER, ORG, LOC, MISC |
| **POS** | Part-of-Speech tagging | NOUN, VERB, ADJ, etc. |
| **Chunking** | Group tokens into phrases | B-NP, I-NP, O |

### IOB Format

```text
Sentence: "EU rejects German call to boycott British lamb."

EU      rejects  German   call  to  boycott  British  lamb  .
B-ORG   O        B-MISC   O     O   O        B-MISC   O     O

B- = Beginning of entity
I- = Inside entity
O  = Outside (no entity)
```

### Loading CoNLL-2003 Dataset

```python
from datasets import load_dataset

raw_datasets = load_dataset("conll2003")

# Explore dataset structure
print(raw_datasets["train"][0])
# {'tokens': ['EU', 'rejects', 'German', ...],
#  'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]}

# Get label names
label_names = raw_datasets["train"].features["ner_tags"].feature.names
# ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### Aligning Labels with Tokens

Since tokenizers split words into subwords, labels need realignment:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True  # Input is already word-tokenized
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # Inside a word - use I- label or -100
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
```

### Training with Trainer

```python
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate
import numpy as np

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_names),
    id2label={i: label for i, label in enumerate(label_names)},
    label2id={label: i for i, label in enumerate(label_names)}
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [label_names[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="bert-finetuned-ner",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Using the Fine-tuned Model

```python
from transformers import pipeline

classifier = pipeline("ner", model="bert-finetuned-ner", aggregation_strategy="simple")

result = classifier("My name is Sarah and I work at Google in New York.")
# [{'entity_group': 'PER', 'word': 'Sarah', ...},
#  {'entity_group': 'ORG', 'word': 'Google', ...},
#  {'entity_group': 'LOC', 'word': 'New York', ...}]
```

---

## 2. Masked Language Modeling (Domain Adaptation)

Fine-tune a language model on domain-specific data before task-specific training.

```text
General LM (BERT) → Fine-tune on Domain → Fine-tune on Task
                          ↓
               "Domain Adaptation"
               (Legal, Medical, Code...)
```

### When to Use Domain Adaptation

- Your domain has specialized vocabulary
- General models treat domain terms as rare `[UNK]` tokens
- You have enough in-domain unlabeled data

### Loading the Dataset

```python
from datasets import load_dataset

# Using IMDB movie reviews as example domain
raw_datasets = load_dataset("imdb")
```

### Preprocessing for MLM

```python
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = raw_datasets.map(
    tokenize_function, 
    batched=True, 
    remove_columns=raw_datasets["train"].column_names
)

# Chunk texts into fixed-length sequences
block_size = 128

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)

# Data collator handles random masking
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm_probability=0.15
)
```

### Training MLM

```python
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

training_args = TrainingArguments(
    output_dir="distilbert-imdb",
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

### Perplexity Metric

```python
import math

eval_results = trainer.evaluate()
perplexity = math.exp(eval_results["eval_loss"])
print(f"Perplexity: {perplexity}")  # Lower is better
```

### Using Domain-Adapted Model

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="distilbert-imdb")
fill_mask("This movie was absolutely [MASK].")
# [{'token_str': 'amazing', 'score': 0.45}, ...]
```

---

## 3. Translation

Translation is a sequence-to-sequence task using encoder-decoder models.

### Loading Translation Data

```python
from datasets import load_dataset

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
```

### Preprocessing Translation Data

```python
from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 128

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    
    # Tokenize targets with text_target argument
    labels = tokenizer(text_target=targets, max_length=max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
```

### Training Translation Model

```python
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# BLEU metric for translation
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Format for BLEU
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

training_args = Seq2SeqTrainingArguments(
    output_dir="marian-finetuned-kde4-en-to-fr",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,  # Important for Seq2Seq
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Using Translation Model

```python
from transformers import pipeline

translator = pipeline("translation", model="marian-finetuned-kde4-en-to-fr")
translator("This plugin allows you to translate web pages.")
# [{'translation_text': 'Ce plugin vous permet de traduire des pages web.'}]
```

---

## 4. Summarization

Summarization condenses long documents into shorter versions.

| Type | Description | Example |
| ---- | ----------- | ------- |
| **Extractive** | Select key sentences | Highlight important parts |
| **Abstractive** | Generate new text | Paraphrase main ideas |

### Loading Summarization Data

```python
from datasets import load_dataset

# Amazon reviews dataset (multilingual)
raw_datasets = load_dataset("amazon_reviews_multi", "en")
```

### Preprocessing Summarization Data

```python
from transformers import AutoTokenizer

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True
    )
    
    labels = tokenizer(
        text_target=examples["review_title"],  # Titles as summaries
        max_length=max_target_length,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
```

### ROUGE Metric

```python
import evaluate
import numpy as np

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {k: round(v * 100, 2) for k, v in result.items()}
```

### Using Summarization Model

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="mt5-finetuned-amazon")
summarizer("Very long product review text here...", max_length=30)
```

---

## 5. Training a Causal LM from Scratch

Train a GPT-style model from scratch for code generation.

### When to Train from Scratch

- Highly specialized domain (code, music, DNA)
- Large amount of domain-specific data
- Existing models don't fit your vocabulary

### Gathering Data

```python
from datasets import load_dataset

# Python code dataset
raw_datasets = load_dataset("codeparrot/codeparrot-clean")
```

### Training a New Tokenizer

```python
from transformers import AutoTokenizer

# Train tokenizer on code corpus
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

def get_training_corpus():
    for i in range(0, len(raw_datasets), 1000):
        yield raw_datasets[i : i + 1000]["content"]

tokenizer = old_tokenizer.train_new_from_iterator(
    get_training_corpus(),
    vocab_size=52000
)
tokenizer.save_pretrained("code-tokenizer")
```

### Preparing Dataset

```python
context_length = 128

def tokenize(examples):
    outputs = tokenizer(
        examples["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)
```

### Initializing a New Model

```python
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_positions=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
```

### Training

```python
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    eval_strategy="steps",
    eval_steps=5000,
    logging_steps=5000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5000,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

trainer.train()
```

### Code Generation

```python
from transformers import pipeline

generator = pipeline("text-generation", model="codeparrot-ds")

prompt = "def calculate_sum(numbers):"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]["generated_text"])
```

---

## 6. Question Answering

Extractive QA finds answer spans within a given context.

```text
Context: "The Eiffel Tower is located in Paris, France."
Question: "Where is the Eiffel Tower?"
Answer: "Paris, France" (start=35, end=48)
```

### SQuAD Dataset

```python
from datasets import load_dataset

raw_datasets = load_dataset("squad")

# Example structure
print(raw_datasets["train"][0])
# {'context': '...', 'question': '...', 
#  'answers': {'text': ['Paris'], 'answer_start': [35]}}
```

### Preprocessing for QA

```python
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_length = 384
stride = 128

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        
        # Find context start/end
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # Check if answer is in this span
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
```

### Training QA Model

```python
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
data_collator = DefaultDataCollator()

training_args = TrainingArguments(
    output_dir="bert-finetuned-squad",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset.select(range(1000)),  # Small eval subset
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

### Using QA Model

```python
from transformers import pipeline

qa = pipeline("question-answering", model="bert-finetuned-squad")

result = qa(
    question="Which deep learning libraries back Transformers?",
    context="Transformers is backed by PyTorch, TensorFlow, and JAX."
)
print(result)
# {'answer': 'PyTorch, TensorFlow, and JAX', 'score': 0.98, ...}
```

---

## Quick Reference

### Task-to-Model Mapping

| Task | Model Class | Pipeline |
| ---- | ----------- | -------- |
| Token Classification | `AutoModelForTokenClassification` | `"ner"` |
| Masked LM | `AutoModelForMaskedLM` | `"fill-mask"` |
| Translation | `AutoModelForSeq2SeqLM` | `"translation"` |
| Summarization | `AutoModelForSeq2SeqLM` | `"summarization"` |
| Causal LM | `AutoModelForCausalLM` | `"text-generation"` |
| Question Answering | `AutoModelForQuestionAnswering` | `"question-answering"` |

### Metrics by Task

| Task | Metric | Library |
| ---- | ------ | ------- |
| NER | seqeval (F1, Precision, Recall) | `evaluate.load("seqeval")` |
| MLM | Perplexity | `math.exp(loss)` |
| Translation | BLEU | `evaluate.load("sacrebleu")` |
| Summarization | ROUGE | `evaluate.load("rouge")` |
| QA | Exact Match, F1 | `evaluate.load("squad")` |

### Data Collators

| Task | Collator |
| ---- | -------- |
| Token Classification | `DataCollatorForTokenClassification` |
| MLM | `DataCollatorForLanguageModeling(mlm=True)` |
| Causal LM | `DataCollatorForLanguageModeling(mlm=False)` |
| Seq2Seq | `DataCollatorForSeq2Seq` |
| QA | `DefaultDataCollator` |

### Training Arguments Comparison

```python
# Standard tasks
TrainingArguments(...)

# Seq2Seq tasks (translation, summarization)
Seq2SeqTrainingArguments(
    ...,
    predict_with_generate=True  # Use generation for evaluation
)
```

### Label Alignment for Token Classification

```python
# Special tokens → -100
# First subword of word → original label
# Subsequent subwords → same label (or -100)
```
