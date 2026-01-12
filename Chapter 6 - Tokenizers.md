# Chapter 6: Tokenizers

This chapter covers training and understanding tokenizers — the foundation of how models process text.

---

## Overview

When you want to train a model from scratch or on a specialized domain, using a pretrained tokenizer may be suboptimal. A tokenizer trained on English won't work well for Japanese; one trained on general text won't be efficient for Python code.

```text
Raw Text -> [Tokenizer] -> Token IDs -> Model
              |
    * Normalization (cleanup)
    * Pre-tokenization (split into words)
    * Subword tokenization (BPE/WordPiece/Unigram)
```

**What you'll learn:**

- Train a new tokenizer from an existing one
- Understand fast tokenizers and their special features
- Learn the three main tokenization algorithms (BPE, WordPiece, Unigram)
- Build tokenizers from scratch

> **Key Insight:** Training a tokenizer is **deterministic** (same input = same output), unlike model training which uses stochastic gradient descent.

---

## Training a New Tokenizer

### From an Existing Tokenizer

The easiest way to create a custom tokenizer is to train one from an existing model's tokenizer:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a corpus
dataset = load_dataset("code_search_net", "python", split="train")

# Memory-efficient generator
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["whole_func_string"]

# Load base tokenizer and train new one
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(
    get_training_corpus(),
    vocab_size=52000
)
```

### Why Use a Generator?

```python
# Bad: Loads everything into memory
corpus = [dataset[i:i+1000]["text"] for i in range(0, len(dataset), 1000)]

# Good: Memory-efficient
corpus = (dataset[i:i+1000]["text"] for i in range(0, len(dataset), 1000))
```

### Saving the Tokenizer

```python
tokenizer.save_pretrained("./my-tokenizer")
tokenizer.push_to_hub("my-username/my-tokenizer")
```

---

## Fast Tokenizers' Special Powers

Fast tokenizers (Rust-backed) provide features beyond basic tokenization.

### Fast vs Slow Tokenizers

| Feature | Slow (Python) | Fast (Rust) |
| ------- | ------------- | ----------- |
| Speed | Baseline | 10-100x faster |
| Offset mapping | No | Yes |
| Word IDs | No | Yes |

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer.is_fast)  # True
```

### BatchEncoding Object

```python
text = "My name is Sylvain and I work at Hugging Face."
encoding = tokenizer(text)

# Access tokens directly
encoding.tokens()
# ['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', ...]

# Get word IDs (which word each token belongs to)
encoding.word_ids()
# [None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, None]
```

### Offset Mapping

```python
# Get character span for a word
start, end = encoding.word_to_chars(3)
print(text[start:end])  # "Sylvain"

# Get token's character span
start, end = encoding.token_to_chars(5)
print(text[start:end])  # "yl"
```

**Use cases:** NER label alignment, QA answer extraction, whole word masking

---

## Normalization and Pre-tokenization

### 1. Normalization

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
normalizer = tokenizer.backend_tokenizer.normalizer

normalizer.normalize_str("Héllò hôw are ü?")
# "hello how are u?"
```

### 2. Pre-tokenization

```python
# BERT: splits on whitespace and punctuation
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are you?")
# [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ...]

# GPT-2: keeps spaces with special 'Ġ' prefix
# [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ...]
```

### SentencePiece

Used by T5, XLNet — treats text as raw bytes, uses special space marker:

```python
tokenizer = AutoTokenizer.from_pretrained("t5-small")
# [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ...]
```

---

## Tokenization Algorithms

| Algorithm | Direction | Scoring | Models |
| --------- | --------- | ------- | ------ |
| **BPE** | Bottom-up | Frequency | GPT-2, RoBERTa |
| **WordPiece** | Bottom-up | Score formula | BERT, DistilBERT |
| **Unigram** | Top-down | Loss impact | T5, ALBERT |

---

## Byte-Pair Encoding (BPE)

### How It Works

1. Start with character vocabulary
2. Find most frequent adjacent pair
3. Merge that pair into new token
4. Repeat until vocab size reached

### Example

```text
Corpus: ("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

Step 1: Split into characters
  ("h" "u" "g", 10), ("p" "u" "g", 5), ...

Step 2: Most frequent pair ("u", "g") = 20 times

Step 3: Merge -> "ug"
  Vocabulary: ["b", "g", "h", "n", "p", "s", "u", "ug"]

Step 4: Repeat...
```

---

## WordPiece

### Key Difference from BPE

Uses a score instead of raw frequency:

```text
score = freq(pair) / (freq(first) * freq(second))
```

Prioritizes merging rare pairs over common ones.

### WordPiece Example

Uses `##` prefix for non-initial subwords:

```text
("h" "##u" "##g", 10), ("p" "##u" "##g", 5), ...
```

---

## Unigram

### Key Difference

- Starts with **large** vocabulary
- **Removes** tokens with lowest loss impact
- Uses probabilistic scoring

### Algorithm

1. Initialize with all possible substrings
2. Compute loss over corpus
3. Remove 10-20% tokens with lowest loss impact
4. Repeat until target size

---

## Building Tokenizers from Scratch

### Components

```text
Tokenizer Pipeline:
1. Normalizer    -> Text cleanup
2. Pre-tokenizer -> Split into words
3. Model         -> BPE / WordPiece / Unigram
4. Post-processor -> Add special tokens
5. Decoder       -> Convert IDs back to text
```

### BPE Tokenizer (GPT-2 style)

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Initialize
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Train
trainer = trainers.BpeTrainer(
    vocab_size=25000,
    special_tokens=["[EOS]"]  # end-of-text token
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Post-processing and decoder
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

# Save
tokenizer.save("tokenizer.json")
```

### WordPiece Tokenizer (BERT style)

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

# Initialize
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Normalization
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])

# Pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Train
trainer = trainers.WordPieceTrainer(
    vocab_size=30000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

### Unigram Tokenizer (T5/XLNet style)

```python
from tokenizers import Tokenizer, models, trainers

# Initialize
tokenizer = Tokenizer(models.Unigram())

# Train
trainer = trainers.UnigramTrainer(
    vocab_size=20000,
    special_tokens=["[PAD]", "[BOS]", "[EOS]"],
    unk_token="[UNK]"
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```

### Wrap for Transformers

```python
from transformers import PreTrainedTokenizerFast

# Wrap tokenizers library tokenizer for use with Transformers
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Now use like any HF tokenizer
wrapped_tokenizer.save_pretrained("./my-tokenizer")
```

---

## Quick Reference

### Training from Existing Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
new_tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=50000)
```

### Fast Tokenizer Features

| Method | Description |
| ------ | ----------- |
| `encoding.tokens()` | Get token strings |
| `encoding.word_ids()` | Map tokens to words |
| `encoding.word_to_chars(i)` | Get char span for word i |
| `encoding.token_to_chars(i)` | Get char span for token i |
| `encoding.char_to_token(i)` | Get token index for char i |

### Tokenizer Components

| Component | Purpose |
| --------- | ------- |
| `normalizer` | Text cleanup (lowercase, accents) |
| `pre_tokenizer` | Split into words |
| `model` | Subword tokenization |
| `post_processor` | Add special tokens |
| `decoder` | Convert back to text |

### Algorithm Comparison

| | BPE | WordPiece | Unigram |
| --- | --- | --------- | ------- |
| **Training** | Merge frequent pairs | Merge by score | Remove low-impact |
| **Direction** | Bottom-up | Bottom-up | Top-down |
| **Used by** | GPT-2, RoBERTa | BERT | T5, XLNet |
