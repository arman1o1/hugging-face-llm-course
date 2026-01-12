# Chapter 9: Introduction to Gradio

This chapter covers how to build interactive demos for your ML models using Gradio.

---

## Overview

Gradio lets you build, customize, and share web-based demos for any ML model, entirely in Python.

```text
                    ┌─────────────────────────┐
                    │     Gradio Concepts     │
                    └─────────────────────────┘
                               │
    ┌──────────────┬───────────┼───────────┬──────────────┐
    ▼              ▼           ▼           ▼              ▼
┌────────┐   ┌──────────┐  ┌───────┐  ┌────────┐   ┌──────────┐
│Interface│  │Components│  │Sharing│  │HF Hub  │   │ Blocks   │
│  Class │   │ I/O      │  │ Demos │  │  Integ │   │  Layout  │
└────────┘   └──────────┘  └───────┘  └────────┘   └──────────┘
```

**Why build ML demos?**

- Present work to non-technical audiences
- Reproduce ML model behavior easily
- Identify and debug failure points
- Discover algorithmic biases

---

## 1. Building Your First Demo

### Installation

```bash
pip install gradio
```

### Hello World Example

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

**What's happening:**

| Component | Description |
| --------- | ----------- |
| `fn` | The function to wrap |
| `inputs` | Input component type (e.g., "text", "image") |
| `outputs` | Output component type |
| `launch()` | Starts the web server |

### Customizing Input Components

```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

# Customize the textbox
textbox = gr.Textbox(label="Type your name:", placeholder="John Doe", lines=2)

gr.Interface(fn=greet, inputs=textbox, outputs="text").launch()
```

---

## 2. Including Model Predictions

### Text Generation with GPT-2

```python
from transformers import pipeline
import gradio as gr

model = pipeline("text-generation")

def predict(prompt):
    completion = model(prompt, max_length=50)[0]["generated_text"]
    return completion

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
```

---

## 3. Understanding the Interface Class

### Core Parameters

```python
gr.Interface(fn, inputs, outputs, ...)
```

| Parameter | Description |
| --------- | ----------- |
| `fn` | Prediction function |
| `inputs` | Input component(s): `"text"`, `"image"`, `"audio"`, etc. |
| `outputs` | Output component(s): `"text"`, `"label"`, `"image"`, etc. |

### Common Input/Output Types

| Type | Use Case |
| ---- | -------- |
| `"text"` / `"textbox"` | Text input/output |
| `"image"` | Image upload |
| `"audio"` | Audio files or microphone |
| `"label"` | Classification labels |
| `"file"` | Any file type |

### Audio Example: Reverse Audio

```python
import numpy as np
import gradio as gr

def reverse_audio(audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

mic = gr.Audio(sources=["microphone"], type="numpy", label="Speak here...")
gr.Interface(reverse_audio, mic, "audio").launch()
```

### Multiple Inputs and Outputs

```python
import numpy as np
import gradio as gr

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def generate_tone(note, octave, duration):
    sr = 48000
    a4_freq, tones_from_a4 = 440, 12 * (octave - 4) + (note - 9)
    frequency = a4_freq * 2 ** (tones_from_a4 / 12)
    duration = int(duration)
    audio = np.linspace(0, duration, duration * sr)
    audio = (20000 * np.sin(audio * (2 * np.pi * frequency))).astype(np.int16)
    return (sr, audio)

gr.Interface(
    generate_tone,
    [
        gr.Dropdown(notes, type="index"),
        gr.Slider(minimum=4, maximum=6, step=1),
        gr.Number(value=1, label="Duration in seconds"),
    ],
    "audio",
).launch()
```

### The `launch()` Method

| Parameter | Description |
| --------- | ----------- |
| `inline` | Display inline in notebooks |
| `inbrowser` | Open in new browser tab |
| `share` | Create public shareable link |

---

## 4. Sharing Demos

### Polishing Your Demo

```python
import gradio as gr

def predict(text):
    return f"Processed: {text}"

title = "My ML Demo"
description = "Enter text to see the model output."
article = "For more info, visit [our docs](https://example.com)."

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[["Hello world"], ["How are you?"]],  # Pre-filled examples
    # live=True,  # Uncomment for real-time updates
).launch()
```

### Optional Interface Parameters

| Parameter | Description |
| --------- | ----------- |
| `title` | Title above the interface |
| `description` | Description (supports Markdown/HTML) |
| `article` | Extended explanation below interface |
| `examples` | Pre-filled example inputs |
| `live` | Update output on every input change |

### Sharing with Temporary Links

```python
gr.Interface(fn, inputs, outputs).launch(share=True)
```

- Creates a public link valid for **72 hours**
- Link format: `XXXXX.gradio.app`
- Processing happens on your device

> ⚠️ **Warning:** Public links are accessible by anyone. Don't expose sensitive data!

### Hosting on Hugging Face Spaces

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Add your code in `app.py`
3. Push to the repo — your demo is live!

---

## 5. Hugging Face Hub Integration

### Loading Models from the Hub

```python
import gradio as gr

title = "GPT-J-6B"
description = "Demo for GPT-J 6B text generation model."

gr.load(
    "EleutherAI/gpt-j-6B",
    src="models",
    inputs=gr.Textbox(lines=5, label="Input Text"),
    title=title,
    description=description,
).launch()
```

> Uses Hugging Face's **Inference API** — no need to load model in memory!
>
> **Note:** Use `src="models"` for HF models and `src="spaces"` for HF Spaces.

### Loading from Hugging Face Spaces

```python
import gradio as gr

# Load and recreate a Space locally
gr.load("abidlabs/remove-bg", src="spaces").launch()

# Customize loaded Space
gr.load(
    "abidlabs/remove-bg",
    src="spaces",
    inputs="webcam",
    title="Remove your webcam background!"
).launch()
```

---

## 6. Advanced Interface Features

### Session State (Chatbot Example)

```python
import random
import gradio as gr

def chat(message, history):
    history = history or []
    if message.startswith("How many"):
        response = str(random.randint(1, 10))
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, response))
    return history, history

gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
).launch()
```

**State pattern:**

1. Add `"state"` to inputs and outputs
2. Accept state as extra function parameter
3. Return updated state as extra return value

### Interpretation (Understanding Predictions)

```python
import gradio as gr

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=3),
    interpretation="default",  # Enable interpretation
).launch()
```

---

## 7. Introduction to Blocks

`Blocks` provides more control over layout and events than `Interface`.

### Simple Blocks Example

```python
import gradio as gr

def flip_text(x):
    return x[::-1]

with gr.Blocks() as demo:
    gr.Markdown("# Flip Text!")
    
    input_box = gr.Textbox(placeholder="Flip this text")
    output_box = gr.Textbox()
    
    input_box.change(fn=flip_text, inputs=input_box, outputs=output_box)

demo.launch()
```

### Key Blocks Concepts

| Concept | Description |
| ------- | ----------- |
| Context manager | Use `with gr.Blocks() as demo:` |
| Component order | Components render in order created |
| Events | Attach functions to component events |
| Auto-interactivity | Blocks detects which components need user input |

### Customizing Layout

```python
import numpy as np
import gradio as gr

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    
    with gr.Tabs():
        with gr.TabItem("Flip Text"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")
        
        with gr.TabItem("Flip Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")
    
    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()
```

### Layout Components

| Component | Description |
| --------- | ----------- |
| `gr.Row()` | Arrange components horizontally |
| `gr.Column()` | Arrange components vertically (default) |
| `gr.Tabs()` | Create tabbed interface |
| `gr.TabItem()` | Individual tab content |

### Events and State

```python
import gradio as gr

# Event types for Textbox
textbox.change(...)   # Value changes
textbox.submit(...)   # User presses Enter

# Event types for Button
button.click(...)     # Button clicked

# Event parameters
button.click(
    fn=my_function,        # Function to run
    inputs=[input1, ...],  # Input components
    outputs=[output1, ...] # Output components
)
```

### Multi-Step Demo (Chaining Models)

```python
from transformers import pipeline
import gradio as gr

asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")

def speech_to_text(speech):
    return asr(speech)["text"]

def text_to_sentiment(text):
    return classifier(text)[0]["label"]

with gr.Blocks() as demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()
    
    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")
    
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)

demo.launch()
```

---

## Quick Reference

### Interface vs Blocks

| Feature | Interface | Blocks |
| ------- | --------- | ------ |
| Simplicity | ✅ Simple, few lines | More code required |
| Layout control | Limited | ✅ Full control |
| Multiple functions | One main function | ✅ Multiple functions |
| Custom events | Limited | ✅ Fine-grained events |
| Best for | Quick demos | Complex apps |

### Common Components

| Component | Input/Output | Description |
| --------- | ------------ | ----------- |
| `Textbox` | Both | Text input/display |
| `Image` | Both | Image upload/display |
| `Audio` | Both | Audio upload/play |
| `Label` | Output | Classification labels |
| `Slider` | Input | Numeric slider |
| `Dropdown` | Input | Selection menu |
| `Button` | Input | Trigger events |
| `Markdown` | Output | Formatted text |
| `Chatbot` | Output | Chat history display |

### Essential Patterns

```python
# Basic Interface
gr.Interface(fn, inputs, outputs).launch()

# With customization
gr.Interface(fn, inputs, outputs, title=..., examples=...).launch()

# Sharing publicly
gr.Interface(...).launch(share=True)

# Load from Hub
gr.load("model-name", src="models")
gr.load("username/space-name", src="spaces")

# Basic Blocks
with gr.Blocks() as demo:
    # Add components
    demo.launch()
```

### Resources

| Resource | URL |
| -------- | --- |
| **Gradio Docs** | [gradio.app/docs](https://gradio.app/docs) |
| **HF Spaces** | [huggingface.co/spaces](https://huggingface.co/spaces) |
| **Gradio Guides** | [gradio.app/guides](https://gradio.app/guides) |
