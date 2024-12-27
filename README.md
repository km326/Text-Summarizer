# Text Summarizer

This project implements a **Text Summarizer** using the **T5 (Text-to-Text Transfer Transformer)** model from Hugging Face's `transformers` library. The summarizer takes an input text and generates a concise summary based on user-specified length constraints. It’s designed to be beginner-friendly and easy to use.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Example Input and Output](#example-input-and-output)
- [Customization](#customization)
- [Credits](#credits)

---

## Features

- Summarizes long pieces of text into concise summaries.
- Customizable summary lengths (minimum and maximum).
- Beginner-friendly structure with clear, step-by-step explanations in the code.
- Uses the **T5-small model** for text summarization, which is efficient and pre-trained for various NLP tasks.

---

## How It Works

1. **Input Preprocessing**: 
   - Cleans the input text and adds a "summarize:" prefix for the T5 model to understand the task.

2. **Tokenization**: 
   - Converts the input text into tokens (numerical representations) using the T5 tokenizer.

3. **Model Inference**:
   - Passes the tokenized input through the T5 model to generate the summary.

4. **Output Decoding**:
   - Converts the generated tokens back into human-readable text.

---

## Requirements

- Python 3.7 or later
- The following Python libraries:
  - `torch`
  - `transformers`
  - `sentencepiece`

---
