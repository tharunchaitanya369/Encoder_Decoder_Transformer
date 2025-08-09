# Portuguese-to-English Neural Machine Translation with Transformer in TensorFlow

This repository implements a Transformer-based Encoder-Decoder architecture for translating Portuguese sentences to English using TensorFlow 2.x.

The project is built from scratch — including positional encoding, masking, multi-head attention, encoder & decoder layers, and custom training loops — following the original ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) paper.

---

## 📌 Features

- Custom Transformer architecture in TensorFlow
- Multi-Head Attention and Positional Encoding implemented from scratch
- Encoder-Decoder attention mechanism
- Padding & Look-Ahead Masking
- Subword tokenization using TensorFlow Datasets
- Portuguese to English translation using the `ted_hrlr_translate` dataset
- Custom learning rate scheduler as per the Transformer paper

---



---

## 🧠 Key Concepts Explained

### 1. Tokenization

- Uses `SubwordTextEncoder` from TensorFlow Datasets to convert text into subword tokens.
- Special tokens:
    - `<start>` → Marks the beginning of a sentence
    - `<end>` → Marks the end of a sentence

**Example:**  
"eu gosto de maçãs" → [`<start>`, 17, 25, 450, 125, `<end>`]

### 2. Positional Encoding

Since Transformers lack recurrence or convolution, positional encoding injects information about word order using unique sinusoidal patterns:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 3. Masking

- **Padding Mask:** Ignores `<pad>` tokens in attention.
- **Look-Ahead Mask:** Prevents decoder from seeing future tokens during training.

### 4. Multi-Head Attention

- Computes attention in multiple subspaces (heads), capturing different relationships:
    1. Linear projections → Q (query), K (key), V (value)
    2. Split into `num_heads`
    3. Apply scaled dot-product attention
    4. Concatenate and project back

### 5. Encoder Layer

Each encoder layer contains:
- Multi-Head Self-Attention
- Feed Forward Network
- Residual Connection + Layer Normalization

### 6. Decoder Layer

Each decoder layer contains:
- Masked Multi-Head Self-Attention
- Encoder-Decoder Attention
- Feed Forward Network
- Residual Connection + Layer Normalization

### 7. Encoder-Decoder Attention

- Queries (Q) from the decoder attend to keys (K) and values (V) from the encoder output.
- Allows the decoder to focus on relevant parts of the input sentence.

### 8. Custom Learning Rate Scheduler

Implements the warmup strategy from the paper:
```
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```
Learning rate increases at first and then decays.

---

## 📊 Dataset

- Uses [`ted_hrlr_translate/pt_to_en`](https://www.tensorflow.org/datasets/community_catalog/huggingface/ted_hrlr_translate) from TensorFlow Datasets.
- Contains TED Talk transcripts in Portuguese-English sentence pairs.

---

