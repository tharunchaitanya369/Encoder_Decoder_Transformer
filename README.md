# Transformer (Portuguese → English) — Deep explanation & code mapping

**Repository goal.** Train a Transformer encoder–decoder in TensorFlow to translate Portuguese → English using the `ted_hrlr_translate/pt_to_en` dataset. This README explains what each core Transformer concept *is*, the *math* behind it, the *shape flows* inside your code, and important implementation notes.

---

# Quick start — what to run

Assumes your code is in `transformer_translation.py`.

```bash
pip install -r requirements.txt
python transformer_translation.py
```

---

# Table of contents

1. Overview: encoder–decoder Transformer (short)
2. Tokenization (subword) — how you did it, gotchas
3. Data pipeline — shapes, padding, filters
4. Embeddings & Positional Encoding — theory, math, numeric example
5. Attention: scaled dot-product + mask — math and numeric worked example
6. Multi-Head Attention — shapes, code mapping
7. Encoder layer — step-by-step (shapes & code)
8. Decoder layer — step-by-step (masked self-attn, encoder-decoder attn)
9. Masks — padding & look-ahead, how you create and combine them
10. Loss, metrics, optimizer, learning rate schedule — why and how
11. Training & evaluation (greedy decoding) — limitations & improvements
12. Common bugs/pitfalls and suggested fixes
13. Next steps & improvements

---

# 1 — Overview (short)

Your model implements the classic Transformer (Vaswani et al., 2017):

* **Encoder**: stack of `N` identical encoder layers. Each layer: self-attention → feed-forward → residual + layernorm.
* **Decoder**: stack of `N` decoder layers. Each decoder layer: masked self-attention → encoder–decoder attention → feed-forward → residual + layernorm.
* **Loss**: sparse categorical cross-entropy, masking pad tokens.
* **Decoding**: greedy (argmax) decoding in `evaluate()`.

---

# 2 — Tokenization (SubwordTextEncoder)

**Where in your code**

```py
token_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train), target_vocab_size=2**13)
token_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train), target_vocab_size=2**13)
```

**What it does**

* Builds subword vocabularies from the training corpus.
* Subword tokenizers split rare words into smaller units and common words as single tokens — helps generalize to OOV forms and reduces vocab size vs word-level.

**Special tokens in your code**

```py
start_token, end_token = [token_en.vocab_size], [token_en.vocab_size + 1]
pt_vocab_size = token_pt.vocab_size + 2
en_vocab_size = token_en.vocab_size + 2
```

You reserve two ids at the *end* of each vocabulary for `start` and `end`. Good to persist `token_pt` and `token_en` (save to disk) so you don’t need to rebuild them every run.

**Important pitfalls & recommendations**

* `tfds.deprecated.text.SubwordTextEncoder` is workable but deprecated. Consider `sentencepiece` or `tensorflow_text` tokenizers in production.
* Make sure **padding id** (the value used by `padded_batch`) does not clash with real tokens. `padded_batch` pads with `0` by default. If your tokenizer uses `0` as a real token, shift all token ids up by 1 to reserve 0 for PAD. (Your code follows the TF tutorial; double-check token id conventions in your runs.)

---

# 3 — Data pipeline (shapes & functions)

**Key steps (your code)**

```py
train_dataset = train.map(tf_encode)                # apply encoding
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None],[None]))
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
```

* `tf_encode` uses `tf.py_function` to wrap your `encode()` (so you end up with tensors shaped `[seq_len]`).
* `filter_max_length` ensures both `pt` and `en` sequences ≤ `MAX_LENGTH` (40).
* `padded_batch(BATCH_SIZE, padded_shapes=([None],[None]))` pads sequences *to the maximum length in that batch* using `0` padding.

**Common shape after batching**

* `inp` (Portuguese) shape: `(BATCH_SIZE, seq_len_in)` — e.g., `(64, 24)` for that batch.
* `tar` (English) shape: `(BATCH_SIZE, seq_len_out)`.

Your `create_padding_mask(seq)` checks `seq == 0` to build pads mask. That means pad token is `0`. Ensure tokenizer mapping respects that.

---

# 4 — Embeddings & Positional Encoding (detailed math + numeric example)

**Why positional encoding?**

* Transformers see tokens in parallel — no recurrence — so we must give a signal encoding token *position*.
* Sinusoidal PE encodes absolute/inferable relative positions and works without learned embeddings (deterministic).

**Your implementation**

```py

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
```

**Mathematical formulas**

For position `pos` and dimension `i` (0-indexed):

* $	ext{angle
ate}(i) = 1 / 10000^{rac{2 loor{i/2}}{d_{model}}}$
* $	ext{PE}_{pos,2k} = 	ext{sin}(pos 	imes 	ext{angle
ate}(2k))$
* $	ext{PE}_{pos,2k+1} = 	ext{cos}(pos 	imes 	ext{angle
ate}(2k+1))$

**Numeric worked example** (small numbers so you can follow)

Take `d_model = 4` and `pos = 2`. Compute `angle_rads` for i = 0,1,2,3.

* For i=0:

  * `i//2 = 0`.
  * exponent = (2 * 0) / 4 = 0 / 4 = 0.
  * 10000^0 = 1.
  * angle
ate = 1 / 1 = 1.
  * angle
ad = pos * angle
ate = 2 * 1 = 2.
  * since i is even (0), PE = sin(2) ≈ 0.9092974268.

* For i=1:

  * `i//2 = 0` → same angle
ate = 1.
  * angle
ad = 2 * 1 = 2.
  * since i is odd (1), PE = cos(2) ≈ -0.4161468365.

* For i=2:

  * `i//2 = 1`.
  * exponent = (2 * 1) / 4 = 2 / 4 = 0.5.
  * 10000^0.5 = sqrt(10000) = 100.
  * angle
ate = 1 / 100 = 0.01.
  * angle
ad = pos * angle
ate = 2 * 0.01 = 0.02.
  * i even (2), PE = sin(0.02) ≈ 0.0199986667.

* For i=3:

  * `i//2 = 1` → same angle
ate 0.01.
  * angle
ad = 0.02.
  * i odd (3), PE = cos(0.02) ≈ 0.9998000067.

So the positional encoding vector at `pos=2`, `d_model=4` is approximately:

```
[ sin(2), cos(2), sin(0.02), cos(0.02) ] ≈ [0.9093, -0.4161, 0.0200, 0.9998]
```

**How it is used**

* In `Encoder.call` and `Decoder.call` you add `pos_encoding[:, :seq_len, :]` to embeddings (after scaling embeddings by `sqrt(d_model)`).

---

# 5 — Scaled dot-product attention (math + numeric example)

**Code** (your function)

```py

def dot_product_attention(q, k, v, mask=None):
    mul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = mul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_qk += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_qk, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
```

**Mathematical steps**

1. Compute raw scores: $	ext{scores} = Q K^T$.
2. Scale: $	ext{scaled} = 	ext{scores} / 	ext{sqrt}(d_k)$. (This reduces variance when `d_k` large.)
3. Apply mask: add `-inf` (implemented with `-1e9`) where mask==1.
4. Softmax along last axis (over `K` sequence length).
5. Weighted sum: $	ext{output} = 	ext{softmax}(	ext{scaled}) V$.

**Numeric example (worked step-by-step)**

Let `Q` length 1, `K` length 2, `d_k = 2`.

* $Q = [1, 0]$ (shape (1,2))
* $K_1 = [1, 0], K_2 = [0, 1]$
* $V_1 = [2, 1], V_2 = [0, 3]$

Compute scores `Q K^T`:

* score	ext{_}1 = 1*1 + 0*0 = 1
* score	ext{_}2 = 1*0 + 0*1 = 0
  → scores = 
* 1, 0

Scale: $	ext{sqrt}(d_k) = 	ext{sqrt}(2)$.

* Compute √2 step-by-step: $	ext{sqrt}(2) 	ext{≈} 1.4142135624$.
* scaled scores = 
* 1 / 1.4142135624, 0 / 1.4142135624 	ext{≈} 
* 0.70710678, 0.0

Softmax:

* exp(0.70710678) 	ext{≈} 2.0281149816  (approx)
* exp(0.0) = 1.0
* sum = 2.0281149816 + 1.0 = 3.0281149816
* softmax = 
* 2.0281149816 / 3.0281149816, 1.0 / 3.0281149816 	ext{≈} 
* 0.669430, 0.330570

Weighted sum:

* output = 0.669430 * V1 + 0.330570 * V2
* V1 = 
* 2, 1, 0.669430 * V1 = 
* 1.33886, 0.66943
* V2 = 
* 0, 3, 0.330570 * V2 = 
* 0.0, 0.99171
* output = 
* 1.33886 + 0.0, 0.66943 + 0.99171 = 
* 1.33886, 1.66114

So attention output 	ext{≈} `[1.33886, 1.66114]`.

**Masking effect**
If a position is masked (mask value 1) you add `-1e9` to that scaled score, making its softmax essentially zero.

---

# 6 — Multi-Head Attention (shapes & mapping to code)

**Where**
Class `MultiHeadAttention` and `split_heads()`.

**Key idea**
Project input into `Q`, `K`, `V` spaces, split into `num_heads` independent attention heads (different linear projections), compute attention in parallel, then concatenate heads.

**Dimension math (explicit)**

Given:

* `d_model = 128`
* `num_heads = 8`

Compute per-head depth:

1. `d_model` / `num_heads`:

   * 128 divided by 8:

     * 8 × 16 = 128 → `depth = 16`.

So each head works with `d_k = 16`.

**Shapes step-by-step for one call**

Inputs:

* Query `q` shape: `(B, seq_len_q, d_model)`
* Key `k` shape: `(B, seq_len_k, d_model)`
* Value `v` shape: `(B, seq_len_v, d_model)`

Inside `call()`:

1. After linear projections:

   * `q = self.wq(q)` → `(B, seq_len_q, d_model)`
   * same for `k`, `v`.
2. `split_heads` reshapes `x` from `(B, seq_len, d_model)` → `(B, num_heads, seq_len, depth)`

   * Steps inside `split_heads`:

     * `tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))` → `(B, seq_len, 8, 16)`
     * `tf.transpose(..., perm=[0,2,1,3])` → `(B, 8, seq_len, 16)`
3. `dot_product_attention(q, k, v, mask)` receives `q,k,v` with shape `(B, 8, seq_len_q, 16)` / `(B,8,seq_len_k,16)` / `(B,8,seq_len_v,16)` respectively.
4. `dot_product_attention` returns:

   * `scaled_attention` shape: `(B, 8, seq_len_q, 16)`
   * `attention_weights` shape: `(B, 8, seq_len_q, seq_len_k)`
5. `transpose` and `reshape` back to `(B, seq_len_q, d_model)`:

   * `tf.transpose(scaled_attention, perm=[0,2,1,3])` → `(B, seq_len_q, 8, 16)`
   * `tf.reshape(..., (B, seq_len_q, d_model))` → `(B, seq_len_q, 128)`

Finally `self.dense` projects back to `(B, seq_len_q, d_model)` (same shape).

**Attention weights you can visualize**

* `attention_weights` per head shape: `(B, num_heads, seq_len_q, seq_len_k)`.

  * Example: `decoder_layer1_block2` typically has shape `(B, num_heads, targ_seq_len, inp_seq_len)`.

---

# 7 — EncoderLayer (step-by-step, code mapping)

**Code (simplified):**

```py
attn_output, _ = self.mha(x, x, x, mask)
attn_output = self.dropout1(attn_output, training=training)
out1 = self.layernorm1(x + attn_output)
ffn_output = self.ffn(out1)   # point-wise feed forward
ffn_output = self.dropout2(ffn_output, training=training)
out2 = self.layernorm2(out1 + ffn_output)
```

**Shape flow example**

* Input `x` shape: `(B, seq_len, d_model)` e.g. `(64, 20, 128)`.
* After `mha`: `(64, 20, 128)`.
* Residual add `x + attn_output`: same shape.
* FFN (`Dense(dff) -> Dense(d_model)`): `(64, 20, 512) -> (64, 20, 128)`.
* Final `out2`: `(64, 20, 128)`.

**Why layernorm after residual?**

* The “Add & Norm” (residual + layernorm) stabilizes gradients and improves training.

---

# 8 — DecoderLayer (detailed)

**Structure (your code)**:

1. `mha1` — masked self-attention on decoder input `x` (uses `look_ahead_mask`).
2. Add & Norm (residual).
3. `mha2` — encoder–decoder attention: `Q = out1`, `K = enc_output`, `V = enc_output` (uses `padding_mask`).
4. Add & Norm.
5. FFN + Add & Norm.

**Shape flow**

* Decoder input `x`: `(B, targ_seq_len, d_model)`
* After `mha1` → `(B, targ_seq_len, d_model)`
* After `mha2` → `(B, targ_seq_len, d_model)`; but attention weights in `mha2` have shape `(B, num_heads, targ_seq_len, inp_seq_len)` (decoder positions attend to encoder positions).
* Feed-forward applies per position and returns `(B, targ_seq_len, d_model)`.

**Why two attentions?**

* First: allow the decoder to attend to previous outputs (masked to prevent peeking).
* Second: allow decoder positions to use encoded information from the source sentence.

---

# 9 — Masks (create, combine, shapes, why)

**Your `create_padding_mask(seq)`**

```py
seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
return seq[:, tf.newaxis, tf.newaxis, :]
```

* Input `seq` shape: `(B, seq_len)`
* `tf.math.equal(seq, 0)` returns boolean `(B, seq_len)` where `1` means pad.
* After `tf.float32` and expanding dims, mask shape = `(B, 1, 1, seq_len)`. This format broadcast-matches during attention score addition which expects shape `(B, num_heads, seq_len_q, seq_len_k)`.

**Your `create_look_ahead_mask(size)`**

```py
mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
```

* `tf.linalg.band_part` with `(-1, 0)` returns lower triangular matrix with ones on and below diagonal. `1 -` makes an upper triangular matrix of 1s where future positions are masked.
* Shape: `(size, size)` i.e., `(targ_seq_len, targ_seq_len)`.

**Combining**

* You create `dec_target_padding_mask` for target padding: `(B, 1, 1, targ_seq_len)`. 
* `look_ahead_mask` is broadcast-combined with `dec_target_padding_mask` via `tf.maximum(dec_target_padding_mask, look_ahead_mask)`. `tf.maximum` ensures that any pad positions or future positions are masked (1 in either → 1 in output).

**Why `-1e9` addition?**

* When a mask value is 1 for a position, you add a very large negative number so after softmax that probability ≈ 0.

---

# 10 — Loss, metrics, optimizer, learning rate schedule

**Loss**

```py
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
```

* `from_logits=True` because your final layer outputs raw logits.
* `loss_function` masks pad tokens by creating a boolean mask `real != 0` and then averaging over non-pad tokens.

**Accuracy**

* `accuracy_function` uses `tf.argmax(pred, axis=2)` and compares to `real`, then masks pads.

**Custom optimizer schedule**

```py
class CustomSchedule(LearningRateSchedule):
    return tf.math.rsqrt(d_model) * tf.math.minimum(rsqrt(step), step * warmup_steps**-1.5)
```

* I.e. $	ext{lr} = d_{model}^{-0.5} 	imes 	ext{min}(	ext{step}^{-0.5}, 	ext{step} 	imes 	ext{warmup}^{-1.5})$
* Warmup increases learning rate for first `warmup_steps`, then decays as $	ext{step}^{-0.5}$.

**Why this schedule?**

* Stabilizes training: small updates at first, then gradually increase to avoid large gradients early on, then decay.

---

# 11 — Training & evaluation

**Training step (`train_step`)**

* `tar_inp = tar[:, :-1]` and `tar_real = tar[:, 1:]`. This is teacher forcing: feed the decoder the ground-truth tokens shifted right.
* `transformer([inp, tar_inp], training=True)` returns logits shape `(B, targ_seq_len-1, target_vocab_size)`.
* Compute `loss_function(tar_real, predictions)` using mask to ignore PAD tokens.
* Compute gradients with `GradientTape` and `optimizer.apply_gradients()`.

**Evaluation / Greedy decode (`evaluate`)**

* Start with `decoder_input = [start_token]` and iteratively:

  * run transformer, take `argmax` of `predictions[:, -1, :]` to get token id
  * stop if `end_token`
* This is **greedy decoding**. Pros: simple, deterministic. Cons: may produce lower-quality translations than beam search.

**Improvements**

* Implement **beam search** with length normalization.
* Add **label smoothing** to loss to avoid over-confident outputs.
* Use **BLEU** or **sacreBLEU** for evaluation (not just printing translations).

---
