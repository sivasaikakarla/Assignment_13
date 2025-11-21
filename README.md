#  SmolLM2-135M: Implementation

A **from-scratch PyTorch implementation** of the **SmolLM2-135M** language model.  
recreates Model using modern components (Flash Attention, AMP), and includes an experiment demonstrating how removing optimizer states affects training stability.

---

#  1. Model Architecture

Embedding → 30 Transformer Blocks → Final RMSNorm → LM Head → Logits

Token Embedding (Weight-Tied):
Input token IDs are mapped to dense vectors using an embedding matrix that is later shared with the LM head.

30× Transformer Blocks:
The embedded sequence is passed through 30 identical Llama-style decoder blocks.

RMSNorm → Grouped Query Attention:
Each block first applies RMSNorm, then performs GQA attention (many Query heads, fewer shared KV heads), followed by a residual connection.

RMSNorm → SwiGLU FeedForward:
A second RMSNorm is applied, followed by a SwiGLU MLP (Gate, Up, Down projections), plus another residual connection.

Final RMSNorm:
After all 30 blocks, a final RMSNorm stabilizes the output representation.

LM Head (Weight-Tied) → Logits:
The normalized hidden state is passed through the LM head (same weights as embeddings) to produce vocabulary logits.


### 🔗 **Note:**  
**Embedding Layer** and **LM Head** share weights (`Wt`), saving ~28M parameters.

---

# 🧩 Key Architectural Features

## 🔵 1. Grouped Query Attention (GQA)

Instead of giving each attention head its own Key/Value projections:

- **9 Query Heads**
- **3 Key/Value Heads** shared across them

### ✅ Benefit:
- Huge reduction in KV-cache size  
- Faster inference  
- Lower memory bandwidth usage  

---

## 🟣 2. SwiGLU MLP

Replaces the older **GeLU FFN**.

It uses **three projections**:

- Gate
- Up
- Down

### Formula:
\[
F(x) = (Swish(xW_g) \otimes xW_u)W_d
\]

### Why SwiGLU?
- Better performance per parameter  
- Smoother gradients  
- Now almost standard in modern LLMs  

---

## 🟢 3. RMSNorm

Used instead of LayerNorm.

### RMSNorm:
\[
	ext{RMSNorm}(x) = rac{x}{	ext{RMS}(x)} \cdot w
\]

### Benefits:
- Removes costly mean-centering  
- More stable for deep networks (SmolLM2 has **30 layers**)  
- Simpler + faster  

---

# 📊 2. Parameter Breakdown (Exact Stats)

We parsed the GGUF tensor metadata and counted every single tensor element.

| Component      | Parameters     | % of Model | Details |
|----------------|---------------:|-----------:|---------|
| **Embeddings** | 28,311,552     | 21.05%     | Vocabulary × Hidden Dim |
| **Attention**  | 26,542,080     | 19.73%     | 9 Q Heads + 3 KV Heads × 30 layers |
| **MLP (SwiGLU)** | 79,626,240     | 59.20%     | The "brain" of the model |
| **Normalization** | 35,136         | 0.03%      | RMSNorm per block |
| **TOTAL**       | **134,515,008** | **100%** | ~135M |

---


This is the Model Architecture reference link (https://ollama.com/library/smollm2:135m/blobs/f535f83ec568)
