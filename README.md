# 🦙 SmolLM2-135M: Reverse Engineered Implementation

A **from-scratch PyTorch implementation** of the **SmolLM2-135M** language model.  
This project reverse-engineers the architecture from **GGUF metadata**, recreates it using modern components (Flash Attention, AMP), and includes an experiment demonstrating how removing optimizer states affects training stability.

---

# 🧠 1. Model Architecture

SmolLM2 follows the **modern LLaMA-style architecture**, not GPT-2.  
Below is the structural breakdown in both diagram form and detailed explanation.

---

##  Architecture

```mermaid
graph TD
    subgraph "Global Structure"
        Input[Input IDs] --> Embed[Embedding Layer (Wt)]
        Embed --> Block0[Llama Block 0]
        Block0 --> Block1[...]
        Block1 --> Block29[Llama Block 29]
        Block29 --> RMS_F[RMSNorm Final]
        RMS_F --> Head[LM Head (Wt)]
        Head --> Output[Logits]
    end

    subgraph "Inside a Block (x30)"
        direction TB
        x[Input x] --> N1[RMSNorm]
        N1 --> GQA[Grouped Query Attention]
        GQA --> Add1((+))
        x --> Add1

        Add1 --> N2[RMSNorm]
        N2 --> MLP[SwiGLU FeedForward]
        MLP --> Add2((+))
        Add1 --> Add2
    end

    style Embed fill:#f9f,stroke:#333,stroke-width:2px
    style Head fill:#f9f,stroke:#333,stroke-width:2px
    style GQA fill:#bbf,stroke:#333
    style MLP fill:#bbf,stroke:#333
```

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

| Component      | Tensor Shape          | Parameters     | % of Model | Details |
|----------------|------------------------|---------------:|-----------:|---------|
| **Embeddings** | `[49152, 576]`         | 28,311,552     | 21.05%     | Vocabulary × Hidden Dim |
| **Attention**  | Q, K, V, O             | 26,542,080     | 19.73%     | 9 Q Heads + 3 KV Heads × 30 layers |
| **MLP (SwiGLU)** | Gate, Up, Down       | 79,626,240     | 59.20%     | The "brain" of the model |
| **Normalization** | `[576]`             | 35,136         | 0.03%      | RMSNorm per block |
| **TOTAL**      | —                      | **134,515,008** | **100%** | ~135M |

---

# 🏁 Summary

This README provides:

✔ A clean breakdown of SmolLM2-135M architecture  
✔ Mermaid diagram of computation flow  
✔ Full explanation of GQA, SwiGLU, RMSNorm  
✔ Exact parameter distribution (reverse-engineered)  

---

If you'd like, I can also generate:

📦 A downloadable PyTorch implementation  
📈 Architecture diagrams as PNG  
🧪 Training scripts + stability experiment  
📘 HuggingFace model card style README  

Just ask!
