🦙 SmolLM2-135M: Reverse Engineered Implementation

A "from-scratch" PyTorch implementation of the SmolLM2-135M language model. This repository reverse-engineers the architecture based on GGUF metadata, implements it using modern techniques (Flash Attention, AMP), and demonstrates the effects of optimizer state loss through a stability experiment.

🧠 1. Model Architecture

Unlike older models (like GPT-2), SmolLM2 uses the modern Llama architecture. Below is the structural breakdown of how the code is implemented.

The Diagram

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


(Note: Wt indicates Weight Tying: The Embedding and LM Head share the same memory to save ~28M parameters.)

Key Architectural Features

GQA (Grouped Query Attention):

Instead of having 9 Key/Value heads (which is slow), we use 3 KV Heads shared across 9 Query Heads.

Benefit: Drastically reduces memory bandwidth during inference.

SwiGLU MLP:

Replaces the standard GeLU feed-forward network.

It uses 3 linear projections (Gate, Up, Down) instead of 2.

Formula: $F(x) = (Swish(xW_g) \otimes xW_u)W_d$

RMSNorm (Root Mean Square Norm):

Used instead of LayerNorm. It re-scales the vector based on root-mean-square without centering the mean.

Benefit: More stable training for deep networks (30 layers).

📊 2. Parameter Breakdown (The Stats)

We analyzed the specific tensor sizes in the checkpoint. Here is the exact distribution of the 135M parameters.

Component

Tensor Shape

Parameters

% of Model

Details

Embeddings

[49152, 576]

28,311,552

21.05%

Vocabulary × Hidden Dim

Attention

Q,K,V,O

26,542,080

19.73%

9 Heads + 3 KV Heads (x30 Layers)

MLP (SwiGLU)

Gate,Up,Down

79,626,240

59.20%

The "Brain" (x30 Layers)

Normalization

[576]

35,136

0.03%

Pre & Post Norms

TOTAL



134,515,008

100%

~135 Million

Note on Efficiency: The MLP layers consume ~60% of the parameter budget, which is typical for Llama models using SwiGLU.

⚡ 3. Optimizations Used

To train a 30-layer model on a consumer/Kaggle T4 GPU (16GB VRAM), we implemented several "Speed Demon" optimizations:

High Precision MatMul: Enabled torch.set_float32_matmul_precision('high') to utilize Tensor Cores on Ampere GPUs.

Flash Attention 2: Replaced manual attention math with the F.scaled_dot_product_attention kernel for $O(N^2)$ memory savings.

Automatic Mixed Precision (AMP): The training loop runs in bfloat16 / float16, keeping weights in float32 only when necessary for stability.

Gradient Scaling: Used torch.amp.GradScaler to prevent underflow during mixed precision training.

🧪 4. The "Cold Restart" Experiment

We performed a stability test to demonstrate how optimizers work internally.

The Protocol

Phase 1: Train the model for 5000 steps. The optimizer builds up internal state (momentum and variance buffers).

The "Crash": We simulate a crash by saving the model weights but deleting the optimizer state.

Phase 2: We resume training with a fresh AdamW optimizer.

The Result

When the training resumes, the Loss Spikes significantly.

Why? AdamW maintains a "momentum" buffer (a moving average of past gradients). When we delete this, the optimizer forgets the direction it was moving and the curvature of the loss landscape. It treats the trained model as a starting point but takes erratic steps until it rebuilds momentum.

🛠️ 5. Usage

Training

To reproduce the training run (approx 45 mins on T4 GPU):

python train.py


Inference

To load the trained model and generate text:

import torch
from model import SmolLM2, SmolLM2Config

# 1. Setup
config = SmolLM2Config()
model = SmolLM2(config)

# 2. Load Weights
model.load_state_dict(torch.load("smollm2_final.pth"))
model.cuda().eval()

# 3. Generate
input_ids = torch.tensor([[464]]).cuda() # Token for "The"
out = model.generate(input_ids, max_new_tokens=50)
print(out)


📂 Repository Structure

train.py: The main training script (includes model definition and training loop).

smollm2_final.pth: The trained checkpoint (not included in repo, generated after training).

README.md: Project documentation.
