import gradio as gr
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken
import os

# =============================================================================
# 1. MODEL DEFINITION (Matches your training script exactly)
# =============================================================================

@dataclass
class SmolLM2Config:
    block_size: int = 1024
    vocab_size: int = 49152
    n_layer: int = 30
    n_head: int = 9
    n_embd: int = 576
    intermediate_size: int = 1536
    n_kv_head: int = 3

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # GQA Expansion
        num_reps = self.n_head // self.n_kv_head
        k = k[:, :, :, None, :].expand(B, T, self.n_kv_head, num_reps, self.head_dim).reshape(B, T, self.n_head, self.head_dim)
        v = v[:, :, :, None, :].expand(B, T, self.n_kv_head, num_reps, self.head_dim).reshape(B, T, self.n_head, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class SmolLM2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(dict(
            embed_tokens = nn.Embedding(config.vocab_size, config.n_embd),
            layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, idx, targets=None):
        x = self.model.embed_tokens(idx)
        for block in self.model.layers:
            x = block(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =============================================================================
# 2. APP INITIALIZATION & INFERENCE
# =============================================================================

# Updated to match your specific saved file
MODEL_PATH = "smollm2_assignment13.pth"
device = "cpu" # Run on CPU for free tier Spaces (Model is small enough)

print("Initializing Model...")
config = SmolLM2Config()
model = SmolLM2(config)

if os.path.exists(MODEL_PATH):
    print(f"Loading weights from {MODEL_PATH}...")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading weights: {e}")
else:
    print(f"⚠️ Warning: {MODEL_PATH} not found. Upload it to the Space files.")

model.to(device)
model.eval()

try:
    enc = tiktoken.get_encoding('gpt2')
except:
    print("Tiktoken error. Ensure internet access is enabled in Space settings if needed.")

def generate_text(prompt, max_tokens, temperature):
    if not prompt:
        return "Please enter a prompt."
    
    try:
        # Encode
        input_ids = enc.encode(prompt)
        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(x, max_new_tokens=int(max_tokens), temperature=temperature)
            
        # Decode
        decoded_text = enc.decode(output_ids[0].tolist())
        return decoded_text
        
    except Exception as e:
        return f"Error: {str(e)}"

# =============================================================================
# 3. GRADIO UI
# =============================================================================

custom_css = """
.container { max_width: 800px; margin: auto; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦙 SmolLM2-135M Assignment Demo")
    gr.Markdown(f"Model loaded from: `{MODEL_PATH}`")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Prompt", placeholder="Once upon a time...", lines=5)
            max_tokens = gr.Slider(10, 200, value=50, step=10, label="Max Tokens")
            temp = gr.Slider(0.1, 2.0, value=0.8, label="Temperature")
            btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="Output", lines=10)

    btn.click(generate_text, inputs=[input_text, max_tokens, temp], outputs=output_text)

if __name__ == "__main__":
    demo.launch()