import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from einops import rearrange

class AttentionStore:
    def __init__(self):
        self.attention_maps = []

    def save_attention(self, attn_map):
        self.attention_maps.append(attn_map.detach().cpu())

    def reset(self):
        self.attention_maps = []

    def get_average_attention(self):
        if not self.attention_maps:
            return None
        # stack and average across layers
        # each map is [batch, heads, seq_q, seq_k]
        return torch.stack(self.attention_maps).mean(dim=0)

class VizAttentionProcessor:
    def __init__(self, store):
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None, # 显式添加此参数以消除 LTX 警告
        n_view=1,
        **kwargs
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # Handle multiview concatenation if needed
        # In ActionTransformerBlock, n_view is passed as 1, but encoder_hidden_states 
        # is already concatenated across views: [B, V*L, C]
        
        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # Compute attention scores manually to store them
        # scale = 1 / sqrt(head_dim)
        scale = query.shape[-1] ** -0.5
        attn_probs = (query @ key.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            # Prepare mask if provided
            attn_probs = attn_probs + attention_mask

        attn_probs = attn_probs.softmax(dim=-1)
        
        # Store the attention map
        self.store.save_attention(attn_probs)

        hidden_states = attn_probs @ value
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def visualize_attention_on_images(
    images, # List of PIL images or numpy arrays [V, H, W, 3]
    attention_map, # [heads, seq_q, seq_k]
    save_path,
    action_idx=0, # which action token to visualize
    grid_size=(24, 32), # spatial resolution of latents
    n_view=3
):
    """
    images: List of 3 images [camera1, camera2, tactile]
    attention_map: [heads, seq_q, V*L]
    """
    heads, seq_q, seq_k = attention_map.shape
    L = seq_k // n_view
    
    # Average across heads
    avg_attn = attention_map.mean(dim=0) # [seq_q, V*L]
    
    # Pick the specific action token's attention
    action_attn = avg_attn[action_idx] # [V*L]
    
    # Split by view
    view_attns = action_attn.view(n_view, L)
    
    fig, axes = plt.subplots(1, n_view, figsize=(5 * n_view, 5))
    if n_view == 1:
        axes = [axes]

    titles = ["Camera 1", "Camera 2", "Tactile"]
    
    for i in range(n_view):
        img = images[i]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).detach().cpu().to(torch.float32).numpy()
            img = (img + 1) / 2.0 # Assuming [-1, 1] range
            img = (img * 255).astype(np.uint8)
        
        attn = view_attns[i].view(*grid_size).detach().cpu().to(torch.float32).numpy()
        
        # Normalize heatmap
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # Upsample heatmap to image size
        from scipy.ndimage import zoom
        h, w = img.shape[:2]
        heatmap = zoom(attn, (h / grid_size[0], w / grid_size[1]))
        
        axes[i].imshow(img)
        axes[i].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[i].set_title(titles[i] if i < len(titles) else f"View {i}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attention visualization to {save_path}")
