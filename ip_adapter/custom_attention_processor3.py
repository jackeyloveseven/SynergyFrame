# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image

# --- Previous classes remain unchanged ---

class AttnProcessor(nn.Module):
    def __init__(self, hidden_size=None, cross_attention_dim=None):
        super().__init__()
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class AttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size=None, cross_attention_dim=None,):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0.")
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class IPAttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, skip=False):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0.")
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.skip = skip
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            text_encoder_hidden_states, ip_hidden_states = torch.split(encoder_hidden_states, [encoder_hidden_states.shape[1] - self.num_tokens, self.num_tokens], dim=1)
            if attn.norm_cross:
                text_encoder_hidden_states = attn.norm_encoder_hidden_states(text_encoder_hidden_states)
        key = attn.to_k(text_encoder_hidden_states)
        value = attn.to_v(text_encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if not self.skip:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_hidden_states = F.scaled_dot_product_attention(query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False)
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)
            hidden_states = hidden_states + self.scale * ip_hidden_states
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class CNAttnProcessor2_0(torch.nn.Module):
    def __init__(self, num_tokens=4):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0.")
        self.num_tokens = num_tokens
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states[:, :encoder_hidden_states.shape[1] - self.num_tokens]
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

class SemanticMaskedStyleAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("This processor requires PyTorch 2.0.")
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.raw_spatial_mask = None

    def set_spatial_mask(self, mask_image: Image.Image):
        self.raw_spatial_mask = mask_image.convert("L")

    def clear_spatial_mask(self):
        self.raw_spatial_mask = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        raise NotImplementedError(
            "SemanticMaskedStyleAttnProcessor is a base class and should not be called directly. "
            "Please use a derived class like SemanticClipAttnProcessor instead."
        )

# ----------------- MODIFICATION START -----------------
# This is the new, robust, and CORRECT implementation of SemanticClipAttnProcessor
class SemanticClipAttnProcessor(SemanticMaskedStyleAttnProcessor):
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, semantic_scale=0.5):
        super().__init__(hidden_size, cross_attention_dim, scale, num_tokens)
        self.semantic_scale = semantic_scale
        self.clip_text_model = None
        self.tokenizer = None

    def set_clip_models(self, clip_text_model, tokenizer):
        self.clip_text_model = clip_text_model
        self.tokenizer = tokenizer
        if self.clip_text_model is not None:
            self.clip_text_model.eval()
            for param in self.clip_text_model.parameters():
                param.requires_grad = False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, sequence_length, _ = hidden_states.shape
            height = width = int(math.sqrt(sequence_length))

        # --- Device and dtype enforcement ---
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Split encoder_hidden_states
        if encoder_hidden_states is None:
            text_encoder_hidden_states = hidden_states
            ip_hidden_states = None
        else:
            text_len = encoder_hidden_states.shape[1] - self.num_tokens
            text_encoder_hidden_states, ip_hidden_states = torch.split(encoder_hidden_states, [text_len, self.num_tokens], dim=1)

        # --- Main Query ---
        query = attn.to_q(hidden_states) # Shape: [B, L_q, D_inner]
        inner_dim = attn.to_k.out_features
        head_dim = inner_dim // attn.heads
        
        # --- Branch 1: Standard Text-conditioned Generation ---
        key_text = attn.to_k(text_encoder_hidden_states).to(device, dtype=dtype)
        value_text = attn.to_v(text_encoder_hidden_states).to(device, dtype=dtype)
        
        query_heads = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_text_heads = key_text.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_text_heads = value_text.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        text_hidden_states = F.scaled_dot_product_attention(
            query_heads, key_text_heads, value_text_heads, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        text_hidden_states = text_hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        # --- Branch 2: IP-Adapter with Semantic-Aware Sparse Attention (CORRECTED) ---
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states).to(device, dtype=dtype)
            ip_value = self.to_v_ip(ip_hidden_states).to(device, dtype=dtype)
            ip_key_heads = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value_heads = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # 2a. Standard attention scores
            attention_scores = (query_heads @ ip_key_heads.transpose(-1, -2)) * attn.scale

            # 2b. Semantic guidance in CLIP space (training-free and FINAL CORRECTED VERSION)
            with torch.no_grad():
                # Step 1: Get a SINGLE semantic vector for each query token (NO HEADS)
                q_proj = query # Use the original query before splitting into heads
                k_proj = key_text # Use the projected text keys
                
                text_attn_probs = torch.softmax((q_proj @ k_proj.transpose(-1, -2)) * attn.scale, dim=-1)
                query_semantic_vecs = text_attn_probs @ text_encoder_hidden_states

                # Step 2: Get the semantic vectors for the IP-Adapter image patches
                ip_semantic_vecs = ip_hidden_states

                # Step 3: Calculate the semantic similarity matrix using RAW DOT PRODUCT
                # NO NORMALIZATION - THIS IS THE KEY FIX
                # We use the raw dot product to preserve the magnitude of the semantic signal
                # and scale it fatores to match the attention scores' scale.
                semantic_matrix = (query_semantic_vecs @ ip_semantic_vecs.transpose(-1, -2)) * attn.scale

                # Step 4: Reshape the matrix to match the heads dimension for fusion
                semantic_matrix_expanded = semantic_matrix.unsqueeze(1).repeat(1, attn.heads, 1, 1)

            # 2c. Fuse scores for sparse attention
            fused_attention_scores = attention_scores + self.semantic_scale * semantic_matrix_expanded
            attention_probs = torch.softmax(fused_attention_scores, dim=-1)

            # 2d. Apply spatial mask (if provided)
            if self.raw_spatial_mask is not None:
                mask = self.raw_spatial_mask.resize((width, height), Image.LANCZOS)
                mask_tensor = to_tensor(mask).to(device, dtype=dtype).squeeze(0)
                spatial_mask = mask_tensor.view(1, 1, height * width, 1)
                attention_probs = attention_probs * spatial_mask

            # 2e. Attend to IP values
            ip_hidden_states_out = (attention_probs @ ip_value_heads).transpose(1, 2).reshape(batch_size, -1, inner_dim)

            # --- Combination ---
            hidden_states = text_hidden_states + self.scale * ip_hidden_states_out
        else:
            hidden_states = text_hidden_states

        # --- Final Projection ---
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
# ----------------- MODIFICATION END -----------------