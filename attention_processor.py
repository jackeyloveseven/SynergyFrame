import torch
from torch import nn
from diffusers.models.attention_processor import Attention

class CrossAttentionSwapProcessor:
    """
    An attention processor that swaps the key and value vectors from a texture prompt
    into the attention calculation for a structure prompt. This allows for detailed
    textural control over a predefined structure.
    """
    def __init__(self, texture_prompt_seq_len):
        self.texture_prompt_seq_len = texture_prompt_seq_len

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, ip_adapter_image_embeds=None):
        batch_size, sequence_length, _ = hidden_states.shape

        # If encoder_hidden_states is None, we are in a self-attention layer.
        # Fall back to the default behavior.
        if encoder_hidden_states is None:
            # Standard self-attention
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

        # Isolate the texture part from the concatenated prompt embeddings.
        texture_embeds = encoder_hidden_states[:, -self.texture_prompt_seq_len :, :]

        # Integrate IP-Adapter embeddings if they are provided.
        # This allows the IP-Adapter image to contribute to the texture/style.
        if ip_adapter_image_embeds is not None:
            # The ip_adapter_image_embeds are passed as a list, take the first element
            ip_embeds = ip_adapter_image_embeds[0]
            kv_source = torch.cat([texture_embeds, ip_embeds], dim=1)
        else:
            kv_source = texture_embeds

        # Standard attention mechanism setup
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        # The query is from the image latents (representing the structure).
        # The key and value are derived from the TEXTURE prompt and IP-Adapter embeddings.
        key = attn.to_k(kv_source)
        value = attn.to_v(kv_source)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        # Standard attention computation
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # Dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
