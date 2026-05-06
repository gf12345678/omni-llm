import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2RMSNorm,
    Qwen2_5OmniForConditionalGeneration as TransformersQwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniFlashAttention2 as TransformersQwen2_5OmniFlashAttention2,
    Qwen2_5OmniMLP,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration as TransformersQwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel as TransformersQwen2_5OmniThinkerTextModel,
    Qwen2_5OmniVisionAttention as TransformersQwen2_5OmniVisionAttention,
    Qwen2_5OmniVisionBlock as TransformersQwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionEncoder as TransformersQwen2_5OmniVisionEncoder,
    Qwen2_5OmniVisionFlashAttention2 as TransformersQwen2_5OmniVisionFlashAttention2,
    Qwen2_5OmniVisionSdpaAttention as TransformersQwen2_5OmniVisionSdpaAttention,
    _flash_attention_forward,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    flash_attn_varlen_func,
    repeat_kv,
)

from .pruning_utils import compute_segmented_token_importance


logger = logging.get_logger(__name__)


def _config_get(config, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _clamp_rate(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _use_fastv(config) -> bool:
    if config is None:
        return False
    return bool(_config_get(config, "use_fastv", True))


def _fastv_keep_rate(config) -> float:
    keep_rate = _config_get(config, "fastv_keep_rate", None)
    keep_rate = _config_get(config, "visual_keep_rate", keep_rate)
    keep_rate = _config_get(config, "video_keep_rate", keep_rate)
    keep_rate = _config_get(config, "keep_rate", keep_rate)
    if keep_rate is not None:
        return _clamp_rate(keep_rate)

    prune_ratio = _config_get(config, "fastv_r", None)
    prune_ratio = _config_get(config, "prune_ratio", prune_ratio)
    prune_ratio = _config_get(config, "rho_video", prune_ratio)
    if prune_ratio is None:
        prune_ratio = 0.5
    return 1.0 - _clamp_rate(prune_ratio)


def _slice_position_embeddings(position_embeddings, keep_mask: torch.Tensor):
    if position_embeddings is None:
        return None
    return tuple(embedding[:, :, keep_mask, :] for embedding in position_embeddings)


def _slice_sequence_tensor(tensor: Optional[torch.Tensor], keep_mask: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dim() == 2:
        return tensor[:, keep_mask]
    if tensor.dim() == 3:
        return tensor[:, :, keep_mask]
    if tensor.dim() == 4:
        return tensor[:, :, keep_mask, :][:, :, :, keep_mask]
    return tensor


def _build_fastv_decode_attention_mask(
    attention_mask: Optional[torch.Tensor],
    prompt_keep_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is None or prompt_keep_mask is None or attention_mask.dim() != 2:
        return None
    prompt_length = prompt_keep_mask.numel()
    if attention_mask.shape[-1] < prompt_length:
        return None

    prompt_keep_mask = prompt_keep_mask.to(attention_mask.device)
    prompt_mask = attention_mask[:, :prompt_length][:, prompt_keep_mask]
    generated_mask = attention_mask[:, prompt_length:]
    return torch.cat([prompt_mask, generated_mask], dim=-1)


def _fastv_scores_from_attention(attentions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if attentions is None:
        return None
    if attentions.dim() == 2:
        return attentions.float()
    if attentions.dim() == 3:
        return attentions[:, -1, :].float()
    if attentions.dim() != 4:
        return None
    return attentions.float().mean(dim=1)[:, -1, :]


def _select_fastv_keep_mask(
    token_scores: torch.Tensor,
    prunable_token_mask: torch.Tensor,
    fastv_config,
) -> Optional[torch.Tensor]:
    if token_scores.shape[0] != 1 or prunable_token_mask.shape[0] != 1:
        logger.warning_once("FastV token dropping currently supports batch_size=1; skip FastV pruning.")
        return None

    prunable_indices = torch.nonzero(prunable_token_mask[0], as_tuple=True)[0]
    num_prunable_tokens = prunable_indices.numel()
    if num_prunable_tokens <= 1:
        return None

    keep_prunable_count = int(round(num_prunable_tokens * _fastv_keep_rate(fastv_config)))
    keep_prunable_count = max(1, min(num_prunable_tokens, keep_prunable_count))
    if keep_prunable_count >= num_prunable_tokens:
        return None

    prunable_scores = token_scores[0, prunable_indices]
    kept_local = torch.topk(prunable_scores, k=keep_prunable_count, largest=True).indices
    kept_prunable_indices = prunable_indices[kept_local]

    keep_mask = torch.ones(prunable_token_mask.shape[1], device=prunable_token_mask.device, dtype=torch.bool)
    keep_mask[prunable_indices] = False
    keep_mask[kept_prunable_indices] = True
    return keep_mask


class Qwen2_5OmniVisionAttention(TransformersQwen2_5OmniVisionAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.full(
            [1, seq_length, seq_length], torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)

        attn_logits = attn_weights.sum(dim=(0, 1)).float() / (q.shape[0] * seq_length) if return_logits else None
        return attn_output, attn_logits


class Qwen2_5OmniVisionFlashAttention2(TransformersQwen2_5OmniVisionFlashAttention2):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = self._apply_rotary_pos_emb_flashatt(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self._apply_rotary_pos_emb_flashatt(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        attn_logits = compute_segmented_token_importance(q, k, cu_seqlens) if return_logits else None
        return attn_output, attn_logits


class Qwen2_5OmniVisionSdpaAttention(TransformersQwen2_5OmniVisionSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_length = hidden_states.shape[0]
        q = self.q(hidden_states).reshape(seq_length, self.num_heads, -1)
        k = self.k(hidden_states).reshape(seq_length, self.num_heads, -1)
        v = self.v(hidden_states).reshape(seq_length, self.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        attn_logits = (
            compute_segmented_token_importance(q.transpose(0, 1), k.transpose(0, 1), cu_seqlens)
            if return_logits
            else None
        )
        return attn_output, attn_logits


QWEN2_5_OMNI_VISION_ATTENTION_CLASSES = {
    "eager": Qwen2_5OmniVisionAttention,
    "flash_attention_2": Qwen2_5OmniVisionFlashAttention2,
    "sdpa": Qwen2_5OmniVisionSdpaAttention,
}


class Qwen2_5OmniVisionBlock(TransformersQwen2_5OmniVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = QWEN2_5_OMNI_VISION_ATTENTION_CLASSES[config._attn_implementation](
            config.hidden_size, num_heads=config.num_heads
        )
        self.mlp = Qwen2_5OmniMLP(config, bias=True)

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb, return_logits: bool = False):
        attn_output, attn_logits = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            return_logits=return_logits,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, attn_logits


class Qwen2_5OmniVisionEncoder(TransformersQwen2_5OmniVisionEncoder):
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.blocks = nn.ModuleList([Qwen2_5OmniVisionBlock(config) for _ in range(config.depth)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        return_attn_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        attn_logits = None
        for layer_num, blk in enumerate(self.blocks):
            cu_seqlens_now = cu_seqlens if layer_num in self.fullatt_block_indexes else cu_window_seqlens
            need_logits = return_attn_logits and layer_num == len(self.blocks) - 1
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_logits = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, rotary_pos_emb, need_logits
                )
            else:
                hidden_states, attn_logits = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    rotary_pos_emb=rotary_pos_emb,
                    return_logits=need_logits,
                )
                if not need_logits:
                    attn_logits = None

        if return_attn_logits and attn_logits is not None:
            attn_logits = attn_logits.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit).mean(dim=1)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        if return_attn_logits and attn_logits is not None:
            attn_logits = attn_logits[reverse_indices]
            return hidden_states, attn_logits

        return hidden_states


class Qwen2_5OmniFastVFlashAttention2(TransformersQwen2_5OmniFlashAttention2):
    def _last_query_attention_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        key_length = key_states.shape[-2]
        attn_scores = torch.matmul(query_states[:, :, -1:, :], key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                last_query_mask = attention_mask[:, :, -1:, :key_length]
                if last_query_mask.dtype == torch.bool:
                    attn_scores = attn_scores.masked_fill(~last_query_mask, torch.finfo(attn_scores.dtype).min)
                else:
                    attn_scores = attn_scores + last_query_mask
            elif attention_mask.dim() == 2:
                last_query_mask = attention_mask[:, None, None, :key_length]
                if last_query_mask.dtype == torch.bool:
                    keep_mask = last_query_mask
                else:
                    keep_mask = last_query_mask > 0
                attn_scores = attn_scores.masked_fill(~keep_mask, torch.finfo(attn_scores.dtype).min)

        if query_states.dtype == torch.float16:
            attn_scores = torch.where(torch.isinf(attn_scores), torch.zeros_like(attn_scores), attn_scores)

        attn_scores = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        return attn_scores.mean(dim=1).squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                "The input hidden states seems to be silently casted in float32; casting back to "
                f"{target_dtype} for flash attention."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_weights = None
        if output_attentions:
            attn_weights = self._last_query_attention_scores(query_states, key_states, attention_mask)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Qwen2_5OmniThinkerTextModel(TransformersQwen2_5OmniThinkerTextModel):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "_attn_implementation", None) == "flash_attention_2":
            for layer_idx, decoder_layer in enumerate(self.layers):
                decoder_layer.self_attn = Qwen2_5OmniFastVFlashAttention2(config, layer_idx)
        self.fastv_prompt_keep_mask = None
        self.fastv_prompt_length = None
        self._last_fastv_keep_mask = None

    def _fastv_layer_index(self, fastv_config) -> int:
        layer_idx = _config_get(fastv_config, "fastv_k", None)
        layer_idx = _config_get(fastv_config, "fastv_layer", layer_idx)
        if layer_idx is None:
            layer_idx = 2
        layer_idx = int(layer_idx)
        if layer_idx <= 0:
            logger.warning_once("FastV requires fastv_k > 0; use fastv_k=1 instead.")
            layer_idx = 1
        return min(layer_idx, len(self.layers) - 1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        fastv_token_mask: Optional[torch.Tensor] = None,
        fastv_config=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        is_prefill = inputs_embeds.shape[1] > 1
        if is_prefill:
            self.fastv_prompt_keep_mask = None
            self.fastv_prompt_length = None
            self._last_fastv_keep_mask = None

        fastv_enabled = (
            not self.training
            and _use_fastv(fastv_config)
            and fastv_token_mask is not None
            and fastv_token_mask.shape[:2] == inputs_embeds.shape[:2]
            and is_prefill
        )

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        fastv_decode_causal_mask = None
        fastv_decode_cache_position = None
        if not is_prefill and cache_position is not None and self.fastv_prompt_keep_mask is not None:
            prompt_length = self.fastv_prompt_keep_mask.numel()
            kept_prompt_length = int(self.fastv_prompt_keep_mask.sum().item())
            fastv_decode_cache_position = cache_position - prompt_length + kept_prompt_length
        fastv_decode_attention_mask = _build_fastv_decode_attention_mask(attention_mask, self.fastv_prompt_keep_mask)
        if fastv_decode_attention_mask is not None or fastv_decode_cache_position is not None:
            fastv_decode_causal_mask = self._update_causal_mask(
                fastv_decode_attention_mask,
                inputs_embeds,
                fastv_decode_cache_position if fastv_decode_cache_position is not None else cache_position,
                past_key_values,
                output_attentions,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        fastv_layer_idx = self._fastv_layer_index(fastv_config)
        fastv_pruned = False
        fastv_scores = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_attention_mask = causal_mask
            layer_cache_position = cache_position
            if (
                (fastv_decode_causal_mask is not None or fastv_decode_cache_position is not None)
                and self.fastv_prompt_keep_mask is not None
                and layer_idx >= fastv_layer_idx
            ):
                if fastv_decode_causal_mask is not None:
                    layer_attention_mask = fastv_decode_causal_mask
                if fastv_decode_cache_position is not None:
                    layer_cache_position = fastv_decode_cache_position

            if fastv_enabled and not fastv_pruned and fastv_scores is not None and layer_idx == fastv_layer_idx:
                keep_mask = _select_fastv_keep_mask(fastv_scores, fastv_token_mask, fastv_config)
                if keep_mask is not None:
                    total_media_tokens = int(fastv_token_mask[0].sum().item())
                    kept_media_tokens = int((fastv_token_mask[0] & keep_mask).sum().item())
                    hidden_states = hidden_states[:, keep_mask, :]
                    attention_mask = _slice_sequence_tensor(attention_mask, keep_mask)
                    position_ids = _slice_sequence_tensor(position_ids, keep_mask)
                    fastv_token_mask = _slice_sequence_tensor(fastv_token_mask, keep_mask)
                    if cache_position is not None and cache_position.shape[0] == keep_mask.shape[0]:
                        cache_position = torch.arange(
                            hidden_states.shape[1], device=cache_position.device, dtype=cache_position.dtype
                        )
                    position_embeddings = _slice_position_embeddings(position_embeddings, keep_mask)
                    causal_mask = self._update_causal_mask(
                        attention_mask,
                        hidden_states,
                        cache_position,
                        past_key_values,
                        output_attentions,
                    )
                    layer_attention_mask = causal_mask
                    layer_cache_position = cache_position
                    self.fastv_prompt_keep_mask = keep_mask.detach()
                    self.fastv_prompt_length = keep_mask.numel()
                    self._last_fastv_keep_mask = keep_mask.detach()
                    fastv_pruned = True
                    logger.info(
                        "FastV keeps %s/%s image/video/audio tokens before decoder layer %s.",
                        kept_media_tokens,
                        total_media_tokens,
                        layer_idx,
                    )

            need_fastv_attention = fastv_enabled and not fastv_pruned and layer_idx == fastv_layer_idx - 1
            layer_output_attentions = output_attentions or need_fastv_attention

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_attention_mask,
                    position_ids,
                    past_key_values,
                    layer_output_attentions,
                    use_cache,
                    layer_cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=layer_output_attentions,
                    use_cache=use_cache,
                    cache_position=layer_cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if layer_output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if need_fastv_attention:
                fastv_scores = _fastv_scores_from_attention(layer_outputs[1])
                if fastv_scores is None:
                    logger.warning_once("FastV could not get attention scores; skip FastV pruning.")

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2_5OmniThinkerForConditionalGeneration(TransformersQwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5OmniVisionEncoder._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.model = Qwen2_5OmniThinkerTextModel._from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.omni_llm_config = None
        self.fastv_config = None

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
        return_attn_logits: bool = False,
    ):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        return self.visual(
            pixel_values_videos,
            grid_thw=video_grid_thw,
            return_attn_logits=return_attn_logits,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_ids is not None and input_ids.shape[1] != 1:
            if input_features is not None:
                audio_features = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                )
                audio_mask = (
                    (input_ids == self.config.audio_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(
                    pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        fastv_config = self.fastv_config if self.fastv_config is not None else self.omni_llm_config

        fastv_token_mask = None
        if input_ids is not None and input_ids.shape[1] != 1:
            fastv_token_mask = (
                (input_ids == self.config.image_token_id)
                | (input_ids == self.config.video_token_id)
                | (input_ids == self.config.audio_token_id)
            )
            if attention_mask is not None:
                fastv_token_mask = fastv_token_mask & attention_mask.to(device=fastv_token_mask.device, dtype=torch.bool)
            if not fastv_token_mask.any():
                fastv_token_mask = None

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            fastv_token_mask=fastv_token_mask,
            fastv_config=fastv_config,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            fastv_keep_mask = getattr(self.model, "_last_fastv_keep_mask", None)
            if fastv_keep_mask is not None and labels.shape[-1] == fastv_keep_mask.numel():
                labels = labels[:, fastv_keep_mask.to(labels.device)]
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen2_5OmniForConditionalGeneration(TransformersQwen2_5OmniForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)


__all__ = [
    "Qwen2_5OmniForConditionalGeneration",
    "Qwen2_5OmniThinkerForConditionalGeneration",
    "Qwen2_5OmniVisionEncoder",
]
