import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.utils import logging
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2RMSNorm,
    Qwen2_5OmniForConditionalGeneration as TransformersQwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniMLP,
    Qwen2_5OmniThinkerCausalLMOutputWithPast,
    Qwen2_5OmniThinkerForConditionalGeneration as TransformersQwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniVisionAttention as TransformersQwen2_5OmniVisionAttention,
    Qwen2_5OmniVisionBlock as TransformersQwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionEncoder as TransformersQwen2_5OmniVisionEncoder,
    Qwen2_5OmniVisionFlashAttention2 as TransformersQwen2_5OmniVisionFlashAttention2,
    Qwen2_5OmniVisionSdpaAttention as TransformersQwen2_5OmniVisionSdpaAttention,
    apply_rotary_pos_emb_vision,
    flash_attn_varlen_func,
)

from .pruning_utils import (
    compute_segmented_token_importance,
    prune_audio_tokens_in_inputs_embeds,
    prune_video_tokens_in_inputs_embeds,
)


logger = logging.get_logger(__name__)


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


class Qwen2_5OmniThinkerForConditionalGeneration(TransformersQwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5OmniVisionEncoder._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        )
        self.omni_llm_config = None

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

        video_attn_logits = None
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
                video_embeds, video_attn_logits = self.get_video_features(
                    pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                    return_attn_logits=True,
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

        if input_ids is not None and input_ids.shape[1] != 1:
            if self.omni_llm_config is None:
                omni_llm_config = {
                    "audio_keep_rate": 0.5,
                    "video_keep_rate": 0.4,
                }
            else:
                omni_llm_config = self.omni_llm_config

            if input_features is not None:
                audio_chunk_lengths = None
                if audio_feature_lengths is not None:
                    _, audio_chunk_lengths = self.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths)
                input_ids, inputs_embeds, attention_mask, position_ids, labels = prune_audio_tokens_in_inputs_embeds(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    audio_token_index=self.config.audio_token_id,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels,
                    audio_chunk_lengths=audio_chunk_lengths,
                    total_keep_rate=omni_llm_config["audio_keep_rate"],
                )

            if pixel_values_videos is not None:
                input_ids, inputs_embeds, attention_mask, position_ids, labels = prune_video_tokens_in_inputs_embeds(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    video_token_index=self.config.video_token_id,
                    video_grid_thw=video_grid_thw,
                    video_attn_logits=video_attn_logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels,
                    keep_rate=omni_llm_config["video_keep_rate"],
                    frames_per_chunk=4,
                )

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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
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
