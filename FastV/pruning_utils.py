import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def prune_audio_tokens_by_similarity(
    audio_features: torch.Tensor,
    chunk_lengths: Optional[torch.Tensor] = None,
    total_keep_rate: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_audio_tokens = audio_features.shape[0]
    keep_mask = torch.ones(num_audio_tokens, dtype=torch.bool, device=audio_features.device)

    if num_audio_tokens <= 1:
        return audio_features, keep_mask

    if chunk_lengths is None:
        chunk_lengths = torch.tensor([num_audio_tokens], device=audio_features.device, dtype=torch.long)
    else:
        chunk_lengths = chunk_lengths.to(device=audio_features.device, dtype=torch.long)

    valid_chunk_lengths = []
    start = 0
    for chunk_length in chunk_lengths.tolist():
        if chunk_length <= 0:
            continue
        end = min(start + chunk_length, num_audio_tokens)
        if start >= end:
            break
        valid_chunk_lengths.append(end - start)
        start = end

    if start < num_audio_tokens:
        valid_chunk_lengths.append(num_audio_tokens - start)

    if not valid_chunk_lengths:
        return audio_features, keep_mask

    normalized_audio = F.normalize(audio_features.float(), p=2, dim=-1)
    keep_mask[:] = False

    chunk_variances = []
    start = 0
    for chunk_length in valid_chunk_lengths:
        end = start + chunk_length
        chunk = audio_features[start:end].float()
        if chunk_length <= 1:
            chunk_variances.append(torch.tensor(0.0, device=audio_features.device))
        else:
            chunk_variances.append(chunk.var(dim=0, unbiased=False).mean())
        start = end

    variance_tensor = torch.stack(chunk_variances)
    chunk_capacity = torch.tensor(
        [max(0, chunk_length - 1) for chunk_length in valid_chunk_lengths],
        device=audio_features.device,
        dtype=torch.long,
    )

    target_total_keep = int(round(num_audio_tokens * total_keep_rate))
    target_total_keep = max(len(valid_chunk_lengths), min(num_audio_tokens, target_total_keep))
    extra_keep_budget = target_total_keep - len(valid_chunk_lengths)

    extra_keep = torch.zeros(len(valid_chunk_lengths), device=audio_features.device, dtype=torch.long)
    if extra_keep_budget > 0 and chunk_capacity.sum().item() > 0:
        if variance_tensor.sum().item() > 0:
            weight = variance_tensor / variance_tensor.sum()
        else:
            weight = chunk_capacity.float() / chunk_capacity.sum()

        raw_extra = weight * extra_keep_budget
        extra_keep = torch.floor(raw_extra).to(torch.long)
        extra_keep = torch.minimum(extra_keep, chunk_capacity)

        remaining_budget = extra_keep_budget - extra_keep.sum().item()
        fractional = raw_extra - torch.floor(raw_extra)
        while remaining_budget > 0:
            available = chunk_capacity - extra_keep
            if torch.all(available <= 0):
                break
            score = fractional.masked_fill(available <= 0, -1.0)
            selected_chunk = int(torch.argmax(score).item())
            extra_keep[selected_chunk] += 1
            fractional[selected_chunk] = -1.0
            remaining_budget -= 1

    chunk_keep_counts = 1 + extra_keep

    start = 0
    for chunk_idx, chunk_length in enumerate(valid_chunk_lengths):
        end = start + chunk_length
        keep_mask[start] = True

        num_extra_keep = int(chunk_keep_counts[chunk_idx].item()) - 1
        if num_extra_keep > 0 and chunk_length > 1:
            chunk_audio = normalized_audio[start:end]
            neighbor_similarity = (chunk_audio[1:] * chunk_audio[:-1]).sum(dim=-1)
            num_extra_keep = min(num_extra_keep, chunk_length - 1)
            selected = torch.topk(neighbor_similarity, k=num_extra_keep, largest=False).indices + 1
            keep_mask[start + selected] = True

        start = end

    return audio_features[keep_mask], keep_mask


def prune_audio_tokens_in_inputs_embeds(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    audio_token_index: int,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    audio_chunk_lengths: Optional[torch.Tensor] = None,
    total_keep_rate: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if input_ids.shape[0] != 1:
        raise NotImplementedError("Audio cosine pruning in inputs_embeds currently only supports batch_size=1.")

    audio_positions = torch.nonzero(input_ids[0] == audio_token_index, as_tuple=True)[0]
    if audio_positions.numel() == 0:
        return input_ids, inputs_embeds, attention_mask, position_ids, labels

    audio_features = inputs_embeds[0, audio_positions, :]
    _, audio_keep_mask = prune_audio_tokens_by_similarity(
        audio_features,
        chunk_lengths=audio_chunk_lengths,
        total_keep_rate=total_keep_rate,
    )
    logger.info(
        "Audio pruning keeps %s/%s tokens.",
        int(audio_keep_mask.sum().item()),
        audio_positions.numel(),
    )

    global_keep_mask = torch.ones_like(input_ids[0], dtype=torch.bool)
    global_keep_mask[audio_positions] = audio_keep_mask

    input_ids = input_ids[:, global_keep_mask]
    inputs_embeds = inputs_embeds[:, global_keep_mask, :]
    if attention_mask is not None:
        attention_mask = attention_mask[:, global_keep_mask]
    if position_ids is not None:
        if position_ids.dim() == 3:
            position_ids = position_ids[:, :, global_keep_mask]
        else:
            position_ids = position_ids[:, global_keep_mask]
    if labels is not None:
        labels = labels[:, global_keep_mask]

    return input_ids, inputs_embeds, attention_mask, position_ids, labels


def select_max_min_tokens(
    frame_tokens: torch.Tensor,
    seed_idx: int,
    keep_count: int,
) -> torch.Tensor:
    num_tokens = frame_tokens.shape[0]
    if num_tokens == 0 or keep_count <= 0:
        return torch.zeros(0, dtype=torch.long, device=frame_tokens.device)

    keep_count = min(keep_count, num_tokens)
    if keep_count == 1:
        return torch.tensor([seed_idx], dtype=torch.long, device=frame_tokens.device)

    norm_tokens = F.normalize(frame_tokens.float(), p=2, dim=-1)
    selected = [int(seed_idx)]
    selected_mask = torch.zeros(num_tokens, dtype=torch.bool, device=frame_tokens.device)
    selected_mask[seed_idx] = True

    while len(selected) < keep_count:
        selected_tokens = norm_tokens[selected]
        similarity = norm_tokens @ selected_tokens.transpose(0, 1)
        min_distance = (1.0 - similarity).min(dim=1).values
        min_distance[selected_mask] = -1.0
        next_idx = int(torch.argmax(min_distance).item())
        selected.append(next_idx)
        selected_mask[next_idx] = True

    return torch.tensor(selected, dtype=torch.long, device=frame_tokens.device)


def prune_video_tokens_in_inputs_embeds(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    video_token_index: int,
    video_grid_thw: Optional[torch.Tensor],
    video_attn_logits: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    keep_rate: float = 0.4,
    frames_per_chunk: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (
        video_grid_thw is None
        or video_attn_logits is None
        or input_ids.shape[0] != 1
        or video_grid_thw.shape[0] != 1
    ):
        return input_ids, inputs_embeds, attention_mask, position_ids, labels

    video_positions = torch.nonzero(input_ids[0] == video_token_index, as_tuple=True)[0]
    if video_positions.numel() == 0:
        return input_ids, inputs_embeds, attention_mask, position_ids, labels
    if video_attn_logits.numel() != video_positions.numel():
        logger.warning(
            "Skip video pruning because attention logits length %s does not match video token length %s.",
            video_attn_logits.numel(),
            video_positions.numel(),
        )
        return input_ids, inputs_embeds, attention_mask, position_ids, labels

    temporal_slices = int(video_grid_thw[0][0].item())
    if temporal_slices <= 0 or video_positions.numel() % temporal_slices != 0:
        return input_ids, inputs_embeds, attention_mask, position_ids, labels

    video_tokens = inputs_embeds[0, video_positions, :]
    tokens_per_frame = video_positions.numel() // temporal_slices
    if tokens_per_frame == 0:
        return input_ids, inputs_embeds, attention_mask, position_ids, labels

    frame_keep_count = max(1, int(round(tokens_per_frame * keep_rate)))
    local_keep_mask = torch.zeros(video_positions.numel(), dtype=torch.bool, device=inputs_embeds.device)
    full_chunk_frame_count = (temporal_slices // frames_per_chunk) * frames_per_chunk

    for chunk_start in range(0, full_chunk_frame_count, frames_per_chunk):
        chunk_end = chunk_start + frames_per_chunk
        for frame_idx in range(chunk_start, chunk_end):
            start = frame_idx * tokens_per_frame
            end = start + tokens_per_frame
            frame_tokens = video_tokens[start:end]
            frame_scores = video_attn_logits[start:end]
            seed_idx = int(torch.argmax(frame_scores).item())

            if (frame_idx - chunk_start) in {0, 2} or frame_idx == chunk_start:
                selected = select_max_min_tokens(frame_tokens, seed_idx, frame_keep_count)
            else:
                ref_frame_idx = frame_idx - 1
                ref_start = ref_frame_idx * tokens_per_frame
                ref_end = ref_start + tokens_per_frame
                ref_tokens = video_tokens[ref_start:ref_end]
                current_norm = F.normalize(frame_tokens.float(), p=2, dim=-1)
                ref_norm = F.normalize(ref_tokens.float(), p=2, dim=-1)
                similarity = (current_norm * ref_norm).sum(dim=-1)

                chosen = [seed_idx]
                candidate_order = torch.argsort(similarity, descending=False)
                for idx in candidate_order.tolist():
                    if idx not in chosen:
                        chosen.append(idx)
                    if len(chosen) >= frame_keep_count:
                        break
                selected = torch.tensor(chosen[:frame_keep_count], dtype=torch.long, device=inputs_embeds.device)

            local_keep_mask[start + selected] = True

    if full_chunk_frame_count < temporal_slices:
        for frame_idx in range(full_chunk_frame_count, temporal_slices):
            start = frame_idx * tokens_per_frame
            end = start + tokens_per_frame
            frame_tokens = video_tokens[start:end]
            frame_scores = video_attn_logits[start:end]
            seed_idx = int(torch.argmax(frame_scores).item())
            selected = select_max_min_tokens(frame_tokens, seed_idx, frame_keep_count)
            local_keep_mask[start + selected] = True

        logger.info(
            "Video pruning applies coverage selection to all %s tail frames because frame count %s is not divisible by %s.",
            temporal_slices - full_chunk_frame_count,
            temporal_slices,
            frames_per_chunk,
        )

    global_keep_mask = torch.ones_like(input_ids[0], dtype=torch.bool)
    global_keep_mask[video_positions] = local_keep_mask
    kept_video_tokens = int(local_keep_mask.sum().item())
    logger.info(
        "Video pruning keeps %s/%s tokens across %s frames (%s per frame).",
        kept_video_tokens,
        video_positions.numel(),
        temporal_slices,
        frame_keep_count,
    )

    input_ids = input_ids[:, global_keep_mask]
    inputs_embeds = inputs_embeds[:, global_keep_mask, :]
    if attention_mask is not None:
        attention_mask = attention_mask[:, global_keep_mask]
    if position_ids is not None:
        if position_ids.dim() == 3:
            position_ids = position_ids[:, :, global_keep_mask]
        else:
            position_ids = position_ids[:, global_keep_mask]
    if labels is not None:
        labels = labels[:, global_keep_mask]

    return input_ids, inputs_embeds, attention_mask, position_ids, labels


def compute_segmented_token_importance(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    q = query_states.transpose(0, 1).float()
    k = key_states.transpose(0, 1).float()
    token_importance = torch.zeros(query_states.shape[0], device=query_states.device, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5

    for i in range(1, len(cu_seqlens)):
        start = int(cu_seqlens[i - 1].item())
        end = int(cu_seqlens[i].item())
        seg_len = end - start
        if seg_len <= 0:
            continue

        q_seg = q[:, start:end, :]
        k_seg = k[:, start:end, :]
        seg_importance = torch.zeros(seg_len, device=query_states.device, dtype=torch.float32)
        chunk_size = 1024
        for offset in range(0, seg_len, chunk_size):
            q_chunk = q_seg[:, offset : offset + chunk_size, :]
            attn_chunk = torch.matmul(q_chunk, k_seg.transpose(-1, -2)) * scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            seg_importance += attn_chunk.sum(dim=(0, 1))
        token_importance[start:end] = seg_importance / (q.shape[0] * seg_len)

    return token_importance


__all__ = [
    "compute_segmented_token_importance",
    "prune_audio_tokens_by_similarity",
    "prune_audio_tokens_in_inputs_embeds",
    "prune_video_tokens_in_inputs_embeds",
    "select_max_min_tokens",
]
