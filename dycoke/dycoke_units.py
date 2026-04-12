import torch
from typing import Dict, List, Tuple, Optional

def prune_dycoke(
    feature: torch.Tensor,
    prune_ratio: float = 0.5,
    num_tokens_per_chunk: int = 288,
) -> torch.Tensor:
    def to_seq(x):
        if x is None:
            return None
        return x.mean(0) if x.dim() == 3 else x
    device = feature.device
    a = to_seq(feature).to(device)
    N = a.size(0)
    num_frames = a.shape[0] // num_tokens_per_chunk

    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    keep_mask = torch.ones(N, dtype=torch.bool, device=device)

    for t in range(num_frames):
        start_idx = t * num_tokens_per_chunk
        end_idx = (t + 1) * num_tokens_per_chunk
        tokens = a[start_idx:end_idx]
        keep_mask[start_idx:end_idx] = False
        group_size = tokens.shape[0] // 4
        main_tokens = group_size * 4
        keep_mask[start_idx+main_tokens:end_idx] = True
        a_main = tokens[:main_tokens]
        # a_norm = a_main / (a_main.norm(dim=-1, keepdim=True) + 1e-6)
        group1 = a_main[:group_size]
        group2 = a_main[group_size : 2 * group_size]
        group3 = a_main[2 * group_size : 3 * group_size]
        group4 = a_main[3 * group_size : 4 * group_size]

        keep_num = int(max(0, min(group_size, round((1.0 - prune_ratio*4/3) * group_size))))

        def keep_low_similarity_tokens(group: torch.Tensor, ref: torch.Tensor, start_idx: int) -> None:
            if keep_num <= 0:
                return
            if keep_num >= group_size:
                keep_mask[start_idx : start_idx + group_size] = True
                return
            sims = torch.nn.functional.cosine_similarity(group, ref, dim=1)
            _, keep_idx = torch.topk(sims, keep_num, largest=False)
            keep_mask[start_idx + keep_idx] = True

        keep_low_similarity_tokens(group2, group1, start_idx+group_size)
        keep_low_similarity_tokens(group4, group3, start_idx+3 * group_size)
        keep_low_similarity_tokens(group3, group1, start_idx+2 * group_size)
        keep_mask[start_idx:start_idx+group_size] = True
    return keep_mask



def omnizip_istm(video_feature, num_tokens_per_frame=196, merging_ratio=[0.7, 0.7]):

    num_frames = video_feature.shape[0] // num_tokens_per_frame
    # assert num_frames == 4
    # assert isinstance(merging_ratio, (list, tuple)) and len(merging_ratio) == 2

    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)

    def dpcknn(tokens, keep_rate=0.5, k=5):
        N = tokens.shape[0]
        num_keep = int(N * keep_rate)
        if num_keep >= N:
            return torch.arange(N, device=tokens.device)
        with torch.no_grad():
            normed = torch.nn.functional.normalize(tokens, dim=1)
            sim = torch.mm(normed, normed.T)
            sim.fill_diagonal_(-float('inf'))
            knn_vals, _ = torch.topk(sim, k, dim=1)
            knn_dist = knn_vals.mean(dim=1)
            selected = torch.topk(-knn_dist, num_keep, largest=True).indices
        return selected

    for t in range(num_frames):
        ratio_id = 0 if t < 2 else 1
        keep_ratio = 1.0 - merging_ratio[ratio_id]

        start_idx = t * num_tokens_per_frame
        end_idx = (t + 1) * num_tokens_per_frame
        tokens = video_feature[start_idx:end_idx]

        if t % 2 == 0:
            spatial_keep_rate = keep_ratio
            num_keep = int(num_tokens_per_frame * spatial_keep_rate)
            if num_keep < num_tokens_per_frame:
                keep_idx = dpcknn(tokens, keep_rate=spatial_keep_rate, k=5)
            else:
                keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx:end_idx][keep_idx] = True
        else:
            prev_tokens = video_feature[(t - 1) * num_tokens_per_frame : t * num_tokens_per_frame]
            prev_norm = torch.nn.functional.normalize(prev_tokens, p=2, dim=1)
            curr_norm = torch.nn.functional.normalize(tokens, p=2, dim=1)
            similarity = torch.nn.functional.cosine_similarity(curr_norm, prev_norm, dim=1)
            num_keep = int(num_tokens_per_frame * keep_ratio)
            if num_keep < num_tokens_per_frame:
                keep_idx = similarity.topk(num_keep, largest=False).indices
            else:
                keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx:end_idx][keep_idx] = True

    return mask


def dycoke(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    num_input_frames: int,
    audio_prune_ratio: float = 0.5,
    video_prune_ratio: float = 0.5,
):
    device = input_embeds.device
    is_batched = input_embeds.dim() == 3
    if is_batched:
        B, L, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, D)
        flat_ids = input_ids.reshape(-1)
    else:
        L, D = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids



    video_token_mask = (flat_ids == video_token_id).to(device)
    audio_token_mask = (flat_ids == audio_token_id).to(device)

    video_indices = torch.nonzero(video_token_mask, as_tuple=True)[0]
    audio_indices = torch.nonzero(audio_token_mask, as_tuple=True)[0]

    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]

    video_token_per_frame = video_feature.shape[0] // num_input_frames

    '''
    print(f"audio_feature : {audio_feature.shape}")
    print(f"video_feature : {video_feature.shape}")
    print(f"num_input_frames : {num_input_frames}")
    print(f"video_token_per_frame : {video_token_per_frame}")
    '''

    audio_mask = prune_dycoke(
        feature=audio_feature,
        prune_ratio=audio_prune_ratio,
        num_tokens_per_chunk=100,
    )
    video_mask = prune_dycoke(
        feature=video_feature,
        prune_ratio=video_prune_ratio,
        num_tokens_per_chunk=576,
    )

    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    assert audio_mask.shape[0] == audio_indices.shape[0]
    assert video_mask.shape[0] == video_indices.shape[0]
    global_mask[video_indices] = video_mask
    global_mask[audio_indices] = audio_mask

    if is_batched:
        input_embeds_out = flat_embeds.reshape(B, L, D)
    else:
        input_embeds_out = flat_embeds
    #print(f"audio_feature")
    return input_embeds_out, global_mask


    '''
    if num_input_frames % 4 == 0:
        group_count = num_input_frames // 4
        num_video_tokens_per_group = max(1, video_feature.shape[0] // group_count)
        num_audio_tokens_per_group = max(1, audio_feature.shape[0] // group_count)
        
        print(f"num_video_tokens_per_group : {num_video_tokens_per_group}")
        print(f"num_audio_tokens_per_group : {num_audio_tokens_per_group}")


        video_merging_ratios = [video_prune_ratio] * group_count

        video_group_masks = []
        for i in range(0, group_count, 2):
            if i + 2 <= group_count:  
                video_merging_ratio = video_merging_ratios[i:i+2]
                v_start = i * num_video_tokens_per_group
                v_end = (i + 2) * num_video_tokens_per_group if i < group_count - 1 else video_feature.shape[0]
                group_feat = video_feature[v_start:v_end]
                group_len = group_feat.size(0)
                if group_len % 4 == 0:
                    group_mask = omnizip_istm(
                        group_feat, num_tokens_per_frame=video_token_per_frame * 2, merging_ratio=video_merging_ratio
                    )
                else:
                    group_mask = torch.ones(group_len, dtype=torch.bool, device=group_feat.device)

                video_group_masks.append(group_mask)
            else:
                video_merging_ratio = video_merging_ratios[i:i+2]
                v_start = i * num_video_tokens_per_group
                v_end = (i + 2) * num_video_tokens_per_group if i < group_count - 1 else video_feature.shape[0]
                group_feat = video_feature[v_start:v_end]
                group_len = group_feat.size(0)
                group_mask = torch.ones(group_len, dtype=torch.bool, device=group_feat.device)

                video_group_masks.append(group_mask)

        video_mask = torch.cat(video_group_masks, dim=0)
        global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)

        assert video_mask.shape[0] == video_indices.shape[0]
        assert audio_mask.shape[0] == audio_indices.shape[0]

        global_mask[video_indices] = video_mask
        global_mask[audio_indices] = audio_mask

        if is_batched:
            input_embeds_out = flat_embeds.reshape(B, L, D)
        else:
            input_embeds_out = flat_embeds

        return input_embeds_out, global_mask
    

    num_video_tokens = video_feature.shape[0]
    num_audio_tokens = audio_feature.shape[0]

    VIDEO_GROUP_SIZE = video_token_per_frame * 4
    AUDIO_GROUP_SIZE = 50

    video_groups = []
    audio_groups = []
    v_ptr = a_ptr = 0

    while v_ptr + VIDEO_GROUP_SIZE <= num_video_tokens and a_ptr + AUDIO_GROUP_SIZE <= num_audio_tokens:
        video_groups.append((v_ptr, v_ptr + VIDEO_GROUP_SIZE))
        audio_groups.append((a_ptr, a_ptr + AUDIO_GROUP_SIZE))
        v_ptr += VIDEO_GROUP_SIZE
        a_ptr += AUDIO_GROUP_SIZE

    if v_ptr < num_video_tokens:
        if a_ptr < num_audio_tokens:
            video_groups.append((v_ptr, num_video_tokens))
            audio_groups.append((a_ptr, num_audio_tokens))
        else:
            video_groups.append((v_ptr, num_video_tokens))
            audio_groups.append((a_ptr, a_ptr))
    elif a_ptr < num_audio_tokens:
        video_groups.append((v_ptr, v_ptr))
        audio_groups.append((a_ptr, num_audio_tokens))

    assert len(video_groups) == len(audio_groups)
    group_num = len(video_groups)


    video_merging_ratios = [video_prune_ratio] * group_num

    video_group_masks = []
    group_count = len(video_groups) // 2 
    idx = 0
    while idx < len(video_groups):
        if idx + 1 < len(video_groups):
            v_start_0, v_end_0 = video_groups[idx]
            v_start_1, v_end_1 = video_groups[idx + 1]
            group_feat = video_feature[v_start_0:v_end_1]
            video_merging_ratio = [video_merging_ratios[idx], video_merging_ratios[idx + 1]]
        else:
            v_start_0, v_end_0 = video_groups[idx]
            group_feat = video_feature[v_start_0:v_end_0]
            video_merging_ratio = [video_merging_ratios[idx]]
        
        group_len = group_feat.size(0)
        is_tail_video_group = (group_len != 2 * VIDEO_GROUP_SIZE) if (idx+1 < len(video_groups)) else (group_len != VIDEO_GROUP_SIZE)
        if group_len == 0:
            group_mask = torch.zeros(0, dtype=torch.bool, device=video_feature.device)
        elif is_tail_video_group:
            group_mask = torch.ones(group_len, dtype=torch.bool, device=video_feature.device)
        else:
            group_mask = omnizip_istm(
                group_feat, num_tokens_per_frame=video_token_per_frame * 2, merging_ratio=video_merging_ratio
            )
        video_group_masks.append(group_mask)
        idx += 2

    video_mask = torch.cat(video_group_masks, dim=0)
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)

    assert video_mask.shape[0] == video_indices.shape[0]
    assert audio_mask.shape[0] == audio_indices.shape[0]

    global_mask[video_indices] = video_mask
    global_mask[audio_indices] = audio_mask

    if is_batched:
        input_embeds_out = flat_embeds.reshape(B, L, D)
    else:
        input_embeds_out = flat_embeds

    return input_embeds_out, global_mask
    '''




