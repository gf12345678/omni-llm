import json
import os
import logging
import tqdm
import sys
from moviepy import VideoFileClip
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
from typing import Dict, List
from transformers import Qwen2_5OmniProcessor
import argparse
import datetime
import subprocess
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from qwen_omni_utils import process_mm_info


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12347 eval/eval_shortvid.py --WAPPER-METHOD omnizip --OMNIZIP_RHO_AUDIO 0.3 --OMNIZIP_RHO_VIDEO 0.6 --OMNIZIP_G 3 --OMNIZIP_CONTEXTUAL_RATIO 0.05 --benchmarks ShortVid  2>./logs/warn.log

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=12359 eval/eval_shortvid.py --WAPPER-METHOD omni_llm  --benchmarks WorldSense  --OMNI_LLM_RHO_AUDIO 0.3  --OMNI_LLM_RHO_VIDEO 0.6  2>./logs/omni_llm_warn.log


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False


class ConsoleSummaryFilter(logging.Filter):
    def filter(self, record):
        message = record.getMessage()
        return message.startswith("[PROGRESS]") or message.startswith("[RESULT]")


class ResultOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.getMessage().startswith("[RESULT]")


def configure_logging(run_output_dir, rank):
    log_dir = os.path.join(run_output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger.handlers.clear()

    formatter = logging.Formatter(
        f"%(asctime)s - rank{rank} - %(levelname)s - %(message)s"
    )
    run_name = os.path.basename(run_output_dir)

    if rank == 0:
        total_file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{run_name}.log"),
            encoding="utf-8",
        )
        total_file_handler.setLevel(logging.INFO)
        total_file_handler.setFormatter(formatter)
        logger.addHandler(total_file_handler)

        accuracy_file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{run_name}_accuracy.log"),
            encoding="utf-8",
        )
        accuracy_file_handler.setLevel(logging.INFO)
        accuracy_file_handler.setFormatter(formatter)
        accuracy_file_handler.addFilter(ResultOnlyFilter())
        logger.addHandler(accuracy_file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.addFilter(ConsoleSummaryFilter())
        logger.addHandler(console_handler)


def log_progress(message):
    logger.info(f"[PROGRESS] {message}")


def log_result(message):
    logger.info(f"[RESULT] {message}")
def load_json_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed evaluation for WorldSense, AVUT and ShortVid."
    )
    available_wrapper_methods = ["base", "omnizip", "omni_llm", "dycoke"]
    available_model_types = ["qwenomni3b", "qwenomni7b"]
    parser.add_argument(
        "--WAPPER-METHOD",
        type=str,
        default="base",
        choices=available_wrapper_methods,
        help=(
            "Pruning/model wrapper method. "
            f"Available choices: {', '.join(available_wrapper_methods)}."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="qwenomni3b",
        choices=available_model_types,
        help=(
            "Base model size to load. "
            f"Available choices: {', '.join(available_model_types)}."
        ),
    )
    parser.add_argument(
        "--mini-test-num",
        "--mini-test-videos",
        dest="mini_test_videos",
        type=int,
        default=None,
        help="If set, each dataset only evaluates the first N distinct videos for quick validation.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all", #WorldSense,AVUT,ShortVid
        help="Comma-separated benchmark names to evaluate, e.g. 'WorldSense,AVUT'. Use 'all' to run every benchmark.",
    )
    parser.add_argument(
        "--RHO_AUDIO",
        "--OMNIZIP_RHO_AUDIO",
        "--OMNI_LLM_RHO_AUDIO",
        "--DYCOKE_RHO_AUDIO",
        dest="rho_audio",
        type=float,
        default=None,
        help="Shared audio pruning ratio. Interpreted by the selected wrapper method.",
    )
    parser.add_argument(
        "--RHO_VIDEO",
        "--OMNIZIP_RHO_VIDEO",
        "--OMNI_LLM_RHO_VIDEO",
        "--DYCOKE_RHO_VIDEO",
        dest="rho_video",
        type=float,
        default=None,
        help="Shared video pruning ratio. Interpreted by the selected wrapper method.",
    )
    parser.add_argument(
        "--OMNIZIP_G",
        type=int,
        default=3,
        help="G parameter",
    )
    parser.add_argument(
        "--OMNIZIP_CONTEXTUAL_RATIO",
        type=float,
        default=0.05,
        help="Contextual ratio",
    )
    return parser.parse_args()


def resolve_wrapper_rhos(args):
    if args.rho_audio is None:
        if args.WAPPER_METHOD == "omnizip":
            rho_audio = 0.3
        elif args.WAPPER_METHOD == "omni_llm":
            rho_audio = 0.5
        elif args.WAPPER_METHOD == "dycoke":
            rho_audio = 0.6
        else:
            rho_audio = 0.0
    else:
        rho_audio = args.rho_audio

    if args.rho_video is None:
        if args.WAPPER_METHOD in {"omnizip", "omni_llm", "dycoke"}:
            rho_video = 0.6
        else:
            rho_video = 0.0
    else:
        rho_video = args.rho_video

    return rho_audio, rho_video


def resolve_selected_benchmarks(benchmark_arg, available_benchmarks):
    if benchmark_arg.strip().lower() == "all":
        return list(available_benchmarks.keys())

    selected_benchmarks = []
    seen = set()
    for benchmark_name in benchmark_arg.split(","):
        normalized_name = benchmark_name.strip()
        if not normalized_name:
            continue
        if normalized_name not in available_benchmarks:
            raise ValueError(
                f"Unknown benchmark '{normalized_name}'. Available benchmarks: {', '.join(available_benchmarks.keys())}"
            )
        if normalized_name not in seen:
            selected_benchmarks.append(normalized_name)
            seen.add(normalized_name)

    if not selected_benchmarks:
        raise ValueError(
            f"No valid benchmarks were provided. Available benchmarks: {', '.join(available_benchmarks.keys())}"
        )

    return selected_benchmarks


def build_run_config_label(args):
    model_label = args.model_type
    rho_audio, rho_video = resolve_wrapper_rhos(args)
    if args.WAPPER_METHOD == "omnizip":
        prune_label = (
            f"oa{rho_audio:g}_"
            f"ov{rho_video:g}_"
            f"g{args.OMNIZIP_G}_"
            f"ctx{args.OMNIZIP_CONTEXTUAL_RATIO:g}"
        )
    elif args.WAPPER_METHOD == "omni_llm":
        prune_label = f"oa{rho_audio:g}_ov{rho_video:g}"
    elif args.WAPPER_METHOD == "dycoke":
        prune_label = f"oa{rho_audio:g}_ov{rho_video:g}_audio50f"
    else:
        prune_label = "no_prune"
    return f"{args.WAPPER_METHOD}_{model_label}_{prune_label}"


def limit_samples_by_video_count(benchmark_data, max_videos):
    if max_videos is None:
        return benchmark_data
    if max_videos <= 0:
        return []

    selected_videos = set()
    limited_samples = []
    for sample in benchmark_data:
        video_name = sample["video"]
        if video_name not in selected_videos:
            if len(selected_videos) >= max_videos:
                continue
            selected_videos.add(video_name)
        limited_samples.append(sample)
    return limited_samples


def split_samples_across_ranks(benchmark_data):
    total_samples = len(benchmark_data)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # 计算每个进程处理的数量
    per_chunk = (total_samples + world_size - 1) // world_size
    start_idx = rank * per_chunk
    end_idx = min(start_idx + per_chunk, total_samples)

    # 每个进程只认领自己的那一部分
    my_benchmark_data = benchmark_data[start_idx:end_idx]

    logger.info(f"进程 {rank}/{world_size} 启动: 负责样本 {start_idx} 到 {end_idx} (共 {len(my_benchmark_data)} 个)")
    return my_benchmark_data


'''    
def omni_zip(model):
    omnizip_config = {
        "rho_audio": OMNIZIP_RHO_AUDIO,
        "rho_video": OMNIZIP_RHO_VIDEO,
        "g": OMNIZIP_G,
        "contextual_ratio": OMNIZIP_CONTEXTUAL_RATIO,
    }

    if WAPPER_METHOD == 'omnizip':  
        model.thinker.omnizip_config = omnizip_config
    else:
        model.thinker.omnizip_config = None
'''


def setup():
    # 初始化进程组，后端用 nccl（GPU 通讯标准）
    dist.init_process_group(backend="nccl")
    
    # 获取当前进程在当前机器上的 GPU 编号
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 这一步极其重要：把当前进程绑定到对应的显卡上
    torch.cuda.set_device(local_rank)
    
    return local_rank


def load_model(model_path, cuda_id, args):
    if args.WAPPER_METHOD == "omnizip":
        from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    elif args.WAPPER_METHOD == "omni_llm":
        from omni_llm.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    elif args.WAPPER_METHOD == "dycoke":
        from dycoke.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    else:
        from transformers import Qwen2_5OmniForConditionalGeneration

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=cuda_id,
        attn_implementation="flash_attention_2",
    )

    rho_audio, rho_video = resolve_wrapper_rhos(args)

    omnizip_config = {
        "rho_audio": rho_audio,
        "rho_video": rho_video,
        "g": args.OMNIZIP_G,
        "contextual_ratio": args.OMNIZIP_CONTEXTUAL_RATIO,
    }
    omni_llm_config = {
        "audio_keep_rate": max(0.0, min(1.0, 1.0 - rho_audio)),
        "video_keep_rate": max(0.0, min(1.0, 1.0 - rho_video)),
    }
    dycoke_config = {
        "audio_prune_ratio": max(0.0, min(1.0, rho_audio)),
        "video_prune_ratio": max(0.0, min(1.0, rho_video)),
    }

    if hasattr(model, "thinker"):
        thinker = model.thinker
        thinker.omnizip_config = omnizip_config if args.WAPPER_METHOD == "omnizip" else None
        thinker.omni_llm_config = omni_llm_config if args.WAPPER_METHOD == "omni_llm" else None
        thinker.dycoke_config = dycoke_config if args.WAPPER_METHOD == "dycoke" else None

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor
    
def load_benchmark_WorldSense_data(data_set):
    benchmark_data = []
    for vid, v in data_set.items():
            for key in v:
                if key.startswith("task"):
                    task_obj = v[key]
                    sample = {
                        "video": v["video_id"] + ".mp4",
                        "video_id": v["video_id"],
                        "video_caption": v.get("video_caption", ""),
                        "problem_type": task_obj.get("task_type", key),
                        "data_type": task_obj.get("task_domain", ""),
                        "domain": v.get("domain", ""),
                        "question": task_obj["question"],
                        "answer": task_obj["answer"],
                        "candidates": task_obj["candidates"]
                    }
                    benchmark_data.append(sample)
    return benchmark_data


def load_benchmark_AVUT_data(data_set):
    benchmark_data = []
    for sample in data_set:
        candidates = []
        for opt in ["A", "B", "C", "D", "E", "F"]:
            option_text = sample.get(f"option_{opt}")
            if option_text:
                candidates.append(f"{opt}. {option_text}")

        benchmark_data.append({
            "video": os.path.basename(sample["video_path"]),
            "video_id": sample.get("video_id", ""),
            "qa_id": sample.get("QA_id", ""),
            "question": sample["question"],
            "candidates": candidates,
            "answer": sample["answer"],
            "task_type": sample.get("task_type", ""),
            "video_type": sample.get("video_type", ""),
        })
    return benchmark_data


def load_benchmark_ShortVid_data(data_set):
    benchmark_data = []
    for sample in data_set:
        benchmark_data.append({
            "video": sample["video"],
            "question": sample["question"],
            "candidates": sample["candidates"],
            "answer": sample["answer"],
            "problem_type": sample.get("problem_type", ""),
            "data_type": sample.get("data_type", ""),
        })
    return benchmark_data

def _check_if_video_has_audio(video_path):
    try:
        clip = VideoFileClip(video_path)
        return clip.audio is not None
    except Exception as e:
        logger.error(f"Failed to load video/audio {video_path}: {e}")
        return False


def _extract_predicted_option(resp_text, candidates):
    valid_options = []
    for candidate in candidates:
        stripped = candidate.strip()
        if stripped and stripped[0].upper() in ["A", "B", "C", "D", "E", "F"]:
            valid_options.append(stripped[0].upper())

    for opt in valid_options:
        if resp_text.upper().strip().startswith(opt):
            return opt

    if len(resp_text) > 0 and resp_text[0].upper() in valid_options:
        return resp_text[0].upper()
    return None


def _run_mcq_inference(video_path, question, candidates, processor, model):
    candidates_text = "\n".join(candidates)
    prompt = f"{question}\nOptions:\n{candidates_text}\nAnswer with the option's letter from the given choices directly."
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Please analyze the video carefully and select the most appropriate answer from the given options."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    try:

        USE_AUDIO_IN_VIDEO = _check_if_video_has_audio(video_path)
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        num_input_frames = videos[0].shape[0]
        model.thinker.nframes = num_input_frames
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)
        
        cont = model.generate(
                    **inputs,
                    return_audio=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=1,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=100,
                    use_cache=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    thinker_do_sample=False,
                )
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        resp_text = answers[0].strip() if answers and answers[0] else ""
        predicted_answer = _extract_predicted_option(resp_text, candidates)
        return predicted_answer, resp_text
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def evaluate_sample_WorldSense(sample, video_dir, processor, model):
    video_path = os.path.join(video_dir, sample["video"])
    predicted_answer, resp_text = _run_mcq_inference(video_path, sample["question"], sample["candidates"], processor, model)
    is_correct = (predicted_answer == sample["answer"])

    result = {
        "video": sample["video"],
        "video_id": sample.get("video_id", ""),
        "question": sample["question"],
        "candidates": sample["candidates"],
        "video_caption": sample.get("video_caption", ""),
        "correct_answer": sample["answer"],
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "problem_type": sample["problem_type"],
        "data_type": sample["data_type"],
        "domain": sample["domain"],
        "model_response": resp_text,
    }

    logger.info(f"WorldSense processed {sample['video']}: {sample['answer']} -> {predicted_answer} ({'✓' if is_correct else '✗'})")
    return result


def evaluate_sample_AVUT(sample, video_dir, processor, model):
    video_path = os.path.join(video_dir, sample["video"])
    predicted_answer, resp_text = _run_mcq_inference(video_path, sample["question"], sample["candidates"], processor, model)
    is_correct = (predicted_answer == sample["answer"])

    result = {
        "video": sample["video"],
        "video_id": sample.get("video_id", ""),
        "qa_id": sample.get("qa_id", ""),
        "question": sample["question"],
        "candidates": sample["candidates"],
        "correct_answer": sample["answer"],
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "task_type": sample["task_type"],
        "video_type": sample["video_type"],
        "model_response": resp_text,
    }

    logger.info(f"AVUT processed {sample['video']}: {sample['answer']} -> {predicted_answer} ({'✓' if is_correct else '✗'})")
    return result


def evaluate_sample_ShortVid(sample, video_dir, processor, model):
    video_path = os.path.join(video_dir, sample["video"])
    predicted_answer, resp_text = _run_mcq_inference(video_path, sample["question"], sample["candidates"], processor, model)
    is_correct = (predicted_answer == sample["answer"])

    result = {
        "video": sample["video"],
        "question": sample["question"],
        "candidates": sample["candidates"],
        "correct_answer": sample["answer"],
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "problem_type": sample["problem_type"],
        "data_type": sample["data_type"],
        "model_response": resp_text,
    }

    logger.info(f"ShortVid processed {sample['video']}: {sample['answer']} -> {predicted_answer} ({'✓' if is_correct else '✗'})")
    return result


def _calculate_base_accuracy(results: List[Dict]) -> Dict:
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r["is_correct"])
    overall_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    return {
        "overall_accuracy": overall_accuracy,
        "total_correct": correct_predictions,
        "total_samples": total_samples,
    }


def _calculate_group_accuracy(results: List[Dict], field_name: str) -> Dict:
    field_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for result in results:
        field_value = result.get(field_name, "")
        field_stats[field_value]["total"] += 1
        if result["is_correct"]:
            field_stats[field_value]["correct"] += 1

    field_accuracy = {}
    for field_value, stats in field_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        field_accuracy[field_value] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        }
    return field_accuracy


def calculate_accuracy_WorldSense(results: List[Dict]) -> Dict:
    metrics = _calculate_base_accuracy(results)
    metrics["domain_accuracy"] = _calculate_group_accuracy(results, "domain")
    metrics["problem_type_accuracy"] = _calculate_group_accuracy(results, "problem_type")
    return metrics


def calculate_accuracy_AVUT(results: List[Dict]) -> Dict:
    metrics = _calculate_base_accuracy(results)
    metrics["video_type_accuracy"] = _calculate_group_accuracy(results, "video_type")
    metrics["task_type_accuracy"] = _calculate_group_accuracy(results, "task_type")
    return metrics


def calculate_accuracy_ShortVid(results: List[Dict]) -> Dict:
    metrics = _calculate_base_accuracy(results)
    metrics["data_type_accuracy"] = _calculate_group_accuracy(results, "data_type")
    metrics["problem_type_accuracy"] = _calculate_group_accuracy(results, "problem_type")
    return metrics


def gather_results_to_rank0(my_results):
    """
    阻塞等待所有进程完成推理后，仅在 rank 0 上合并所有结果。
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    logger.info(f"Rank {rank} 已完成本地推理，等待其他进程到达同步点...")
    dist.barrier()

    gathered_list = [None] * world_size if rank == 0 else None
    dist.gather_object(my_results, gathered_list, dst=0)

    if rank != 0:
        return None

    final_total_results = []
    for part in gathered_list:
        if part:
            final_total_results.extend(part)

    return final_total_results


def log_accuracy_metrics(dataset_name: str, metrics: Dict):
    log_result("=" * 50)
    log_result(f"{dataset_name} EVALUATION COMPLETED")
    log_result("=" * 50)
    log_result(f"Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['total_correct']}/{metrics['total_samples']})")

    for key, value in metrics.items():
        if key in {"overall_accuracy", "total_correct", "total_samples"}:
            continue
        log_result(f"{key}:")
        for field_name, stats in value.items():
            log_result(f"  {field_name}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")


def evaluate_dataset_distributed(dataset_name, data_path, video_dir, output_dir, load_fn, evaluate_fn, calculate_fn, processor, model, mini_test_videos=None):
    raw_data = load_json_data(data_path)
    benchmark_data = load_fn(raw_data)
    benchmark_data = limit_samples_by_video_count(benchmark_data, mini_test_videos)
    benchmark_data = split_samples_across_ranks(benchmark_data)
    rank = dist.get_rank()

    if mini_test_videos is not None and rank == 0:
        log_progress(f"{dataset_name} mini test enabled: first {mini_test_videos} videos")
    log_progress(f"{dataset_name} started on rank {rank}, local samples: {len(benchmark_data)}")

    results = []
    for i, sample in enumerate(
        tqdm.tqdm(
            benchmark_data,
            desc=f"Evaluating {dataset_name}",
            disable=(rank != 0),
            file=sys.stdout,
        )
    ):
        logger.info(f"[{dataset_name}] Processing sample {i+1}/{len(benchmark_data)}: {sample['video']}")
        result = evaluate_fn(sample, video_dir, processor, model)
        results.append(result)

    log_progress(f"{dataset_name} local evaluation finished on rank {rank}")

    all_final_results = gather_results_to_rank0(results)

    if rank == 0:
        log_progress(f"{dataset_name} gather finished, total samples: {len(all_final_results)}")
        metrics = calculate_fn(all_final_results)

        final_results_file = os.path.join(output_dir, "final_results_total.json")
        with open(final_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_final_results, f, indent=2, ensure_ascii=False)

        metrics_file = os.path.join(output_dir, "accuracy_metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"[{dataset_name}] 总结果已写入 {final_results_file}")
        logger.info(f"[{dataset_name}] 指标已写入 {metrics_file}")
        log_accuracy_metrics(dataset_name, metrics)

    dist.barrier()


if __name__ == "__main__":
    args = parse_args()

    model_path_map = {
        "qwenomni3b": "/home/gaofeng/Qwen2.5-Omni-3B",
        "qwenomni7b": "/home/gaofeng/omni-dataset/Qwen2.5-Omni-7B",
    }
    model_path = model_path_map[args.model_type]
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config_label = build_run_config_label(args)
    log_prefix = f"{current_time}_{run_config_label}"
    run_output_dir = os.path.join("logs", "eval_shortvid_runs", log_prefix)

    benchmark = {
        "WorldSense": {
            "data_path": "/home/gaofeng/omni-dataset/WorldSense/worldsense_qa.json",
            "video_dir": "/home/gaofeng/omni-dataset/WorldSense/videos",
            "output_dir": os.path.join(run_output_dir, "WorldSense"),
            "load_fn": load_benchmark_WorldSense_data,
            "evaluate_fn": evaluate_sample_WorldSense,
            "calculate_fn": calculate_accuracy_WorldSense,
        },
        "AVUT": {
            "data_path": "/home/gaofeng/omni-dataset/AVUTBenchmark/AV_Human_filtered_data.json",
            "video_dir": "/home/gaofeng/omni-dataset/AVUTBenchmark",
            "output_dir": os.path.join(run_output_dir, "AVUT"),
            "load_fn": load_benchmark_AVUT_data,
            "evaluate_fn": evaluate_sample_AVUT,
            "calculate_fn": calculate_accuracy_AVUT,
        },
        "ShortVid": {
            "data_path": "/home/gaofeng/omni-dataset/ShortVid-Bench/ShortVid-Bench-1k.json",
            "video_dir": "/home/gaofeng/omni-dataset/ShortVid-Bench/videos_compressed",
            "output_dir": os.path.join(run_output_dir, "ShortVid"),
            "load_fn": load_benchmark_ShortVid_data,
            "evaluate_fn": evaluate_sample_ShortVid,
            "calculate_fn": calculate_accuracy_ShortVid,
        },
    }

    selected_benchmark_names = resolve_selected_benchmarks(args.benchmarks, benchmark)

    local_rank = setup()
    rank = dist.get_rank()
    configure_logging(run_output_dir, rank)
    model, processor = load_model(model_path, f"cuda:{local_rank}", args)
    model.eval()

    if rank == 0:
        log_progress(f"Run outputs are being written to {run_output_dir}/")
        log_progress(f"Wrapper method: {args.WAPPER_METHOD}")
        log_progress(f"Selected benchmarks: {', '.join(selected_benchmark_names)}")

    for dataset_name in selected_benchmark_names:
        config = benchmark[dataset_name]
        os.makedirs(config["output_dir"], exist_ok=True)
        evaluate_dataset_distributed(
            dataset_name=dataset_name,
            data_path=config["data_path"],
            video_dir=config["video_dir"],
            output_dir=config["output_dir"],
            load_fn=config["load_fn"],
            evaluate_fn=config["evaluate_fn"],
            calculate_fn=config["calculate_fn"],
            processor=processor,
            model=model,
            mini_test_videos=args.mini_test_videos,
        )