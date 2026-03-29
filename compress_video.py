import os
import subprocess
import shutil

MAX_SIZE_MB = 90
FOLDER = "/home/gaofeng/omni-dataset/ShortVid-Bench/videos"

OUTPUT_FOLDER = "/home/gaofeng/omni-dataset/ShortVid-Bench/videos_compressed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def compress_video(video_path, output_path):
    if os.path.exists(output_path):
        print(f"  Result already exists, skipping: {os.path.basename(output_path)}")
        return
    
    duration = get_video_duration(video_path)

    target_size_bits = MAX_SIZE_MB * 1024 * 1024 * 8
    bitrate = int(target_size_bits / duration * 0.95)  # 留一点余量


    #temp_output = video_path + ".tmp.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_output = output_path
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-b:v", str(bitrate),
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "96k",
        temp_output
    ]

    subprocess.run(cmd)

#    if os.path.exists(temp_output):
#        os.replace(temp_output, video_path)

def process_folder(folder):

    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(".mp4"):
                continue

            path = os.path.join(root, f)
            rel_path = os.path.relpath(path, folder)
            output_path = os.path.join(OUTPUT_FOLDER, rel_path)

            size_mb = os.path.getsize(path) / (1024 * 1024)

            if size_mb <= MAX_SIZE_MB:
                if not os.path.exists(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    print(f"Copying (small enough): {f} ({size_mb:.1f}MB)")
                    shutil.copy2(path, output_path)
                else:
                    print(f"Already exists, skip copy: {f}")
                continue

            print(f"compressing: {f} ({size_mb:.1f}MB)")
            compress_video(path, output_path)

process_folder(FOLDER)