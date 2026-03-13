"""
Resume avatar video generation from a previously generated video.

Extracts frames from the existing mp4, VAE-encodes them to reconstruct
latents, then continues generation from the next segment.

Usage:
    torchrun avatar_resume.py \
        --checkpoint_dir=./weights/LongCat-Video-Avatar \
        --input_json=assets/avatar/single_example_1.json \
        --resume_video=./outputs_avatar_single/video_continue_24.mp4 \
        --resume_segment=24 \
        --audio_guidance_scale=5.0 \
        --resolution=720p \
        --ref_img_index=-10 \
        --mask_frame_range=5
"""

import os
import math
import time
import random
import json
import argparse

import numpy as np
import torch
import PIL.Image
import librosa
import cv2

from avatar_model_loader import AvatarModelLoader
from avatar_generate import extract_vocal_from_speech, generate_random_uid, torch_gc
from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process.torch_utils import save_video_ffmpeg


def extract_frames_from_video(video_path):
    """Extract all frames from mp4 as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(PIL.Image.fromarray(frame_rgb))
    cap.release()
    return frames


def encode_frames_to_latent(pipe, frames, device):
    """VAE-encode a list of PIL frames to get normalized latent."""
    video_tensor = pipe.video_processor.preprocess_video(
        pipe.video_processor, frames,
        height=frames[0].size[1], width=frames[0].size[0],
        resize_mode="crop"
    )
    video_tensor = video_tensor.to(device=device, dtype=pipe.vae.dtype)

    from longcat_video.pipeline_longcat_video_avatar import retrieve_latents
    latent = retrieve_latents(pipe.vae.encode(video_tensor), sample_mode="argmax")
    latent = pipe.normalize_latents(latent)
    return latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./weights/LongCat-Video-Avatar")
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--resume_video", type=str, required=True, help="Path to the last saved mp4")
    parser.add_argument("--resume_segment", type=int, required=True, help="Last completed segment number")
    parser.add_argument("--output_dir", type=str, default="./outputs_avatar_single")
    parser.add_argument("--resolution", type=str, default="480p", choices=["480p", "720p"])
    parser.add_argument("--num_segments", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--text_guidance_scale", type=float, default=4.0)
    parser.add_argument("--audio_guidance_scale", type=float, default=4.0)
    parser.add_argument("--stage_1", type=str, default="ai2v", choices=["ai2v", "at2v"])
    parser.add_argument("--ref_img_index", type=int, default=10)
    parser.add_argument("--mask_frame_range", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # load models
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)
    loader = AvatarModelLoader(
        checkpoint_dir=args.checkpoint_dir,
        context_parallel_size=args.context_parallel_size,
    )
    models = loader.load()

    pipe = models.pipe
    vocal_separator = models.vocal_separator
    audio_output_dir_temp = models.audio_output_dir_temp
    local_rank = models.local_rank
    global_rank = models.global_rank
    cp_rank = models.cp_rank

    # default inference params
    save_fps = 16
    num_frames = 93
    num_cond_frames = 13
    audio_stride = 2

    if args.resolution == "480p":
        height, width = 480, 832
    elif args.resolution == "720p":
        height, width = 768, 1280

    # load input data
    with open(args.input_json, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    prompt = input_data["prompt"]
    negative_prompt = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    raw_speech_path = input_data["cond_audio"]["person1"]

    # auto-calculate num_segments
    num_segments = args.num_segments
    if num_segments <= 0:
        import soundfile as sf
        audio_info = sf.info(raw_speech_path)
        audio_duration = audio_info.duration
        first_segment_duration = num_frames / save_fps
        next_segment_duration = (num_frames - num_cond_frames) / save_fps
        num_segments = max(1, math.ceil((audio_duration - first_segment_duration) / next_segment_duration) + 1)
        print(f"Audio duration: {audio_duration:.1f}s -> auto num_segments: {num_segments}")

    resume_segment = args.resume_segment
    if resume_segment >= num_segments:
        print(f"Already completed: resume_segment={resume_segment} >= num_segments={num_segments}")
        torch.distributed.destroy_process_group()
        return

    print(f"Resuming from segment {resume_segment + 1}/{num_segments}")

    # seed - must match original run
    seed = args.seed + global_rank
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    # advance generator state to match where original run was
    # each segment uses the generator, so we need to consume the same random states
    # This is approximate - the exact state depends on internal pipeline usage
    # For better results, we re-encode from the video frames

    # audio processing
    if cp_rank == 0:
        temp_vocal_path = extract_vocal_from_speech(
            raw_speech_path,
            f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav",
            vocal_separator,
            audio_output_dir_temp,
        )
        assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

        generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
        source_duration = len(speech_array) / sr
        added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.0] * added_sample_nums)

        full_audio_emb = pipe.get_audio_embedding(speech_array, fps=save_fps * audio_stride, device=local_rank, sample_rate=sr)
        if torch.isnan(full_audio_emb).any():
            raise ValueError("broken audio embedding with nan values")

        if context_parallel_util.get_cp_size() > 1:
            full_audio_emb_shape_list = list(full_audio_emb.size())
            full_audio_emb_tensor_shape_list = torch.tensor(full_audio_emb_shape_list, dtype=torch.int64, device=full_audio_emb.device)
            context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
            context_parallel_util.cp_broadcast(full_audio_emb)

        if os.path.exists(temp_vocal_path):
            os.remove(temp_vocal_path)

    elif context_parallel_util.get_cp_size() > 1:
        full_audio_emb_tensor_shape_list = torch.zeros(3, dtype=torch.int64, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
        full_audio_emb_shape_list = full_audio_emb_tensor_shape_list.tolist()
        full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb)

    # extract frames from resume video
    if local_rank == 0:
        print(f"Extracting frames from {args.resume_video}...")
    all_generated_frames = extract_frames_from_video(args.resume_video)
    if local_rank == 0:
        print(f"Extracted {len(all_generated_frames)} frames")

    # get the last segment's frames (last 93 frames)
    current_video = all_generated_frames[-num_frames:]

    # VAE-encode to reconstruct latents
    if local_rank == 0:
        print("VAE-encoding frames to reconstruct latents...")

    latent = encode_frames_to_latent(pipe, current_video, local_rank)

    # ref_latent from first frame
    first_frame_latent = encode_frames_to_latent(pipe, [all_generated_frames[0]], local_rank)
    ref_latent = first_frame_latent[:, :, :1].clone()
    del first_frame_latent

    if local_rank == 0:
        print(f"Latent shape: {latent.shape}, ref_latent shape: {ref_latent.shape}")

    # calculate audio_start_idx for the resume point
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    for seg_i in range(resume_segment):
        if seg_i == 0:
            audio_start_idx = 0
        else:
            audio_start_idx += audio_stride * (num_frames - num_cond_frames)
    # after the loop, audio_start_idx is at the START of the last completed segment
    # we need to advance once more for the next segment
    if resume_segment > 0:
        audio_start_idx += audio_stride * (num_frames - num_cond_frames)

    # continue generation from resume_segment onwards
    os.makedirs(args.output_dir, exist_ok=True)
    width, height = current_video[0].size

    for segment_idx in range(resume_segment, num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx + 1}/{num_segments}...")

        audio_end_idx = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)

        output_tuple = pipe.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=args.num_inference_steps,
            text_guidance_scale=args.text_guidance_scale,
            audio_guidance_scale=args.audio_guidance_scale,
            generator=generator,
            output_type="both",
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=args.ref_img_index,
            mask_frame_range=args.mask_frame_range,
        )
        output, latent = output_tuple
        output = output[0]
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_generated_frames))
            save_video_ffmpeg(
                output_tensor,
                os.path.join(args.output_dir, f"video_continue_{segment_idx + 1}"),
                raw_speech_path,
                fps=save_fps,
                quality=5,
            )
            del output_tensor

        # advance audio position for next segment
        audio_start_idx += audio_stride * (num_frames - num_cond_frames)

        torch_gc()

    if local_rank == 0:
        print("Resume generation complete.")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
