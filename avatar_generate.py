"""
Avatar video generation - uses pre-loaded models to generate video.
Separated from model loading so models can be loaded once and reused.
"""

import os
import math
import time
import random
import json

import numpy as np
import torch
import PIL.Image
import librosa
from diffusers.utils import load_image

from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process.torch_utils import save_video_ffmpeg


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    return timestamp_part + random_part


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None
    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    cmd = f"mv '{default_vocal_path}' '{target_path}'"
    os.system(cmd)
    return target_path


def generate(models, request):
    """
    Generate avatar video using pre-loaded models.

    Args:
        models: AvatarModels instance from AvatarModelLoader.load()
        request: dict with keys:
            - input_json (str): path to input JSON file
            - output_dir (str): output directory (default: ./outputs_avatar_single)
            - resolution (str): '480p' or '720p' (default: '480p')
            - num_segments (int): 0 = auto from audio length (default: 0)
            - num_inference_steps (int): default 50
            - text_guidance_scale (float): default 4.0
            - audio_guidance_scale (float): default 4.0
            - stage_1 (str): 'ai2v' or 'at2v' (default: 'ai2v')
            - ref_img_index (int): default 10
            - mask_frame_range (int): default 3
            - seed (int): random seed (default: 42)
    """
    pipe = models.pipe
    vocal_separator = models.vocal_separator
    audio_output_dir_temp = models.audio_output_dir_temp
    local_rank = models.local_rank
    global_rank = models.global_rank
    cp_rank = models.cp_rank

    # parse request with defaults
    input_json = request['input_json']
    output_dir = request.get('output_dir', './outputs_avatar_single')
    resolution = request.get('resolution', '480p')
    num_segments = request.get('num_segments', 0)
    num_inference_steps = request.get('num_inference_steps', 50)
    text_guidance_scale = request.get('text_guidance_scale', 4.0)
    audio_guidance_scale = request.get('audio_guidance_scale', 4.0)
    stage_1 = request.get('stage_1', 'ai2v')
    ref_img_index = request.get('ref_img_index', 10)
    mask_frame_range = request.get('mask_frame_range', 3)
    global_seed = request.get('seed', 42)

    os.makedirs(output_dir, exist_ok=True)

    # default inference params
    save_fps = 16
    num_frames = 93
    num_cond_frames = 13
    audio_stride = 2

    if resolution == '480p':
        height, width = 480, 832
    elif resolution == '720p':
        height, width = 768, 1280

    # load input data
    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    prompt = input_data['prompt']
    negative_prompt = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    raw_speech_path = input_data['cond_audio']['person1']

    # auto-calculate num_segments from audio length
    if num_segments <= 0:
        import soundfile as sf
        audio_info = sf.info(raw_speech_path)
        audio_duration = audio_info.duration
        first_segment_duration = num_frames / save_fps
        next_segment_duration = (num_frames - num_cond_frames) / save_fps
        num_segments = max(1, math.ceil((audio_duration - first_segment_duration) / next_segment_duration) + 1)
        print(f"Audio duration: {audio_duration:.1f}s -> auto num_segments: {num_segments}")

    # seed
    seed = global_seed + global_rank
    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed)

    # audio processing
    if cp_rank == 0:
        temp_vocal_path = extract_vocal_from_speech(
            raw_speech_path,
            f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav",
            vocal_separator,
            audio_output_dir_temp
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

    # prepare audio embedding for the first clip
    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames
    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)

    if local_rank == 0:
        print(f"Generating segment 1/{num_segments}...")

    # stage 1 generation
    if stage_1 == 'at2v':
        output_tuple = pipe.generate_at2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            audio_emb=audio_emb,
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, "at2v_demo_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()

    elif stage_1 == 'ai2v':
        image_path = input_data['cond_image']
        image = load_image(image_path)
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type='both',
            generator=generator,
            audio_emb=audio_emb,
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, "ai2v_demo_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    if context_parallel_util.get_cp_size() > 1:
        torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    # long video generation (continuation segments)
    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    all_generated_frames = video

    for segment_idx in range(1, num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx + 1}/{num_segments}...")

        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
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
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type='both',
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
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
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"video_continue_{segment_idx + 1}"), raw_speech_path, fps=save_fps, quality=5)
            del output_tensor

    if local_rank == 0:
        print("Generation complete.")
