"""
Avatar model loader - loads all models once and keeps them in memory.
Usage:
    loader = AvatarModelLoader(checkpoint_dir="./weights/LongCat-Video-Avatar", context_parallel_size=1)
    models = loader.load()
"""

import os
import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, UMT5EncoderModel, Wav2Vec2FeatureExtractor
from audio_separator.separator import Separator

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process.wav2vec2 import Wav2Vec2ModelWrapper


class AvatarModels:
    """Container for all loaded models."""
    def __init__(self):
        self.pipe = None
        self.vocal_separator = None
        self.local_rank = None
        self.global_rank = None
        self.cp_rank = None
        self.cp_size = None
        self.audio_output_dir_temp = None


class AvatarModelLoader:
    def __init__(self, checkpoint_dir="./weights/LongCat-Video-Avatar", context_parallel_size=1):
        self.checkpoint_dir = checkpoint_dir
        self.context_parallel_size = context_parallel_size

    def load(self) -> AvatarModels:
        models = AvatarModels()

        # prepare distributed environment
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()

        # initialize context parallel
        context_parallel_util.init_context_parallel(
            context_parallel_size=self.context_parallel_size,
            global_rank=global_rank,
            world_size=num_processes
        )
        cp_rank = context_parallel_util.get_cp_rank()
        cp_size = context_parallel_util.get_cp_size()
        cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

        models.local_rank = local_rank
        models.global_rank = global_rank
        models.cp_rank = cp_rank
        models.cp_size = cp_size

        # initialize video/text models
        base_dir = os.path.join(self.checkpoint_dir, '..', 'LongCat-Video')
        tokenizer = AutoTokenizer.from_pretrained(base_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
        text_encoder = UMT5EncoderModel.from_pretrained(base_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(base_dir, subfolder="vae", torch_dtype=torch.bfloat16)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            self.checkpoint_dir, subfolder="avatar_single", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16
        )

        # initialize audio models
        wav2vec_path = os.path.join(self.checkpoint_dir, 'chinese-wav2vec2-base')
        audio_encoder = Wav2Vec2ModelWrapper(wav2vec_path).to(local_rank)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path, local_files_only=True)

        # initialize vocal separator
        vocal_separator_path = os.path.join(self.checkpoint_dir, 'vocal_separator/Kim_Vocal_2.onnx')
        audio_output_dir_temp = Path("./audio_temp_file")
        audio_output_dir_temp.mkdir(exist_ok=True)
        audio_separator_model_path = os.path.dirname(vocal_separator_path)
        audio_separator_model_name = os.path.basename(vocal_separator_path)
        vocal_separator = Separator(
            output_dir=audio_output_dir_temp / "vocals",
            output_single_stem="vocals",
            model_file_dir=audio_separator_model_path,
        )
        vocal_separator.load_model(audio_separator_model_name)

        models.vocal_separator = vocal_separator
        models.audio_output_dir_temp = audio_output_dir_temp

        # initialize pipeline
        pipe = LongCatVideoAvatarPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
            audio_encoder=audio_encoder,
            wav2vec_feature_extractor=wav2vec_feature_extractor,
        )
        pipe.to(local_rank)
        models.pipe = pipe

        if local_rank == 0:
            print("All models loaded and ready.")

        return models
