"""
Avatar video generation server.
Loads models once at startup, then accepts generation requests via stdin.

Usage (single GPU):
    torchrun --nproc_per_node=1 avatar_server.py --checkpoint_dir ./weights/LongCat-Video-Avatar

Usage (multi GPU with context parallel):
    torchrun --nproc_per_node=2 avatar_server.py --checkpoint_dir ./weights/LongCat-Video-Avatar --context_parallel_size 2

Once running, type JSON requests (one per line) to generate videos:
    {"input_json": "assets/avatar/single_example_1.json"}
    {"input_json": "assets/avatar/single_example_1.json", "resolution": "720p", "output_dir": "./my_output"}

Type 'quit' or 'exit' to stop the server.
"""

import sys
import json
import argparse
import traceback

import torch
import torch.distributed as dist

from avatar_model_loader import AvatarModelLoader
from avatar_generate import generate


def broadcast_string(s, src=0):
    """Broadcast a string from src rank to all other ranks."""
    if dist.get_rank() == src:
        data = s.encode('utf-8')
        length = torch.tensor([len(data)], dtype=torch.long, device='cuda')
    else:
        length = torch.tensor([0], dtype=torch.long, device='cuda')
    dist.broadcast(length, src=src)
    n = length.item()
    if dist.get_rank() == src:
        tensor = torch.ByteTensor(list(data)).cuda()
    else:
        tensor = torch.zeros(n, dtype=torch.uint8, device='cuda')
    dist.broadcast(tensor, src=src)
    return tensor.cpu().numpy().tobytes().decode('utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="./weights/LongCat-Video-Avatar")
    parser.add_argument("--context_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # load models once
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)
    loader = AvatarModelLoader(
        checkpoint_dir=args.checkpoint_dir,
        context_parallel_size=args.context_parallel_size,
    )
    models = loader.load()
    print("=" * 60)
    print("Server ready. Enter JSON requests (one per line).")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    is_main = (models.local_rank == 0)

    while True:
        # only rank 0 reads from stdin, then broadcasts to all ranks
        line = ""
        if is_main:
            try:
                line = sys.stdin.readline()
            except EOFError:
                line = ""
            if not line:
                line = "__EXIT__"
            else:
                line = line.strip()
                if not line:
                    line = "__SKIP__"
                elif line.lower() in ('quit', 'exit'):
                    line = "__EXIT__"

        line = broadcast_string(line, src=0)

        if line == "__SKIP__":
            continue
        if line == "__EXIT__":
            if is_main:
                print("Shutting down.")
            break

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            if is_main:
                print(f"Invalid JSON: {e}")
            continue

        try:
            if is_main:
                print(f"Starting generation: {request.get('input_json', '?')}")
            generate(models, request)
            if is_main:
                print("Done.")
        except Exception:
            if is_main:
                traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
