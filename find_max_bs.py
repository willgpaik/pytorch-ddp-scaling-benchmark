#!/usr/bin/env python3
"""
find_max_bs.py: Find the largest safe per-GPU batch size for a given model and precision.
Doubles batch size until OOM, then recommends the last successful value (already a power of 2).

Usage:
    python find_max_bs.py --model resnet152 --precision fp16
    python find_max_bs.py --model vit_b_16  --precision bf16
"""

import argparse
import gc
import contextlib

import torch
import torch.nn as nn
from torchvision import models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet152",
                   choices=["resnet50", "resnet101", "resnet152", "vit_b_16"])
    p.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--start-bs", type=int, default=32,
                   help="Starting batch size (doubles each step)")
    return p.parse_args()


def is_vit(name):
    return name.startswith("vit")


def one_step(model, optimizer, scaler, images, labels, amp_dtype, use_amp):
    optimizer.zero_grad(set_to_none=True)
    with (torch.amp.autocast("cuda", dtype=amp_dtype) if use_amp
          else contextlib.nullcontext()):
        out = model(images)
        loss = nn.CrossEntropyLoss()(out, labels)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def probe(model, optimizer, scaler, bs, image_size, num_classes,
          device, amp_dtype, use_amp, use_channels_last):
    torch.cuda.reset_peak_memory_stats(device)
    images = torch.randn(bs, 3, image_size, image_size, device=device)
    if use_channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    labels = torch.randint(0, num_classes, (bs,), device=device)
    one_step(model, optimizer, scaler, images, labels, amp_dtype, use_amp)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    total_mem = torch.cuda.get_device_properties(device).total_memory
    total_gb = total_mem / (1024 ** 3)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[args.precision]
    use_amp = args.precision != "fp32"
    use_channels_last = not is_vit(args.model)

    print(f"GPU   : {torch.cuda.get_device_name(device)}")
    print(f"Total : {total_gb:.1f} GB")
    print(f"Model : {args.model}  Precision: {args.precision}")
    print()

    model = getattr(models, args.model)(num_classes=args.num_classes)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    print("Scanning (2x each step, stop at OOM):")
    bs = args.start_bs
    last_good_bs = None
    last_good_mem = None

    while True:
        try:
            peak = probe(model, optimizer, scaler, bs,
                         args.image_size, args.num_classes,
                         device, amp_dtype, use_amp, use_channels_last)
            peak_gb = peak / (1024 ** 3)
            pct = peak / total_mem * 100
            print(f"  BS={bs:>6}: {peak_gb:.2f} GB ({pct:.1f}%%) -- OK")
            last_good_bs = bs
            last_good_mem = peak
            bs = bs * 2

        except torch.cuda.OutOfMemoryError:
            print(f"  BS={bs:>6}: OOM")
            torch.cuda.empty_cache()
            gc.collect()
            break

    if last_good_bs is None:
        print(f"\nERROR: Even BS={args.start_bs} OOMs.")
        return

    # Largest power of 2 (last_good_bs is already a power of 2 due to doubling)
    final_gb = last_good_mem / (1024 ** 3)
    print()
    print("=" * 50)
    print(f"Recommended BS : {last_good_bs}")
    print(f"Peak memory    : {final_gb:.2f} GB / {total_gb:.1f} GB "
          f"({final_gb/total_gb*100:.1f}%%)")
    print()
    print(f"  sweep weak  : PER_GPU_BS={last_good_bs}")
    print(f"  sweep strong: GLOBAL_BS (1-GPU max) = {last_good_bs}")
    print("=" * 50)


if __name__ == "__main__":
    main()
