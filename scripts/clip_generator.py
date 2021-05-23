import argparse
import math
import sys

import click
import torch

from clip_generator.trainer import Trainer


# @click.command()
# @click.pass_context
# @click.option('--network', 'network', help='a jsonfile containing 2 keys: checkpoint and config', required=True)
# @click.option('--lr', 'lr', help='Learning rate', default=0.05)
# @click.option('--seed', type=int, help='The seed')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
# @click.option('--text', help='text', type=str, required=True)
# @click.option('--one-cycle', 'one_cycle', type=bool, required=False)
# @click.option('--steps-per-epoch', help='num step', type=int, default=100)
# @click.option('--epochs', help='num step', type=int, default=1)
# @click.option('--base-image', help='num step', type=int, default=1)
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(["A wonderful apple"], '../models/imagenet/vqgan_imagenet_f16_16384.ckpt', "../models/imagenet/vqgan_imagenet_f16_16384.yaml")
    trainer.train_loop()


if __name__ == '__main__':
    sys.path.append('..')
    main()
