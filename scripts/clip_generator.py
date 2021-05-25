import argparse
import math
import sys

import click
import clip
import torch

from clip_generator.trainer import Trainer
from clip_generator.dreamer import load_vqgan_model


@click.command()
@click.pass_context
@click.option('--config', 'config', help='config', required=True)
@click.option('--checkpoint', 'checkpoint', help='checkpoint', required=True)
@click.option('--lr', 'lr', help='Learning rate', default=0.05)
@click.option('--seed', type=int, help='The seed', default=None)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--text', help='text', type=str, required=True)
@click.option('--steps', help='nb_steps', type=int, default=None)
def main(ctx, config, checkpoint, lr, seed, outdir, text, steps):
    # cannot run on CPU ^_^
    device = 'cuda:0'
    vqgan_model = load_vqgan_model(config, checkpoint)
    clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
    trainer = Trainer(text.split('//') ,vqgan_model, clip_model, learning_rate=lr, outdir=outdir, device=device, steps=steps)
    trainer.start()


if __name__ == '__main__':
    sys.path.append('..')
    main()
