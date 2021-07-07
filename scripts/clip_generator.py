import argparse
import datetime
import math
import random
import shutil
import sys
import time

import click
import clip

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
@click.option('--one-cycle', help='one_cycle', type=bool, default=False)
@click.option('--unreal_engine', help='add unreal and rtx to prompt', type=bool, default=False)
def main(ctx, config, checkpoint, lr, seed, outdir, text, steps, one_cycle, unreal_engine):
    # cannot run on CPU ^_^
    print(locals())
    now = datetime.datetime.now()
    device = 'cuda:0'
    vqgan_model = load_vqgan_model(config, checkpoint)
    clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
    prompts = text.split('//')
    if unreal_engine:
        prompts = [prompt + " (unreal engine) (rtx on)" for prompt in prompts]
    trainer = Trainer(prompts,
                      vqgan_model,
                      clip_model,
                      seed=seed,
                      save_every=50,
                      image_size=(700,700),
                      learning_rate=lr,
                      outdir=f'{outdir}/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{text.replace("//", "_")}/',
                      device=device,
                      steps=steps,
                      crazy_mode=one_cycle)
    try:
        trainer.start()
    finally:
        trainer.close()
        shutil.copy(trainer.get_generated_image_path(),
                    f'{outdir}/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{trainer.prompts[0].replace("//", "_")}.png')


if __name__ == '__main__':
    sys.path.append('..')
    main()
