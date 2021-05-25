import itertools
import shutil
import time
from pathlib import Path

import clip
import torch
from progressbar import progressbar
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.transforms import functional as TF

from .discriminator import ClipDiscriminator
from .dreamer import Generator
from .dreamer import ZSpace


class Trainer:
    def __init__(self,
                 prompts,
                 vqgan_model,
                 clip_model,
                 device='cuda:0',
                 learning_rate=0.05,
                 outdir='./out',
                 image_size=(512, 512),
                 cutn=64,
                 cut_pow=1.,
                 seed=None,
                 steps=None,
                 save_every=50):

        if seed is None:
            torch.manual_seed(int(time.time()))
        else:
            torch.manual_seed(seed)

        self.save_every = save_every
        self.outdir = Path(outdir)
        if steps is None:
            self.iterator = itertools.count(start=0)
        else:
            self.iterator = range(steps)

        self.outdir.mkdir(exist_ok=True, parents=True)
        self.clip_discriminator = ClipDiscriminator(clip_model, prompts, cutn, cut_pow, device)

        self.generator = Generator(vqgan_model).to(device)
        self.z_space = ZSpace(vqgan_model, image_size, device=device)
        self.optimizer = optim.Adam([self.z_space.z], lr=learning_rate)
        self.scheduler = None
        #if steps is not None:
        #    self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=learning_rate * 10, total_steps=steps)

    def get_generated_image_path(self):
        return self.outdir / f'progress_latest.png'

    @torch.no_grad()
    def save_image(self, i, generated_image, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        pil_image = TF.to_pil_image(generated_image[0].cpu())
        pil_image.save(str(self.outdir / f'progress_{i}.png'))
        shutil.copy(
            str(self.outdir / f'progress_{i}.png'),
            str(self.outdir / f'progress_latest.png')
        )

    def start(self):
        for it in self.epoch():
            ...

    def epoch(self):
        for i in progressbar(self.iterator):
            self.optimizer.zero_grad()
            generated_image = self.generator(self.z_space.z)
            losses = self.clip_discriminator(generated_image)
            if i % self.save_every == 0:
                self.save_image(i, generated_image, losses)
                yield i

            sum(losses).backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.z_space.clamp()
            i = i + 1
