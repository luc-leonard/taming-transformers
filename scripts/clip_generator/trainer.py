import clip
import torch
from torch import optim

from .discriminator import ClipDiscriminator
from .dreamer import load_vqgan_model, ZSpace
from .dreamer import Generator
from torchvision.transforms import functional as TF


class Trainer:
    def __init__(self,
                 prompts,
                 vqgan_checkpoint,
                 vqgan_config,
                 learning_rate=0.05,
                 image_size=(480, 480),
                 cutn = 64,
                 cut_pow=1.):

        clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).cuda()
        self.clip_discriminator = ClipDiscriminator(clip_model, prompts, cutn, cut_pow, 'cuda')

        vqgan = load_vqgan_model(vqgan_config, vqgan_checkpoint)
        self.generator = Generator(vqgan).cuda()
        self.z_space = ZSpace(vqgan, image_size, self.clip_discriminator.make_cutouts, 'cuda')
        self.optimizer = optim.Adam([self.z_space.z], lr=learning_rate)

    @torch.no_grad()
    def show_image(self, i, generated_image, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        print(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        TF.to_pil_image(generated_image[0].cpu()).save(f'progress_{i}.png')
        print(i, losses)

    def train_loop(self):
        i = 0
        while True:
            self.optimizer.zero_grad()
            generated_image = self.generator(self.z_space.z)
            losses = self.clip_discriminator(generated_image)
            if i % 50 == 0:
                self.show_image(i, generated_image, losses)

            sum(losses).backward()
            self.optimizer.step()
            self.z.clamp()
            i = i + 1
