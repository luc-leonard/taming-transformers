import asyncio
import datetime
import os

import shutil
import threading


import clip
import discord

from scripts.clip_generator.dreamer import load_vqgan_model
from scripts.clip_generator.trainer import network_list, Trainer
from scripts.clip_generator_irc import GenerationArgs, parse_prompt_args


TOKEN = os.getenv('DISCORD_API_KEY')

class DreamerClient(discord.Client):
    def __init__(self, **options):
        super().__init__(**options)
        self.clip = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
        self.current_user = None
        self.stop_flag = False

    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        print(f'{message.channel!r}')

        args = message.content.split()
        if message.content.startswith('!stop'):
            if self.current_user == message.author:
                self.stop_flag = True
            else:
                await message.channel.send(f'Only {self.current_user} can stop me')

        if args[0] == '!generate':
            prompt = ' '.join(args[1:])
            try:
                arguments = parse_prompt_args('--prompt osef')
                arguments.prompt = prompt
            except Exception as ex:
                message.channel.send(str(ex))
                return
            await message.channel.send(f'generating {arguments.prompt}')

            trainer = self.generate_image(arguments)

            self.current_user = message.author
            self.stop_flag = False

            self.generating_thread = threading.Thread(target=self.train, args=(trainer, message.channel))
            self.generating_thread.start()



    async def send_progress(self, trainer, channel, iteration):
        print('sending progress')
        with open(trainer.get_generated_image_path(), 'rb') as fp:
            await channel.send(f'step {iteration} / {trainer.steps}', file=discord.File(fp))

    async def _train(self, trainer, channel):
        now = datetime.datetime.now()
        for it in trainer.epoch():
            if it > 0 and it % 100 == 0:
                await self.send_progress(trainer, channel, it)
            await asyncio.sleep(0)
            if self.stop_flag:
                break

        await channel.send('Done !')
        trainer.close()
        shutil.copy(trainer.get_generated_image_path(),
                    f'./discord_out/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{trainer.prompts[0].replace("//", "_")}.png')
        await self.send_progress(trainer, channel, trainer.steps)

    def train(self, trainer, channel):
        self.loop.create_task(self._train(trainer, channel))

    def generate_image(self, arguments: GenerationArgs):
        vqgan_model = load_vqgan_model(arguments.config, arguments.checkpoint).to('cuda')
        now = datetime.datetime.now()
        trainer = Trainer(arguments.prompt.split('||'),
                          vqgan_model,
                          self.clip,
                          learning_rate=arguments.learning_rate,
                          save_every=arguments.refresh_every,
                          outdir=f'./discord_out/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{arguments.prompt}',
                          device='cuda:0',
                          image_size=(640, 640),
                          crazy_mode=arguments.crazy_mode,
                          cutn=arguments.cut,
                          steps=arguments.steps,
                          full_image_loss=arguments.full_image_loss,
                          nb_augments=arguments.nb_augments,
                          )
        return trainer


client = DreamerClient()
client.run(TOKEN)
