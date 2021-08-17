import asyncio
import datetime
import os

import shutil
import threading
from typing import Dict, Callable

import clip
import discord

from scripts.clip_generator.dreamer import load_vqgan_model
from scripts.clip_generator.trainer import network_list, Trainer
from scripts.clip_generator_irc import GenerationArgs, parse_prompt_args
from scripts.guided_diffusion_hd.clip_guided import generate
from scripts.guided_diffusion_hd.clip_guided import Trainer as Diffusion_trainer


class DreamerClient(discord.Client):
    def __init__(self, **options):
        super().__init__(**options)

        self.clip = None#
        self.current_user = None
        self.stop_flag = False
        self.commands = self.make_commands()
        self.default_arguments = {}

    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')

    def make_commands(self) -> Dict[str, Callable[[discord.Message], None]]:
        return {
            '!generate': self.generate_command,
            '!generate_diffusion': self.generate_diffusion_command,
            '!stop': self.stop_command,
            '!set': self.set_command,
        }

    def set_command(self, message: discord.Message):
        [attribute, value] = message.content[len("!set") + 1:].split("=")
        print(attribute, value)
        self.default_arguments[attribute] = value

    def stop_command(self, message: discord.Message):
        if self.current_user == message.author:
            self.stop_flag = True
        else:
            self.loop.create_task(message.channel.send(f'Only {self.current_user} can stop me'))

    def generate_command(self, message: discord.Message):
        prompt = message.content[len("!generate"):]
        try:
            arguments = parse_prompt_args('--prompt osef')
            arguments.prompt = prompt
        except Exception as ex:
            print(ex)
            self.loop.create_task(message.channel.send('error: ' + str(ex)))
            return
        self.loop.create_task(message.channel.send(f'generating {arguments.prompt}'))
        print(arguments)
        trainer = self.generate_image(arguments)
        self.arguments = arguments

        self.current_user = message.author
        self.stop_flag = False

        self.generating_thread = threading.Thread(target=self.train, args=(trainer, message.channel))
        self.generating_thread.start()

    def generate_diffusion_command(self, message: discord.Message):
        prompt = message.content[len("!generate_diffusion"):]
        try:
            arguments = parse_prompt_args('--prompt osef')
            arguments.prompt = prompt
        except Exception as ex:
            print(ex)
            self.loop.create_task(message.channel.send('error: ' + str(ex)))
            return
        self.loop.create_task(message.channel.send(f'generating {arguments.prompt}'))
        print(arguments)
        trainer = self.generate_image_diffusion(arguments)
        self.arguments = arguments

        self.current_user = message.author
        self.stop_flag = False

        self.generating_thread = threading.Thread(target=self.train, args=(trainer, message.channel))
        self.generating_thread.start()

    async def on_message(self, message: discord.Message):
        if message.author == self.user:
            return

        print(f'{message.channel!r}')

        args = message.content.split()
        if args[0] in self.commands:
            self.commands[args[0]](message)

    async def send_progress(self, trainer, channel, iteration):
        print('sending progress')
        await channel.send(f'step {iteration} / {trainer.steps}', file=discord.File(trainer.get_generated_image_path()))

    async def _train(self, trainer, channel):
        now = datetime.datetime.now()
        for it in trainer.epoch():
            if it % 100 == 0:
                await self.send_progress(trainer, channel, it)
            await asyncio.sleep(0)
            if self.stop_flag:
                break

        await channel.send('Done !')
        trainer.close()
        shutil.copy(trainer.get_generated_image_path(),
                    f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{trainer.prompt.replace("//", "_")}.png')
        await self.send_progress(trainer, channel, trainer.steps)

    def train(self, trainer, channel):
        self.loop.create_task(self._train(trainer, channel))

    def generate_image_diffusion(self, arguments: GenerationArgs):
        print('default arguments', self.default_arguments)
        final_arguments = {**arguments.dict(), **self.default_arguments}
        print(final_arguments)
        arguments = GenerationArgs(**final_arguments)
        now = datetime.datetime.now()

        trainer = Trainer(arguments.prompt.split('||'),
            vqgan_model=load_vqgan_model(arguments.config, arguments.checkpoint).to('cuda'),
            clip_model=clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to('cuda:0'),
            learning_rate = arguments.learning_rate,
            save_every = arguments.refresh_every,
            outdir = f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{arguments.prompt}',
            device = 'cuda:0',
            image_size = (600, 600),
            crazy_mode = arguments.crazy_mode,
            cutn = arguments.cut,
            steps = arguments.steps,
            full_image_loss = True,
            nb_augments = 1,
        )
        return trainer

    def generate_image(self, arguments: GenerationArgs):
        print('default arguments', self.default_arguments)
        final_arguments = {**arguments.dict(), **self.default_arguments}
        print(final_arguments)
        arguments = GenerationArgs(**final_arguments)
        vqgan_model = None #load_vqgan_model(arguments.config, arguments.checkpoint).to('cuda')
        now = datetime.datetime.now()
        trainer = Diffusion_trainer(arguments.prompt.split('||')[0],
            learning_rate = arguments.learning_rate,
            save_every = arguments.refresh_every,
            outdir = f'./discord_out_diffusion/{now.strftime("%Y_%m_%d")}/{now.isoformat()}_{arguments.prompt}',
            device = 'cuda:0',
            image_size = (600, 600),
            crazy_mode = arguments.crazy_mode,
            cutn = arguments.cut,
            steps = arguments.steps,
            full_image_loss = True,
            nb_augments = 1,
        )
        return trainer

TOKEN = os.getenv('DISCORD_API_KEY')
client = DreamerClient()
client.run(TOKEN)
