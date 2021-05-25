import argparse
import random
import threading
from typing import List

import clip
import irc
import irc.bot

from clip_generator.trainer import Trainer
from clip_generator.dreamer import load_vqgan_model
import argparse

from dataclasses import dataclass


@dataclass
class GenerationArgs:
    prompt: str
    crazy_mode: bool
    steps: int
    refresh_every: int


def parse_args(args: List[str]) -> GenerationArgs:
    return GenerationArgs()

class IrcBot(irc.bot.SingleServerIRCBot):

    def __init__(self, channel: str, nickname: str, server: str, port=6667):
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port)], nickname, nickname)
        self.generating = None
        self.generating_thread = None
        self.channel = channel
        self.current_generating_user = None
        self.stop_generating = False
        print('loading clip')
        self.clip = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
        print('loading VQGAN')
        self.vqgan_model = load_vqgan_model('../models/imagenet/vqgan_imagenet_f16_16384.yaml', '../models/imagenet/vqgan_imagenet_f16_16384.ckpt')

    def on_nicknameinuse(self, c: irc.client, e):
        c.nick(c.get_nickname() + "_")

    def on_welcome(self, c, e):
        c.join(self.channel)

    def on_privmsg(self, c: irc.client.ServerConnection, e: irc.client.Event):
        ...

    def train(self, trainer, c):
        for it in trainer.epoch():
            if self.stop_generating is True:
                break
            if it % 100 == 0:
                c.privmsg(self.channel, f'generation {it}/500')
        self.generating = None
        c.privmsg(self.channel, f'Generation done. Ready to take an order')

    def generate_image(self, prompt):
        trainer = Trainer([prompt],
                          self.vqgan_model,
                          self.clip,
                          learning_rate=0.05,
                          save_every=10,
                          outdir=f'./irc_out/{random.randint(0, 50000)}',
                          device='cuda:0',
                          steps=500)
        return trainer

    def on_pubmsg(self, c, e):
        text = e.arguments[0]
        if len(text) > 100:
            text = text[:99]
        args = text.split()

        if args[0] == '!' + 'generate':
            if self.generating is not None:
                c.privmsg(e.target, f'currently generating {self.generating}, try again later.')
                return
            prompt = ' '.join(args[1:])

            c.privmsg(e.target, f'generating {prompt}')
            trainer = self.generate_image(prompt)
            generated_image_path = trainer.get_generated_image_path()
            c.privmsg(e.target, f'{prompt} => http://82.65.144.151:8082/{generated_image_path.relative_to(".")}')
            self.generating = prompt
            self.stop_generating = False
            self.current_generating_user = e.source
            self.generating_thread = threading.Thread(target=self.train, args=(trainer,c))
            self.generating_thread.start()

        if args[0] == '!' + 'stop' and e.source == self.current_generating_user:
            self.stop_generating = True


    def on_dccmsg(self, c, e):
        pass

    def on_dccchat(self, c, e):
        pass

    def do_command(self, e, cmd):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", help="the name")


    parser.add_argument("--server", help="the model")
    parser.add_argument("--channel", help="the model")

    return parser.parse_args()


def main():
    args = parse_arguments()
    bot = IrcBot(args.channel, args.name, args.server)

    bot.start()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
