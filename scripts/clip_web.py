import threading
import time

from flask import Flask, request, redirect, send_from_directory
import clip

from clip_generator.dreamer import load_vqgan_model
from clip_generator.trainer import Trainer

app = Flask(__name__)

clip = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to('cuda:0')
print('loading VQGAN')
vqgan_model = load_vqgan_model('../models/imagenet/vqgan_imagenet_f16_16384.yaml',
                                    '../models/imagenet/vqgan_imagenet_f16_16384.ckpt')


@app.route("/web_out/<path:path>")
def get_image(path):
    return send_from_directory('web_out', path)

@app.route("/")
def generate_image():
    prompt = request.args['prompt']
    print(prompt)
    trainer = Trainer([prompt],
                      vqgan_model,
                      clip,
                      learning_rate=0.05,
                      save_every=10,
                      outdir=f'./web_out/{prompt}_{time.time()}',
                      device='cuda:0',
                      image_size=(800, 800),
                      crazy_mode=False,
                      steps=500)
    generating_thread = threading.Thread(target=trainer.start)
    generating_thread.start()
    print(trainer.get_generated_image_path())
    return redirect(str(trainer.get_generated_image_path()))

app.run(host='0.0.0.0')
