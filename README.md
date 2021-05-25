# CLIP Assisted Tamed Transformer (CATT)

I added a small script in the `scripts` subfolder to generate image from text, using clip from OPENAI

`python clip_generator.py --config ../models/imagenet/vqgan_imagenet_f16_16384.yaml --checkpoint ../models/imagenet/vqgan_imagenet_f16_16384.ckpt --text 'A cute fox wearing a tuxedo' --outdir ./out/fox`
