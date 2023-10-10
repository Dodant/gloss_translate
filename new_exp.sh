#!/bin/bash

# gloss2text
common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|gpt2|0s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=64
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|gpt2|1s" --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv' --max-len=48 --batch-size=48
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|gpt2|2s" --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv' --max-len=72 --batch-size=32
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|gpt2|3s" --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv' --max-len=96 --batch-size=16

common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=30"
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|tri_|0s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=24 --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|tri_|1s" --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv' --max-len=48 --batch-size=12 --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|tri_|2s" --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv' --max-len=72 --batch-size=8  --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="g2t|tri_|3s" --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv' --max-len=96 --batch-size=6  --model='skt/ko-gpt-trinity-1.2B-v0.5'


# text2gloss
common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|gpt2|0s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=48 --direction='t2g'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|gpt2|1s" --train_dataset='MY_DATA/shots/text2gloss/1-shot.csv' --max-len=48 --batch-size=36 --direction='t2g'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|gpt2|2s" --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv' --max-len=72 --batch-size=24 --direction='t2g'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|gpt2|3s" --train_dataset='MY_DATA/shots/text2gloss/3-shot.csv' --max-len=96 --batch-size=12 --direction='t2g'

common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=30"
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|tri_|0s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=24 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|tri_|1s" --train_dataset='MY_DATA/shots/text2gloss/1-shot.csv' --max-len=48 --batch-size=12 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|tri_|2s" --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv' --max-len=72 --batch-size=8  --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="t2g|tri_|3s" --train_dataset='MY_DATA/shots/text2gloss/3-shot.csv' --max-len=96 --batch-size=6  --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'