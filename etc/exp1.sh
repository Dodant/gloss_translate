#!/bin/bash
## gloss2text

common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=30"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="0s|g2t|gpt2" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=64
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="1s|g2t|gpt2" --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv' --max-len=48 --batch-size=48
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="2s|g2t|gpt2" --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv' --max-len=72 --batch-size=32
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="3s|g2t|gpt2" --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv' --max-len=96 --batch-size=16

#edit
CUDA_VISIBLE_DEVICES=0 $common_options --tag="2s|g2t|gpt2" --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv' --max-len=96 --batch-size=16

#common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=15"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="0s|g2t|tri" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=12 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="1s|g2t|tri" --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv' --max-len=48 --batch-size=10 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="2s|g2t|tri" --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv' --max-len=72 --batch-size=8 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="3s|g2t|tri" --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv' --max-len=96 --batch-size=6 --model='skt/ko-gpt-trinity-1.2B-v0.5'

## text2gloss
# 0-shot
common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=30"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="0s|t2g|gpt2" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=64 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="1s|t2g|gpt2" --train_dataset='MY_DATA/shots/text2gloss/1-shot.csv' --max-len=48 --batch-size=48 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="2s|t2g|gpt2" --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv' --max-len=72 --batch-size=32 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="3s|t2g|gpt2" --train_dataset='MY_DATA/shots/text2gloss/3-shot.csv' --max-len=96 --batch-size=16 --direction='t2g'

#edit
CUDA_VISIBLE_DEVICES=0 $common_options --tag="1s|t2g|gpt2" --train_dataset='MY_DATA/shots/text2gloss/1-shot.csv' --max-len=72 --batch-size=24 --direction='t2g'
CUDA_VISIBLE_DEVICES=0 $common_options --tag="2s|t2g|gpt2" --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv' --max-len=96 --batch-size=16 --direction='t2g'

common_options="python train_g2t2g.py --gpus=1 --train --max_epochs=15"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="0s|t2g|tri" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=24 --batch-size=12 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="1s|t2g|tri" --train_dataset='MY_DATA/shots/text2gloss/1-shot.csv' --max-len=48 --batch-size=10 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="2s|t2g|tri" --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv' --max-len=72 --batch-size=8 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="3s|t2g|tri" --train_dataset='MY_DATA/shots/text2gloss/3-shot.csv' --max-len=96 --batch-size=6 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'

#edit
CUDA_VISIBLE_DEVICES=0 $common_options --tag="2s|t2g|tri" --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv' --max-len=96 --batch-size=6 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
