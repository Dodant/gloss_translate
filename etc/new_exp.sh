#!/bin/bash

common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=8 --t2g"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='REREAUG|TRI_|T2G|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=2 --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'

common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/REREAUG|TRI_|T2G|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/REREAUG|TRI_|T2G|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'





#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=48"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|GPT2|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=1 --totalaug
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=32"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|GPT2|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=2 --totalaug
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=16"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|GPT2|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=3 --totalaug
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=8"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|GPT2|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=4 --totalaug

#common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|1s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|2s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|3s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|4s.ckpt' --k3
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|1s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|2s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|3s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|4s.ckpt' --bk
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|1s.ckpt' --k3 --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|2s.ckpt' --k3 --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|3s.ckpt' --k3 --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|4s.ckpt' --k3 --t2g
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|1s.ckpt' --bk --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|2s.ckpt' --bk --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|3s.ckpt' --bk --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|GPT2|4s.ckpt' --bk --t2g


#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=10"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=1 --totalaug
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=8"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=2 --totalaug
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|TRI_|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=3 --totalaug
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=4"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOTAUG|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max_shots=4 --totalaug

#common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|1s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|2s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|3s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|4s.ckpt' --k3
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|1s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|2s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|3s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|4s.ckpt' --bk
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|1s.ckpt' --k3 --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|2s.ckpt' --k3 --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|3s.ckpt' --k3 --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|4s.ckpt' --k3 --t2g
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|1s.ckpt' --bk --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|2s.ckpt' --bk --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|3s.ckpt' --bk --t2g
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOTAUG|TRI_|4s.ckpt' --bk --t2g

# gpt2
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=48"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|GPT2|1s' --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|GPT2|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|1s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|1s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|1s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|1s.ckpt' --k3
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=32"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|GPT2|2s' --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|GPT2|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|2s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|2s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|2s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|2s.ckpt' --k3
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=16"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|GPT2|3s' --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|GPT2|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|3s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|3s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|3s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|3s.ckpt' --k3
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=8"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|GPT2|4s' --train_dataset='MY_DATA/shots/gloss2text/4-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|GPT2|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|4s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|GPT2|4s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|4s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|GPT2|4s.ckpt' --k3

#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=48 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=1
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=32 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=2
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=16 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=3
##common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=8 --t2g"
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=4
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|1s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|2s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|3s.ckpt' --k3
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|4s.ckpt' --k3
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|1s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|2s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|3s.ckpt' --bk
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2|4s.ckpt' --bk

#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=10"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|G2T|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=1 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=8"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|G2T|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=2 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|G2T|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|G2T|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|G2T|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|G2T|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'



# tri_
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=10"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=8"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|3s' --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|3s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|3s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|3s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|3s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=4"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/4-shot.csv'  --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv'  --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|4s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|4s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|4s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|4s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'

#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=10 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=1 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=8 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=2 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|TRI_|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=4 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=4 --model='skt/ko-gpt-trinity-1.2B-v0.5'

#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_|3s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|G2T|TRI_|4s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'

#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_|3s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_|4s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'

# t2g
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=16 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=3
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=3
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=3
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|GPT2' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=3
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|GPT2.ckpt' --bk
#
### tri_
##common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=10"
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
##common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
##common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=8"
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
##common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
##common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=6"
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|3s' --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
##common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|3s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|3s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|3s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|3s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
##common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=4"
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|G2T|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/4-shot.csv'  --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|G2T|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv'  --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
##common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6"
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|4s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|G2T|TRI_|4s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|4s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|G2T|TRI_|4s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYAUG|T2G|TRI_' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug --max_shots=3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYAUG|T2G|TRI_.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'

## text2gloss
# gpt2
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=48 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|GPT2|1s' --train_dataset='MY_DATA/shots/text2gloss/1-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|GPT2|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|1s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|1s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|1s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|1s.ckpt' --k3
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=32 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|GPT2|2s' --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|GPT2|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|2s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|2s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|2s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|2s.ckpt' --k3
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=16 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|GPT2|3s' --train_dataset='MY_DATA/shots/text2gloss/3-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|GPT2|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|3s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|3s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|3s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|3s.ckpt' --k3

#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=8 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|GPT2|4s' --train_dataset='MY_DATA/shots/gloss2text/4-shot.csv'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|GPT2|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|4s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|GPT2|4s.ckpt' --k3
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|4s.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|GPT2|4s.ckpt' --k3


# tri_
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=48 --batch-size=10 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|TRI_|1s' --train_dataset='MY_DATA/shots/text2gloss/1-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|TRI_|1s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|1s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|1s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=72 --batch-size=8 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|TRI_|2s' --train_dataset='MY_DATA/shots/text2gloss/2-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|TRI_|2s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|2s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|2s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=96 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|TRI_|3s' --train_dataset='MY_DATA/shots/text2gloss/3-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|TRI_|3s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|3s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|3s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|3s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|3s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'

#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50 --max-len=120 --batch-size=4 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM|T2G|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/4-shot.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|T2G|TRI_|4s' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g.py --gpus=1 --test --max-len=24 --batch-size=6 --t2g"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|4s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NRM|T2G|TRI_|4s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|4s.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|T2G|TRI_|4s.ckpt' --k3 --model='skt/ko-gpt-trinity-1.2B-v0.5'