common_options="python train_g2t2g_aug.py --gpus=1 --max-len=36 --batch-size=64 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|BL|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=72 --batch-size=48 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|1s|MY' --train_dataset='MY_DATA/shots/gloss2text/1-shot.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=108 --batch-size=32 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|2s|MY' --train_dataset='MY_DATA/shots/gloss2text/2-shot.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=144 --batch-size=16 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|3s|MY' --train_dataset='MY_DATA/shots/gloss2text/3-shot.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=180 --batch-size=8 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|4s|MY' --train_dataset='MY_DATA/shots/gloss2text/4-shot.csv'

common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6"
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|BL|MY.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|1s|MY.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|2s|MY.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|3s|MY.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|4s|MY.ckpt' --bk

common_options="python train_g2t2g_aug.py --gpus=1 --max-len=36 --batch-size=64 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|BL|3K' --train_dataset='GKSL/GKSL3k_train.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=72 --batch-size=48 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|1s|3K' --train_dataset='GKSL/GKSL3k_train-1.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=108 --batch-size=32 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|2s|3K' --train_dataset='GKSL/GKSL3k_train-2.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=144 --batch-size=16 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|3s|3K' --train_dataset='GKSL/GKSL3k_train-3.csv'
common_options="python train_g2t2g_aug.py --gpus=1 --max-len=180 --batch-size=8 --train --max_epochs=50"
CUDA_VISIBLE_DEVICES=0 $common_options --tag='NML|gpt2|4s|3K' --train_dataset='GKSL/GKSL3k_train-4.csv'

common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6"
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|BL|3K.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|1s|3K.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|2s|3K.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|3s|3K.ckpt' --bk
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|4s|3K.ckpt' --bk

#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=72 --batch-size=48 --train --max_epochs=50 --max_shots=1"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|1s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|1s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|1s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=108 --batch-size=32 --train --max_epochs=50 --max_shots=2"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|2s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|2s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|2s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=144 --batch-size=16 --train --max_epochs=50 --max_shots=3"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|3s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|3s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|3s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=180 --batch-size=8 --train --max_epochs=50 --max_shots=4"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|4s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|4s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|4s|MY' --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|1s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|1s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|1s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|2s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|2s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|2s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|3s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|3s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|3s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|4s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|4s|MY.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|4s|MY.ckpt' --bk

#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=72 --batch-size=48 --train --max_epochs=50 --max_shots=1"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|1s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|1s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|1s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=108 --batch-size=32 --train --max_epochs=50 --max_shots=2"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|2s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|2s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|2s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=144 --batch-size=16 --train --max_epochs=50 --max_shots=3"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|3s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|3s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|3s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=180 --batch-size=8 --train --max_epochs=50 --max_shots=4"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|gpt2|4s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|gpt2|4s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|gpt2|4s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug
#
#common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|1s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|1s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|1s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|2s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|2s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|2s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|3s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|3s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|3s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|4s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|4s|13K.ckpt' --bk
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|4s|13K.ckpt' --bk

#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=72 --batch-size=10 --train --max_epochs=50 --max_shots=1"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|TRI_|1s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|TRI_|1s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|TRI_|1s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=108 --batch-size=8 --train --max_epochs=50 --max_shots=2"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|TRI_|2s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|TRI_|2s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|TRI_|2s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=144 --batch-size=6 --train --max_epochs=50 --max_shots=3"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|TRI_|3s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|TRI_|3s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|TRI_|3s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --max-len=180 --batch-size=4 --train --max_epochs=50 --max_shots=4"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG|TRI_|4s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --aug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='DYN|TRI_|4s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --dynoaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='TOT|TRI_|4s|13K' --train_dataset='GKSL/GKSL13k_train.csv' --totalaug --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#common_options="python train_g2t2g_aug.py --gpus=1 --test --max-len=24 --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|TRI_|1s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|TRI_|1s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|TRI_|1s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|TRI_|2s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|TRI_|2s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|TRI_|2s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|TRI_|3s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|TRI_|3s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|TRI_|3s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|TRI_|4s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|TRI_|4s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|TRI_|4s|13K.ckpt' --bk --model='skt/ko-gpt-trinity-1.2B-v0.5'








#common_options="python train_g2t2g.py --gpus=1"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="NRM-g2t|gpt2|4s" --train_dataset='MY_DATA/shots/gloss2text/4-shot.csv' --max-len=120 --batch-size=8 --train --max_epochs=50 --train
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM-g2t|gpt2|4s-bk' --model_params='model_chp/NRM-g2t|gpt2|4s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=120 --test
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='NRM-g2t|gpt2|4s-3k' --model_params='model_chp/NRM-g2t|gpt2|4s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=120 --test
#common_options="python train_g2t2g_aug.py --gpus=1"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-g2t|gpt2|4s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=120 --batch-size=8 --train --max_epochs=50 --train
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|4s-bk' --model_params='model_chp/AUG-g2t|gpt2|4s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=120 --test
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|4s-3k' --model_params='model_chp/AUG-g2t|gpt2|4s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=120 --test





# gloss2text
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-g2t|gpt2|1s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=48 --batch-size=48
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-g2t|gpt2|2s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=72 --batch-size=32
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-g2t|gpt2|3s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=96 --batch-size=16

# gloss2text
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=30"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="REAUG-g2t|tri_|1s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=48 --batch-size=12 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="REAUG-g2t|tri_|2s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=72 --batch-size=8  --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="REAUG-g2t|tri_|3s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=96 --batch-size=6  --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
# text2gloss
#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=50"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|gpt2|1s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=48 --batch-size=48 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|gpt2|2s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=72 --batch-size=32 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|gpt2|3s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=96 --batch-size=16 --direction='t2g'

#common_options="python train_g2t2g_aug.py --gpus=1 --train --max_epochs=30"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|tri_|1s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=48 --batch-size=12 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|tri_|2s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=72 --batch-size=8  --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|tri_|3s" --train_dataset='MY_DATA/shots/gloss2text/0-shot.csv' --max-len=96 --batch-size=6  --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'

# test
#common_options="python train_g2t2g.py --gpus=1 --test --batch-size=4"
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='REAUG-g2t|tri_|1s-3k' --model_params='model_chp/REAUG-g2t|tri_|1s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='REAUG-g2t|tri_|2s-3k' --model_params='model_chp/REAUG-g2t|tri_|2s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##CUDA_VISIBLE_DEVICES=0 $common_options --tag='REAUG-g2t|tri_|3s-3k' --model_params='model_chp/REAUG-g2t|tri_|3s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5'
##
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='REAUG-g2t|tri_|1s-bk' --model_params='model_chp/REAUG-g2t|tri_|1s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='REAUG-g2t|tri_|2s-bk' --model_params='model_chp/REAUG-g2t|tri_|2s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='REAUG-g2t|tri_|3s-bk' --model_params='model_chp/REAUG-g2t|tri_|3s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5'

#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|gpt2|1s-3k" --model_params='model_chp/AUG-t2g|gpt2|1s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=48 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|gpt2|2s-3k' --model_params='model_chp/AUG-t2g|gpt2|2s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=72 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|gpt2|3s-3k' --model_params='model_chp/AUG-t2g|gpt2|3s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=96 --direction='t2g'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|gpt2|1s-bk" --model_params='model_chp/AUG-t2g|gpt2|1s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=48 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|gpt2|2s-bk' --model_params='model_chp/AUG-t2g|gpt2|2s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=72 --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|gpt2|3s-bk' --model_params='model_chp/AUG-t2g|gpt2|3s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=96 --direction='t2g'

#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|tri_|1s-3k" --model_params='model_chp/AUG-t2g|tri_|1s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|tri_|2s-3k' --model_params='model_chp/AUG-t2g|tri_|2s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|tri_|3s-3k' --model_params='model_chp/AUG-t2g|tri_|3s.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --tag="AUG-t2g|tri_|1s-bk" --model_params='model_chp/AUG-t2g|tri_|1s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|tri_|2s-bk' --model_params='model_chp/AUG-t2g|tri_|2s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-t2g|tri_|3s-bk' --model_params='model_chp/AUG-t2g|tri_|3s.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5' --direction='t2g'

#
#common_options="python train_g2t2g_aug.py --gpus=1 --test --batch-size=6"
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|gpt2|1s---epoch=15-train_loss=14.20.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=48
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|gpt2|2s---epoch=15-train_loss=15.69.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=72
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|gpt2|3s---epoch=00-train_loss=16.88.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=96
#
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|tri_|1s---epoch=03-train_loss=11.62.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|tri_|2s---epoch=02-train_loss=12.54.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|tri_|3s---epoch=19-train_loss=12.70.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|gpt2|1s---epoch=15-train_loss=14.20.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=48
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|gpt2|2s---epoch=15-train_loss=15.69.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=72
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|gpt2|3s---epoch=00-train_loss=16.88.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=96
#
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|tri_|1s---epoch=03-train_loss=11.62.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|tri_|2s---epoch=02-train_loss=12.54.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --direction-'t2g' --model_params='model_chp/AUG-t2g|tri_|3s---epoch=19-train_loss=12.70.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5'


#common_options="python train_g2t2g.py --gpus=1 --test --batch-size=4"
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|1s-3k' --model_params='model_chp/AUG-g2t|gpt2|1s---epoch=15-train_loss=12.23.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=48
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|2s-3k' --model_params='model_chp/AUG-g2t|gpt2|2s---epoch=05-train_loss=15.61.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=72
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|3s-3k' --model_params='model_chp/AUG-g2t|gpt2|3s---epoch=04-train_loss=17.14.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=96
#
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|tri_|1s-3k' --model_params='model_chp/AUG-g2t|tri_|1s---epoch=08-train_loss=2.19.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|tri_|2s-3k' --model_params='model_chp/AUG-g2t|tri_|2s---epoch=15-train_loss=1.29.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|tri_|3s-3k' --model_params='model_chp/AUG-g2t|tri_|3s---epoch=06-train_loss=0.97.ckpt' --test_dataset='GKSL/GKSL3k(final)_mix.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|1s-bk' --model_params='model_chp/AUG-g2t|gpt2|1s---epoch=15-train_loss=12.23.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=48
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|2s-bk' --model_params='model_chp/AUG-g2t|gpt2|2s---epoch=05-train_loss=15.61.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=72
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|gpt2|3s-bk' --model_params='model_chp/AUG-g2t|gpt2|3s---epoch=04-train_loss=17.14.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=96
#
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|tri_|1s-bk' --model_params='model_chp/AUG-g2t|tri_|1s---epoch=08-train_loss=2.19.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=48 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|tri_|2s-bk' --model_params='model_chp/AUG-g2t|tri_|2s---epoch=15-train_loss=1.29.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=72 --model='skt/ko-gpt-trinity-1.2B-v0.5'
#CUDA_VISIBLE_DEVICES=0 $common_options --tag='AUG-g2t|tri_|3s-bk' --model_params='model_chp/AUG-g2t|tri_|3s---epoch=06-train_loss=0.97.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --max-len=96 --model='skt/ko-gpt-trinity-1.2B-v0.5'