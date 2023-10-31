
common_options="python train_g2t2g_aug.py --gpus=1 --bleu --k3"
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|4s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|3s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|2s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|1s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|BL|3K.ckpt'

CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|4s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|3s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|2s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|1s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/NML|gpt2|BL|MY.ckpt'

CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|4s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|4s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|4s|MY.ckpt'

CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|3s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|3s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|3s|MY.ckpt'

CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|2s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|2s|MY.ckpt'
CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|2s|MY.ckpt'

#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|1s|MY.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|1s|MY.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|1s|MY.ckpt'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|4s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|4s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|4s|3K.ckpt'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|3s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|3s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|3s|3K.ckpt'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|2s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|2s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|2s|3K.ckpt'
#
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/TOT|gpt2|1s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/DYN|gpt2|1s|3K.ckpt'
#CUDA_VISIBLE_DEVICES=0 $common_options --model_params='model_chp/AUG|gpt2|1s|3K.ckpt'