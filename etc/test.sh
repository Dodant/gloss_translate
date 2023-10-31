#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|m-b|gpt2|||epoch=03-train_loss=41.99.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|m-3|gpt2|||epoch=11-train_loss=40.90.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|13-b|gpt2|||epoch=02-train_loss=41.41.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|13-3|gpt2|||epoch=01-train_loss=40.96.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv'

CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|m-b|tri|||epoch=03-train_loss=44.87.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|m-3|tri|||epoch=00-train_loss=47.99.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|13-b|tri|||epoch=04-train_loss=43.40.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/g2t|13-3|tri|||epoch=04-train_loss=47.13.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'

CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|m-3|gpt2|||epoch=04-train_loss=41.63.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|m-b|gpt2|||epoch=03-train_loss=42.16.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|13-b|gpt2|||epoch=04-train_loss=45.31.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|13-3|gpt2|||epoch=08-train_loss=44.07.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv'

CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|m-b|tri|||epoch=02-train_loss=42.19.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|m-3|tri|||epoch=02-train_loss=43.37.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|13-b|tri|||epoch=04-train_loss=41.36.ckpt' --test_dataset='MY_DATA/gloss_from_book.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'
CUDA_VISIBLE_DEVICES=0 python train_g2t2g.py --gpus 1 --test --model_params='model_chp/t2g|13-3|tri|||epoch=04-train_loss=42.78.ckpt' --test_dataset='GKSL/GKSL3k(final)_.csv' --model='skt/ko-gpt-trinity-1.2B-v0.5'








