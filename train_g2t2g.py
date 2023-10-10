import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import GPT2LMHeadModel as GPT2
from transformers import GPT2TokenizerFast as GPT2T

U_TKN, S_TKN = '<usr>', '<sys>'
BOS, EOS = '</s>', '</s>'
MASK, SENT, UNK, PAD = '<mask>', '<sent>', '<unk>', '<pad>'

warnings.filterwarnings(action='ignore')

logger = logging.getLogger("torch.distributed.nn.jit.instantiator")
logger.setLevel(logging.WARNING)


class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first_log = True
        self.max_len = max_len
        self.TKNZR = GPT2T.from_pretrained(args.model, bos_token=BOS, eos_token=EOS,
                                           unk_token=UNK, pad_token=PAD, mask_token=MASK)

    def __len__(self):
        return len(self._data)

    def _tokenize_question(self, question):
        return self.TKNZR.tokenize(U_TKN + question + EOS)

    def _tokenize_answer(self, answer):
        return self.TKNZR.tokenize(S_TKN + answer + EOS)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        if args.direction == 'g2t':
            q_toked, a_toked = self._tokenize_question(turn['gloss']), self._tokenize_answer(turn['spoken'])
        else:
            q_toked, a_toked = self._tokenize_question(turn['spoken']), self._tokenize_answer(turn['gloss'])

        # Adjust token lengths if combined length exceeds max_len
        if len(q_toked) + len(a_toked) > self.max_len:
            a_toked = self._adjust_token_length(q_toked, a_toked)

        # Generate labels and mask
        labels = [MASK] * len(q_toked) + a_toked[1:]
        mask = [0] * len(q_toked) + [1] * (self.max_len - len(q_toked))
        # Convert tokens to ids and pad to max length
        labels_ids = self._pad_to_max(self.TKNZR.convert_tokens_to_ids(labels))
        token_ids = self._pad_to_max(self.TKNZR.convert_tokens_to_ids(q_toked + a_toked))
        # Log the first sample for debugging purposes
        # if self.first_log:
        #     self._log_sample(q_toked, a_toked, labels)
        #     self.first_log = False
        return token_ids, np.array(mask), labels_ids

    def _adjust_token_length(self, q_toked, a_toked):
        """Ensure total tokens do not exceed max_len."""
        a_len = self.max_len - len(q_toked)
        if a_len <= 0:
            q_toked = q_toked[-(int(self.max_len / 2)):]
            a_len = self.max_len - len(q_toked)
            assert a_len > 0, "Length of answer tokens is less than or equal to 0 after adjustment."
        return a_toked[:a_len]

    def _pad_to_max(self, token_ids):
        """Pad the token_ids to max length."""
        padding_length = self.max_len - len(token_ids)
        token_ids += [self.TKNZR.pad_token_id] * padding_length
        return token_ids

    # def _log_sample(self, q_toked, a_toked, labels):
    #     """Log the first sample to provide insight into tokenization."""
    #     logging.info(f"contexts: {q_toked}")
    #     logging.info(f"response: {a_toked}")
    #     logging.info(f"labels: {labels}")


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.save_hyperparameters(hparams)
        self.neg = -1e18
        self.kogpt2 = GPT2.from_pretrained(args.model)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        self.TKNZR = GPT2T.from_pretrained(args.model, bos_token=BOS, eos_token=EOS,
                                           unk_token=UNK, pad_token=PAD, mask_token=MASK)
        self.DS = args.train_dataset

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len', type=int, default=32, help='max sentence length on input')
        parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
        parser.add_argument('--lr', type=float, default=5e-5, help='The initial learning rate')
        parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
        parser.add_argument('--tag', type=str, default='0', help='tag for model checkpoint')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return {'loss': loss_avg}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        self.log('avg_train_loss', avg_loss)

    # def validation_step(self, batch, batch_idx):
    #     token_ids, mask, label = batch
    #     out = self(token_ids)
    #     mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
    #     mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
    #     loss = self.loss_function(mask_out.transpose(2, 1), label)
    #     loss_avg = loss.sum() / mask.sum()
    #     self.log('val_loss', loss_avg)
    #     return {"val_loss": loss_avg}
    #
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     self.log("avg_val_loss", avg_loss)
    #     return {"avg_val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('test_loss', loss_avg)
        return {"test_loss": loss_avg}

    def test_epoch_end(self, test_step_outputs):
        avg_loss = torch.stack([x["test_loss"] for x in test_step_outputs]).mean()
        self.log("avg_test_loss", avg_loss)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'cosine_schedule_with_warmup',
            'monitor': 'loss',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data, mask, label = map(np.array, [[item[i] for item in batch] for i in range(3)])
        return torch.from_numpy(data), torch.from_numpy(mask), torch.from_numpy(label)

    def train_dataloader(self):
        data1 = pd.read_csv(self.DS)
        data1 = data1[:int(len(data1) * 0.9)]
        train_set = CharDataset(data1, max_len=self.hparams.max_len)
        return DataLoader(train_set, batch_size=self.hparams.batch_size, num_workers=2,
                          shuffle=True, collate_fn=self._collate_fn, drop_last=True)

    # def val_dataloader(self):
    #     val_set = CharDataset(val_data, max_len=self.hparams.max_len)
    #     return DataLoader(val_set, batch_size=self.hparams.batch_size, num_workers=2,
    #                       shuffle=False, collate_fn=self._collate_fn)

    def test_dataloader(self):
        data4 = pd.read_csv(args.test_dataset)
        data4 = data4[int(len(data4) * 0.9):]
        test_set = CharDataset(data4, max_len=self.hparams.max_len)
        return DataLoader(test_set, batch_size=self.hparams.batch_size, num_workers=2, collate_fn=self._collate_fn)

    def chat(self):

        with torch.no_grad():
            if args.direction == 'g2t': q_str, a_str = 'gloss  >>> ', 'spoken >>>'
            else: q_str, a_str = 'spoken >>> ', 'gloss  >>>'

            while True:
                try:
                    q = input(q_str).strip()
                except UnicodeDecodeError:
                    print("UnicodeDecodeError\n")
                    continue
                if q == 'quit': break
                input_ids = torch.LongTensor(self.TKNZR.encode(U_TKN + q + S_TKN)).unsqueeze(dim=0)
                a = self.generate_with_repetition_penalty(input_ids, repetition_penalty=2.0,
                                                          max_length=self.hparams.max_len + 10)
                print(f"\r{a_str} {a.replace('▁', ' ').strip()}\n")
                a = ''
                while True:
                    input_ids = torch.LongTensor(self.TKNZR.encode(U_TKN + q + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    pred = torch.argmax(pred, dim=-1).squeeze().numpy().tolist()
                    gen = self.TKNZR.convert_ids_to_tokens(pred)[-1]
                    if gen == EOS: break
                    if len(a) > self.hparams.max_len + 10: break
                    a += gen
                    print(f"processing ... {a}")
                print(f"\r{a_str} {a.replace('▁', ' ').strip()}\n")

def configure_parser():
    parser = argparse.ArgumentParser(description='Gloss Translator based on KoGPT-2')

    parser.add_argument('--train_dataset', type=str, default='MY_DATA/mydata(edited).csv')
    parser.add_argument('--test_dataset', type=str, default='MY_DATA/gloss_from_book.csv')

    parser.add_argument('--direction', type=str, default='g2t', help='g2t or t2g')
    parser.add_argument('--model', type=str, default='skt/kogpt2-base-v2', help='model name')
    parser.add_argument('--model_params', type=str, default='model_chp/last.ckpt', help='model binary')

    parser.add_argument('--train', action='store_true', default=False, help='training mode')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--chat', action='store_true', default=False, help='translation on given user input')

    parser = KoGPT2Chat.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser


def train_model(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_chp',
        filename=f'{args.tag}|||' + '{epoch:02d}-{train_loss:.2f}',
        verbose=True,
        save_last=True,
        monitor='train_loss',
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min'
    )

    model = KoGPT2Chat(args)
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, early_stop_callback], gradient_clip_val=1.0)
    trainer.fit(model)
    trainer.test(model)
    logging.info(f'best model path {checkpoint_callback.best_model_path}')


def test_model(model_params):
    model = KoGPT2Chat.load_from_checkpoint(model_params)
    trainer = Trainer.from_argparse_args(args)
    trainer.test(model)


def chat_model(model_params):
    model = KoGPT2Chat.load_from_checkpoint(model_params)
    model.chat()


if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    logging.info(args)

    if args.train: train_model(args)
    if args.test: test_model(args.model_params)
    if args.chat: chat_model(args.model_params)
