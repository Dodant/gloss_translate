import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

U_TKN, S_TKN = '<usr>', '<sys>'
BOS, EOS = '</s>', '</s>'
MASK, SENT, UNK, PAD = '<mask>', '<sent>', '<unk>', '<pad>'

warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_ = 'skt/kogpt2-base-v2'
TOKENIZER = PreTrainedTokenizerFast.from_pretrained(MODEL_,
                                                    bos_token=BOS, eos_token=EOS, unk_token=UNK,
                                                    pad_token=PAD, mask_token=MASK)


class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first_log = True
        self.max_len = max_len

    def __len__(self):
        return len(self._data)

    def _tokenize_question(self, question, sentiment):
        return TOKENIZER.tokenize(U_TKN + question + SENT + sentiment)

    def _tokenize_answer(self, answer):
        return TOKENIZER.tokenize(S_TKN + answer + EOS)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        a_toked = self._tokenize_question(turn['gloss'], str(turn['label']))
        q_toked = self._tokenize_answer(turn['spoken'])

        # Adjust token lengths if combined length exceeds max_len
        if len(q_toked) + len(a_toked) > self.max_len:
            a_toked = self._adjust_token_length(q_toked, a_toked)

        # Generate labels and mask
        labels = [MASK] * len(q_toked) + a_toked[1:]
        mask = [0] * len(q_toked) + [1] * len(a_toked) + [0] * (self.max_len - len(q_toked) - len(a_toked))

        # Convert tokens to ids and pad to max length
        labels_ids = self._pad_to_max(TOKENIZER.convert_tokens_to_ids(labels))
        token_ids = self._pad_to_max(TOKENIZER.convert_tokens_to_ids(q_toked + a_toked))

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
        while len(token_ids) < self.max_len:
            token_ids.append(TOKENIZER.pad_token_id)
        return token_ids

    # def _log_sample(self, q_toked, a_toked, labels):
    #     """Log the first sample to provide insight into tokenization."""
    #     logging.info(f"contexts: {q_toked}")
    #     logging.info(f"response: {a_toked}")
    #     logging.info(f"labels: {labels}")


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.train_set = None
        self.save_hyperparameters(hparams)
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained(MODEL_)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len', type=int, default=32, help='max sentence length on input')
        parser.add_argument('--batch-size', type=int, default=64, help='batch size for training')
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
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
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
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = np.array([item[0] for item in batch])
        mask = np.array([item[1] for item in batch])
        label = np.array([item[2] for item in batch])
        return torch.from_numpy(data), torch.from_numpy(mask), torch.from_numpy(label)

    def train_dataloader(self):
        data1 = pd.read_csv('MY_DATA/mydata(edited).csv')
        data2 = pd.read_csv('MY_DATA/gloss_from_book.csv')
        data3 = pd.read_csv('GKSL/GKSL3k(labeled).csv')
        data = pd.concat([data1, data2, data3], ignore_index=True)
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        return DataLoader(self.train_set, batch_size=self.hparams.batch_size, num_workers=8,
                          shuffle=True, collate_fn=self._collate_fn, drop_last=True)

    def chat(self, sent='0'):

        with torch.no_grad():
            while True:
                q = input('gloss > ').strip()
                if q == 'quit':
                    break
                a = ''
                while True:
                    input_ids = torch.LongTensor(TOKENIZER.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(
                        dim=0)
                    pred = self(input_ids)
                    gen = TOKENIZER.convert_ids_to_tokens(
                        torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]

                    if gen == EOS:
                        break

                    a += gen.replace('▁', ' ')
                    print(f"processing ... {a.strip()}")
                print(f'translation > {a}')


def configure_parser():
    parser = argparse.ArgumentParser(description='Gloss Translator based on KoGPT-2')
    parser.add_argument('--chat', action='store_true', default=False, help='translation on given user input')
    parser.add_argument('--sentence', type=str, default='0',
                        help='0 is declarative, 1 is interrogative, 2 is exclamatory')
    parser.add_argument('--model_params', type=str, default='model_chp/model_last.ckpt',
                        help='model binary for translation')
    parser.add_argument('--train', action='store_true', default=False, help='training mode')
    parser = KoGPT2Chat.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    return parser


def train_model(args):
    checkpoint_callback = ModelCheckpoint(
        dirpath='model_chp',
        filename='{epoch:02d}-{train_loss:.2f}' + '-' + f'{args.tag}',
        verbose=True,
        save_last=True,
        monitor='train_loss',
        mode='min',
        # prefix='model_'
    )

    model = KoGPT2Chat(args)
    trainer = Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], gradient_clip_val=1.0)
    trainer.fit(model)
    logging.info(f'best model path {checkpoint_callback.best_model_path}')


def chat_model(model_params):
    model = KoGPT2Chat.load_from_checkpoint(model_params)
    model.chat()


if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    logging.info(args)

    if args.train: train_model(args)
    if args.chat: chat_model(args.model_params)
