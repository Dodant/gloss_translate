from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
    BartForConditionalGeneration, BartTokenizer, pipeline
)

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import mecab_ko as Mecab
from nltk.translate.bleu_score import sentence_bleu

model_name = "gogamza/kobart-base-v2"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
style_map = {'gloss': '글로스', 'spoken': '구어체'}


class TextStyleTransferDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        text2 = row['gloss']
        text1 = row['spoken']
        # target_style = row.index[1]
        # target_style_name = style_map[target_style]

        encoder_text = f"글로스 말투로 변환: {text1}"
        decoder_text = f"{text2}{self.tokenizer.eos_token}"
        model_inputs = self.tokenizer(encoder_text, max_length=64, truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = tokenizer(decoder_text, max_length=64, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        del model_inputs['token_type_ids']

        return model_inputs


def generate_text(pipe, text, num_return_sequences=5, max_length=60):
    # target_style_name = style_map[target_style]
    text = f"글로스 말투로 변환: {text}"
    out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
    return [x['generated_text'] for x in out]


def func1(model_pth, train_dataset, test_dataset):
    gksl3k = 'GKSL/GKSL3k.csv'
    gksl13k = 'GKSL/GKSL13k.csv'
    df1 = pd.read_csv(gksl3k)
    df2 = pd.read_csv(gksl13k)
    gloss_combined = pd.concat([df1['gloss'], df2['gloss']], ignore_index=True)
    spoken_combined = pd.concat([df1['spoken'], df2['spoken']], ignore_index=True)

    df = pd.DataFrame({
        'gloss': gloss_combined,
        'spoken': spoken_combined
    })

    dataset = TextStyleTransferDataset(df, tokenizer)
    df_train, df_test = train_test_split(dataset, test_size=0.01, random_state=42)
    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_pth,  # The output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=30,  # number of training epochs
        per_device_train_batch_size=256,  # batch size for training
        per_device_eval_batch_size=256,  # batch size for evaluation
        eval_steps=500,  # Number of update steps between two evaluations.
        save_steps=1000,  # after # steps model is saved
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        prediction_loss_only=True,
        evaluation_strategy="steps",
        save_total_limit=3
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=df_train,
        eval_dataset=df_test,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(model_pth)

    nlg = pipeline('text2text-generation', model=model_pth, tokenizer=tokenizer)

    # mecab = Mecab.Tagger('-Owakati')

    df = pd.read_csv(test_dataset)
    SCORE1, SCORE2, SCORE3, SCORE4, SCORET = [], [], [], [], []

    for i in range(len(df)):
        gt_text = df.iloc[i]['gloss']
        input_text = df.iloc[i]['spoken']
        output_text = generate_text(nlg, input_text, num_return_sequences=1, max_length=1000)[0]
        # gtext = tokenizer(gt_text, max_length=64, truncation=True)['input_ids']
        # outputtext = tokenizer(output_text, max_length=64, truncation=True)['input_ids']
        # gtext = mecab.parse(gt_text).split()
        # outputtext = mecab.parse(output_text).split()
        gtext = gt_text.split()
        outputtext = output_text.split()
        score1 = sentence_bleu([gtext], outputtext, weights=(1, 0, 0, 0))
        score2 = sentence_bleu([gtext], outputtext, weights=(0, 1, 0, 0))
        score3 = sentence_bleu([gtext], outputtext, weights=(0, 0, 1, 0))
        score4 = sentence_bleu([gtext], outputtext, weights=(0, 0, 0, 1))
        scoret = sentence_bleu([gtext], outputtext)
        print(f'{i}/{len(df)} {input_text} -> {output_text}')
        print(gt_text)
        print(f"BLEU score: {score1:.3f} {score2:.3f} {score3:.3f} {score4:.3f} {scoret:.3f}")
        SCORE1.append(score1)
        SCORE2.append(score2)
        SCORE3.append(score3)
        SCORE4.append(score4)
        SCORET.append(scoret)
    BLEU1, BLEU2, BLEU3, BLEU4, BLEUT = np.mean(SCORE1), np.mean(SCORE2), np.mean(SCORE3), np.mean(SCORE4), np.mean(SCORET)
    print(f"{model_pth}  BLEU score: {BLEU1:.3f} {BLEU2:.3f} {BLEU3:.3f} {BLEU4:.3f} {BLEUT:.3f}")
    with open('etc/bleu_score.txt', 'a') as f:
        f.write(f"{model_pth}  BLEU score: {BLEU1:.3f} {BLEU2:.3f} {BLEU3:.3f} {BLEU4:.3f} {BLEUT:.3f}\n")


if __name__ == "__main__":
    # gksl3k = 'GKSL/GKSL3k.csv'
    # mydata = 'MY_DATA/shots/gloss2text/0-shot.csv'
    # gksl13k = 'GKSL/GKSL13k.csv'
    # test_g = 'MY_DATA/gloss_from_book.csv'
    # func1('model/k3tg', gksl3k, test_g)
    # func1('model/k13tg', gksl13k, test_g)
    # func1('model/mytg', mydata, test_g)
    # glsk_13k_br = 'GKSL/GKSL13k_BR.csv'
    # glsk_13k_sp = 'GKSL/GKSL13k_SP.csv'
    # glsk_13k_sr = 'GKSL/GKSL13k_SR.csv'
    test_g = 'MY_DATA/gloss_from_book.csv'
    func1('model/13k3ktg', 'd', test_g)
    # func1('model/13ksp', glsk_13k_sp, test_g)
    # func1('model/13ksr', glsk_13k_sr, test_g)
