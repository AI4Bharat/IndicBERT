import argparse

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, EarlyStoppingCallback,
                          set_seed)

import wandb

from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import indic_tokenize
from indicnlp.transliterate import unicode_transliterate

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--train_data", type=str, default="paws-x")
parser.add_argument("--eval_data", type=str, default="hi")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--label_all_tokens", type=bool, default=True)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--og_lang", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

def train(config=None):
    wandb.init(config=config)
    config = wandb.config
    wandb.run.name = str(args.model_name) + '-' + str(args.train_data) + '-' + 'lr-'+str(config.lr) + '-wd-' + str(config.weight_decay)
    wandb.run.save()

    def preprocess_function(examples):
        return tokenizer(examples["sentence1"],
                        examples["sentence2"],
                        truncation=True,
                        padding="max_length",
                        max_length=512
                        )

    def x_preprocess_function(examples):
        return tokenizer(examples["english"],
                        examples["sentence2"],
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_seq_length
                        )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    dataset = load_dataset(f"{args.train_data}", "en")
    label_list = dataset["train"].features["label"].names
    metric = load_metric('glue', 'mnli')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_list), use_auth_token=True)

    if args.do_train:
        train_dataset = dataset["train"].map(
            preprocess_function,
            batched=True,
        )
        print(f"Length of Training Dataset: {len(train_dataset)}")

        validation_dataset = dataset["validation"].map(
            preprocess_function,
            batched=True,
        )
        print(f"Lenght of Validation Dataset: {len(validation_dataset)}")

        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/{args.model_name}-{str(config.lr)}-{str(config.weight_decay)}",
            save_total_limit=5,
            save_strategy="epoch",
            learning_rate=config.lr,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            do_eval=True,
            evaluation_strategy="epoch",
            weight_decay=config.weight_decay,
            fp16=args.fp16,
            warmup_ratio=args.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            optim='adamw_torch'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        trainer.evaluate()

if args.do_train:
    sweep_config = {
        'method': 'grid',
        # 'parameters': {
        #     'lr': {
        #         'values': [1e-5, 3e-5, 5e-6]
        #     },
        #     'weight_decay': {
        #         'values': [0.0, 0.01]
        #     }
        # }
        'parameters': {
            'lr': {
                'values': [3e-5]
            },
            'weight_decay': {
                'values': [0.01]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="varta", entity="dsuamnth17")
    wandb.agent(sweep_id, train)


elif args.do_predict:

    supported_xlit_langs = ['as', 'bn', 'gu', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

    def preprocess_line(line, lang):
        normfactory = indic_normalize.IndicNormalizerFactory()
        normalizer = normfactory.get_normalizer(lang)
        return unicode_transliterate.UnicodeIndicTransliterator.transliterate(
            " ".join(
                indic_tokenize.trivial_tokenize(
                    normalizer.normalize(line.strip()), lang
                )
            ),
            lang,
            "hi",
        )

    def transliterate(example, lang):
        example["sentence1"] = preprocess_line(example["sentence1"], lang)
        example["sentence2"] = preprocess_line(example["sentence2"], lang)
        return example

    dataset = load_dataset(f"{args.train_data}", "en")
    label_list = dataset["train"].features["label"].names
    metric = load_metric('glue', 'mnli')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(label_list), use_auth_token=True)
    
    def preprocess_function(examples):
        return tokenizer(examples["sentence1"],
                        examples["sentence2"],
                        truncation=True,
                        padding="max_length",
                        max_length=512
                        )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)


    def zero_shot(dataset):
        test_dataset = dataset['test'].map(
            preprocess_function,
            batched=True,
        )
        print(f"Length of Test Dataset: {len(test_dataset)}")

        results = trainer.predict(test_dataset).metrics
        print(f"Results for {args.eval_data} dataset: {results}")

    training_args = TrainingArguments(
        output_dir=args.model_name,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    dataset = load_dataset("ai4bharat/IndicXParaphrase", f"{args.eval_data}", use_auth_token=True)
    if args.og_lang in supported_xlit_langs:
        dataset = dataset.map(lambda x: transliterate(x, args.og_lang))
    zero_shot(dataset)