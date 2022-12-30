import argparse

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)

import wandb

from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import indic_tokenize
from indicnlp.transliterate import unicode_transliterate

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--train_data", type=str, default="PAN-X.en")
parser.add_argument("--in_lang", action="store_true")
parser.add_argument("--eval_data", type=str, default="hi")
parser.add_argument("--benchmark", type=str, default="naamapadam")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--metric_for_best_model", type=str, default="eval_f1")
parser.add_argument("--label_all_tokens", type=bool, default=True)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--num_train_epochs", type=int, default=10)
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
    if args.in_lang:
        wandb.run.name = str(args.model_name) + '-in-lang-' + str(args.train_data) + '-' + 'lr-'+str(config.lr) + '-wd-' + str(config.weight_decay)
    else:
        wandb.run.name = str(args.model_name) + '-' + str(args.train_data) + '-' + 'lr-'+str(config.lr) + '-wd-' + str(config.weight_decay)
    wandb.run.save()

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation='longest_first', is_split_into_words=True, max_length=args.max_seq_length)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    if args.in_lang:
        dataset = load_dataset('ai4bharat/naamapadam', f'hi')
        dataset = dataset.shuffle(seed=args.seed)
        k = dataset['train'].train_test_split(train_size=20000)
        dataset['train'] = k['train']
    else:
        dataset =  load_dataset("xtreme", f"{args.train_data}")

    label_list = dataset["train"].features["ner_tags"].feature.names

    metric = load_metric("seqeval")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", use_auth_token=True)
    model = AutoModelForTokenClassification.from_pretrained(f"{args.model_name}", use_auth_token=True, num_labels=len(label_list))
    data_collator = DataCollatorForTokenClassification(tokenizer)

    if args.do_train:
        
        train_dataset = dataset['train'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset['train'].column_names,
        )
        print(f"Length of Training dataset: {len(train_dataset)}")

        validation_dataset = dataset['validation'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset['validation'].column_names,
        )
        print(f"Length of Validation dataset: {len(validation_dataset)}")

        if args.in_lang:
            name = f"{args.output_dir}/in-lang-{args.model_name}-{str(config.lr)}-{str(config.weight_decay)}"
        else:
            name = f"{args.output_dir}/{args.model_name}-{str(config.lr)}-{str(config.weight_decay)}"

        training_args = TrainingArguments(
            output_dir=f"{name}",
            save_total_limit=10,
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
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        trainer.evaluate()

if args.do_train:
    sweep_config = {
        'method': 'grid',
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
        example['tokens'] = [preprocess_line(token, lang) for token in example['tokens']]
        return example

    if args.in_lang:
        dataset = load_dataset('ai4bharat/naamapadam', f'{args.in_lang}')
    else:
        dataset =  load_dataset("xtreme", f"{args.train_data}")

    label_list = dataset["train"].features["ner_tags"].feature.names
    metric = load_metric("seqeval")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", use_auth_token=True)
    model = AutoModelForTokenClassification.from_pretrained(f"{args.model_name}", use_auth_token=True, num_labels=len(label_list))
    data_collator = DataCollatorForTokenClassification(tokenizer)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation='longest_first', is_split_into_words=True, max_length=args.max_seq_length)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def zero_shot(dataset):

        val_dataset = dataset['validation'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset['validation'].column_names,
        )
        print(f"Length of Val dataset: {len(val_dataset)}")

        results = trainer.predict(val_dataset).metrics
        print(f"Results for {args.eval_data} dataset: {results}")

        test_dataset = dataset['test'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset['test'].column_names,
        )
        print(f"Length of Test dataset: {len(test_dataset)}")

        results = trainer.predict(test_dataset).metrics
        print(f"Results for {args.eval_data} dataset: {results}")


    training_args = TrainingArguments(
        output_dir=f"{args.model_name}",
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    if args.benchmark == "naamapadam":
        dataset = load_dataset(f"ai4bharat/naamapadam", f"{args.eval_data}", use_auth_token=True)
    else:
        bench, lang = args.eval_data.split(".")[0], args.eval_data.split(".")[1]
        dataset = load_dataset(bench, lang)

    if args.og_lang in supported_xlit_langs:
        dataset = dataset.map(lambda x: transliterate(x, args.og_lang))
    zero_shot(dataset)