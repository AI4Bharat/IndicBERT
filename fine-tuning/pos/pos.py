import argparse

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)

import wandb

wandb.init(project="indicxtreme")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--train_data", type=str, default="udpos.English")
parser.add_argument("--eval_data", type=str, default="udpos.Hindi")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--metric_for_best_model", type=str, default="eval_f1")
parser.add_argument("--label_all_tokens", type=bool, default=True)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--num_train_epochs", type=int, default=10)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation='longest_first', is_split_into_words=True, max_length=args.max_seq_length)
    
    labels = []
    for i, label in enumerate(examples["pos_tags"]):
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

metric = load_metric("seqeval")
dataset = load_dataset("xtreme", f"{args.train_data}")

label_list = dataset["train"].features["pos_tags"].feature.names

tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)
model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list), use_auth_token=True)
data_collator = DataCollatorForTokenClassification(tokenizer)

if args.do_train:

    train_dataset = dataset["train"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    print(f"Length of Training Dataset: {len(train_dataset)}")

    validation_dataset = dataset["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["validation"].column_names,
    )
    print(f"Lenght of Validation Dataset: {len(validation_dataset)}")

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.model_name}",
        save_total_limit=5,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
    )

    # earlystoppingcallback = EarlyStoppingCallback(early_stopping_patience=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[earlystoppingcallback]
    )
    trainer.train()

    trainer.evaluate()

elif args.do_predict:
    def zero_shot(dataset):
        test_dataset = dataset['test'].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset['test'].column_names,
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
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    dataset = load_dataset("xtreme", f"{args.eval_data}")
    zero_shot(dataset)