import argparse

import numpy as np
from datasets import load_dataset, load_metric
from transformers import (AutoModelForMultipleChoice, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

import wandb

wandb.init(project="indicxtreme")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--train_data", type=str, default="social_i_qa")
parser.add_argument("--eval_data", type=str, default="translation-hi")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--task_name", type=str, default="siqa")
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--metric_for_best_model", type=str, default="eval_f1")
parser.add_argument("--label_all_tokens", type=bool, default=True)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--num_train_epochs", type=int, default=5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def preprocess_function(examples):

    if args.task_name == "siqa":
        ending_names = [f"answer{i}" for i in "ABC"]
        context_name = "context"
        question_header_name = "question"
    elif args.task_name == "xcopa":
        ending_names = [f"choice{i}" for i in "12"]
        context_name = "premise"
        question_header_name = "question"

    first_sentences = [
        [context] * len(ending_names) for context in examples[context_name]
    ]
    question_headers = examples[question_header_name]
    if args.task_name == "xcopa" or args.task_name == "siqa":
        second_sentences = [
            [f"{header} {examples[end][i]}" for end in ending_names]
            for i, header in enumerate(question_headers)
        ]
    else:
        # remove {header} as our dataset has not question headers
        second_sentences = [
            [f"{examples[end][i]}" for end in ending_names]
            for i, header in enumerate(question_headers)
        ]

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        # max_length=args.max_seq_length,
        # padding="max_length",
    )
    # Un-flatten
    return {
        k: [
            v[i : i + len(ending_names)]
            for i in range(0, len(v), len(ending_names))
        ]
        for k, v in tokenized_examples.items()
    }

# Metric
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    m = {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
    return {f"eval_{k}" if "eval_" not in k else k: v for k, v in m.items()}

dataset = load_dataset(f"{args.train_data}")
label_list = dataset['train'].unique("label")
label_list.sort()  # Let's sort it for determinism
label_to_id = {v: i for i, v in enumerate(label_list)}

tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", use_auth_token=True)
model = AutoModelForMultipleChoice.from_pretrained(f"{args.model_name}", use_auth_token=True)

if args.do_train:

    def convert_label_to_int(example):
        # for siqa
        example["label"] = int(example["label"]) - 1
        return example

    dataset['train'] = dataset['train'].map(convert_label_to_int)
    dataset['validation'] = dataset['validation'].map(convert_label_to_int)

    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        # remove_columns=dataset["train"].column_names,
    )
    print(f"Length of Training dataset: {len(train_dataset)}")

    validation_dataset = dataset['validation'].map(
        preprocess_function,
        batched=True,
        # remove_columns=dataset['validation'].column_names,
    )
    print(f"Length of Validation dataset: {len(validation_dataset)}")

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
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
        # callbacks=[earlystoppingcallback]
    )
    trainer.train()

    trainer.evaluate()


if args.do_predict:

    def zero_shot(dataset):
        test_dataset = dataset['test'].map(
            preprocess_function,
            batched=True,
            # remove_columns=dataset['test'].column_names,
        )
        print(f"Length of Test dataset: {len(test_dataset)}")

        results = trainer.predict(test_dataset)
        print(f"Results for {args.eval_data} dataset: {results.metrics}")

    training_args = TrainingArguments(
        output_dir=f"{args.model_name}",
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer),
        compute_metrics=compute_metrics,
    )

    dataset = load_dataset("ai4bharat/IndicXCOPA", f"{args.eval_data}", use_auth_token=True)
    zero_shot(dataset)