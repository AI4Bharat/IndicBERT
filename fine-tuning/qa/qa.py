import argparse
import collections

import numpy as np
from datasets import load_dataset, load_metric
from tqdm.auto import tqdm
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

import wandb

from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import indic_tokenize
from indicnlp.transliterate import unicode_transliterate

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--train_data", type=str, default="SQuAD")
parser.add_argument("--eval_data", type=str, default="indicqa.hi")
parser.add_argument("--metric", type=str, default="squad")
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--max_length", type=int, default=384)
parser.add_argument("--stride", type=int, default=128)
parser.add_argument("--n_best", type=int, default=20)
parser.add_argument("--max_answer_length", type=int, default=30)
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

    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=args.max_length,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=args.max_length,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs


    def compute_metrics(start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > args.max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)


    metric = load_metric(f"{args.metric}")
    dataset_squad =  load_dataset("xtreme", f"{args.train_data}")
    # dataset_tydiqa = load_dataset("tydiqa", "secondary_task")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", use_auth_token=True)
    model = AutoModelForQuestionAnswering.from_pretrained(f"{args.model_name}", use_auth_token=True)


    if args.do_train:
        # train dataset
        # dataset = concatenate_datasets([dataset_squad["train"], dataset_tydiqa["train"]])

        train_dataset = dataset_squad['train'].map(
            preprocess_training_examples,
            batched=True,
            remove_columns=dataset_squad['train'].column_names,
        )
        print(f"Length of Training dataset: {len(train_dataset)}")

        # validation dataset
        validation_dataset = dataset_squad["validation"].map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=dataset_squad["validation"].column_names,
        )
        print(f"Length of Validation datasets: {len(validation_dataset)}")

        # training arguments
        training_args = TrainingArguments(
            output_dir=f"{args.output_dir}/{args.model_name}-{str(config.lr)}-{str(config.weight_decay)}",
            save_total_limit=5,
            save_strategy="epoch",
            learning_rate=config.lr,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=config.weight_decay,
            fp16=args.fp16,
            warmup_ratio=args.warmup_ratio,
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

        predictions, _, _ = trainer.predict(validation_dataset)
        start_logits, end_logits = predictions
        print(compute_metrics(start_logits, end_logits, validation_dataset, dataset_squad["validation"]))

if args.do_train:
    sweep_config = {
        'method': 'grid',
        'parameters': {
            # 'lr': {
            #     'values': [1e-5, 3e-5, 5e-6]
            # },
            # 'weight_decay': {
            #     'values': [0.0, 0.01]
            # }
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
        example['context'] = preprocess_line(example['context'], lang)
        example['question'] = preprocess_line(example['question'], lang)
        example['answers']['text'] = [preprocess_line(example['answers']['text'][0], lang)]
        if example['answers']['text'][0] != "":
            example['answers']['answer_start'] = [example['context'].find(example['answers']['text'][0])]
        return example

    metric = load_metric(f"{args.metric}")
    dataset_squad =  load_dataset("xtreme", f"{args.train_data}")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", use_auth_token=True)
    model = AutoModelForQuestionAnswering.from_pretrained(f"{args.model_name}", use_auth_token=True)

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=args.max_length,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    
    def compute_metrics(start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -args.n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -args.n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > args.max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)


    def zero_shot(dataset, name):
        val_dataset = dataset["validation"].map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=dataset["validation"].column_names,

        )
        predictions_d, _, _ = trainer.predict(val_dataset)
        start_logits_d, end_logits_d = predictions_d
        print(f"Results for {name}: {compute_metrics(start_logits_d, end_logits_d, val_dataset, dataset['validation'])}")

    training_args = TrainingArguments(
        output_dir=f"{args.model_name}",
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    eval_dataset = load_dataset("squad")
    zero_shot(eval_dataset, "En-SQuAD")
    eval_dataset = load_dataset("xtreme", f"MLQA.hi.hi")
    zero_shot(eval_dataset, "Hi-MLQA")
    for lang in "as bn gu hi kn ml mr or pa ta te".split():
        eval_dataset = load_dataset("ai4bharat/IndicQA", f"indicqa.{lang}", use_auth_token=True)
        if lang in supported_xlit_langs:
            eval_dataset = eval_dataset.map(lambda x: transliterate(x, lang))
        zero_shot(eval_dataset, f"indicqa-{lang}")