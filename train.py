#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from datetime import timedelta

import datasets
import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
# from transformers.utils import get_full_repo_name
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from custom_tokenizer.indicXLM_sentencepiece_tokenizer import IndicXLMSentencePieceTokenizer
from signal import signal, SIGPIPE, SIG_DFL  
signal(SIGPIPE,SIG_DFL)

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    # parser.add_argument(
    #     "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    # )
    # parser.add_argument(
    #     "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    # )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--eval_every",
        type=str,
        default=None,
        help="Whether the should be evaluated at end of every n steps",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--mono_languages", type=str, default=None, help="A comma separated list of languages to use for training."
    )
    parser.add_argument(
        "--it_languages", type=str, default=None, help="A comma separated list of languages to use for training."
    )
    parser.add_argument(
        "--xlit_languages", type=str, default=None, help="A comma separated list of languages to use for training."
    )

    parser.add_argument(
        "--mono_train_dir", type=str, default=None, help="A directory containing the training data."
    )
    parser.add_argument(
        "--it_train_dir", type=str, default=None, help="A directory containing the training data."
    )
    parser.add_argument(
        "--xlit_train_dir", type=str, default=None, help="A directory containing the training data."
    )

    parser.add_argument(
        "--mono_eval_dir", type=str, default=None, help="A directory containing the training data."
    )
    parser.add_argument(
        "--it_eval_dir", type=str, default=None, help="A directory containing the training data."
    )
    parser.add_argument(
        "--xlit_eval_dir", type=str, default=None, help="A directory containing the training data."
    )
    args = parser.parse_args()

    # Sanity checks
    if (args.mono_languages is None or args.it_languages is None or args.xlit_languages is None) and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        pass
        '''
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")
        '''
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    timeout_ = InitProcessGroupKwargs(timeout=timedelta(seconds=14400))
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir, kwargs_handlers=[timeout_]) if args.with_tracking else Accelerator(kwargs_handlers=[timeout_])
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = IndicXLMSentencePieceTokenizer.from_pretrained(args.tokenizer_name, pretraining=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line and args.dataset_name is None:
        padding = "max_length" if args.pad_to_max_length else False


        if os.path.isdir('hf-eval'):
            eval_mono_dataset = load_from_disk('hf-eval')
            logger.info("Loaded Evaluation data from disk")
        else:
            mono_langs = args.mono_languages.split(',')
            eval_mono_dict = {}
            for ml in mono_langs:
                logger.info(f"Currently running {ml}")
                data_files = {
                    f'{ml}': args.mono_eval_dir + f'/{ml}.txt'
                }
                raw_datasets = load_dataset('text', data_files=data_files)

                column_names = raw_datasets[f"{ml}"].column_names
                text_column_name = "text" if "text" in column_names else column_names[0]

                def tokenize_function(examples):
                    # Remove empty lines
                    examples[text_column_name] = [
                        line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
                    ]
                    try:
                        return tokenizer(
                            examples[text_column_name],
                            padding=padding,
                            truncation=True,
                            max_length=max_seq_length,
                            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                            # receives the `special_tokens_mask`.
                            return_special_tokens_mask=True,
                        )
                    except:
                        logger.info(examples[text_column_name])

                tokenizer = IndicXLMSentencePieceTokenizer.from_pretrained('abaw_tok', pretraining=True, src_lang=f'{ml}')
                with accelerator.main_process_first():
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=args.preprocessing_num_workers,
                        remove_columns=[text_column_name],
                        load_from_cache_file=not args.overwrite_cache,
                        desc="Running tokenizer on dataset line_by_line",
                    )

                    eval_mono_dict[f'{ml}'] = tokenized_datasets

            eval_mono_dataset = concatenate_datasets([eval_mono_dict[f'{xx}'][f'{xx}'] for xx in mono_langs])
            # save to disk
            eval_mono_dataset.save_to_disk('hf-eval')
        
        # process monolingual data
        if os.path.isdir('hf-mono'):
            full_mono = load_from_disk('hf-mono')
            logger.info("Loaded Monolingual data from disk")
        else:
            mono_langs = args.mono_languages.split(',')
            logger.info(f'Total langs: {len(mono_langs)}')
            mono_dict = {}
            
            for ml in mono_langs:
                if ml == 'sa':
                    continue
                logger.info(f"Currently running {ml}")
                data_files = {
                    f'{ml}': args.mono_train_dir + f'/{ml}.shuf'
                }
                raw_datasets = load_dataset('text', data_files=data_files)

                column_names = raw_datasets[f"{ml}"].column_names
                text_column_name = "text" if "text" in column_names else column_names[0]

                def tokenize_function(examples):
                    # Remove empty lines
                    examples[text_column_name] = [
                        line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
                    ]
                    try:
                        return tokenizer(
                            examples[text_column_name],
                            padding=padding,
                            truncation=True,
                            max_length=max_seq_length,
                            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                            # receives the `special_tokens_mask`.
                            return_special_tokens_mask=True,
                        )
                    except:
                        logger.info(examples[text_column_name])

                tokenizer = IndicXLMSentencePieceTokenizer.from_pretrained('abaw_tok', pretraining=True, src_lang=f'{ml}')
                with accelerator.main_process_first():
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=args.preprocessing_num_workers,
                        remove_columns=[text_column_name],
                        load_from_cache_file=not args.overwrite_cache,
                        desc="Running tokenizer on dataset line_by_line",
                    )

                    mono_dict[f'{ml}'] = tokenized_datasets

            full_mono = concatenate_datasets([mono_dict[f'{xx}'][f'{xx}'] for xx in mono_langs])
            full_mono.save_to_disk('hf-mono')

        # process xlit data
        if os.path.isdir('hf-xlit'):
            full_xlit = load_from_disk('hf-xlit')
            logger.info("Loaded Xlit data from disk")
        else:
            xlit_langs = args.xlit_languages.split(',')
            xlit_dict = {}
            for xl in xlit_langs:
                logger.info(f"Currently running {xl}")
                data_files = {
                    f'{xl}': args.xlit_train_dir + f'/en-{xl}.csv'
                }
                raw_datasets = load_dataset('csv', data_files=data_files, column_names=['src', 'tgt'])

                def tokenize_function(examples):
                    try:
                        return tokenizer(
                            examples['src'], examples['tgt'],
                            padding=padding,
                            truncation=True,
                            max_length=max_seq_length,
                            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                            # receives the `special_tokens_mask`.
                            return_special_tokens_mask=True,
                        )
                    except:
                        logger.info(examples['src'])
                        logger.info(examples['tgt'])
                
                tokenizer = IndicXLMSentencePieceTokenizer.from_pretrained('abaw_tok', pretraining=True, src_lang=f'{xl}', tgt_lang='en')
                with accelerator.main_process_first():
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=args.preprocessing_num_workers,
                        remove_columns=['src', 'tgt'],
                        load_from_cache_file=not args.overwrite_cache,
                        desc="Running tokenizer on dataset line_by_line",
                    )

                    xlit_dict[f'{xl}'] = tokenized_datasets

            full_xlit = concatenate_datasets([xlit_dict[f'{xx}'][f'{xx}'] for xx in xlit_langs])
            full_xlit.save_to_disk('hf-xlit')

        # process indictrans data
        if os.path.isdir('hf-it'):
            full_it = load_from_disk('hf-it')
            logger.info("Loaded IndicTrans data from disk")
        else:
            it_langs = args.it_languages.split(',')
            indictrans_dict = {}
            for il in it_langs:
                logger.info(f"Currently running {il}")
                data_files = {
                    f'{il}': args.it_train_dir + f'/en-{il}.csv'
                }
                raw_datasets = load_dataset('csv', data_files=data_files, column_names=['src', 'tgt'])

                def tokenize_function(examples):
                    try:
                        return tokenizer(
                            examples['src'], examples['tgt'],
                            padding=padding,
                            truncation=True,
                            max_length=max_seq_length,
                            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                            # receives the `special_tokens_mask`.
                            return_special_tokens_mask=True,
                        )
                    except:
                        logger.info(examples['src'])
                        logger.info(examples['tgt'])
                
                tokenizer = IndicXLMSentencePieceTokenizer.from_pretrained('abaw_tok', pretraining=True, src_lang=f'{il}', tgt_lang='en')
                with accelerator.main_process_first():
                    tokenized_datasets = raw_datasets.map(
                        tokenize_function,
                        batched=True,
                        num_proc=args.preprocessing_num_workers,
                        remove_columns=['src', 'tgt'],
                        load_from_cache_file=not args.overwrite_cache,
                        desc="Running tokenizer on dataset line_by_line",
                    )

                    indictrans_dict[f'{il}'] = tokenized_datasets

            full_it = concatenate_datasets([indictrans_dict[f'{xx}'][f'{xx}'] for xx in it_langs])
            full_it.save_to_disk('hf-it')
            
        logger.info(full_mono)
        logger.info(full_it)
        logger.info(full_xlit)
        logger.info(eval_mono_dataset)

        full_train_dataset = concatenate_datasets([full_mono, full_it, full_xlit])
        # full_train_dataset.save_to_disk('hf_full_pretrain_data')
        logger.info('Concatenated full train data')

        # full_train_dataset = load_from_disk('hf_full_pretrain_data')
        # logger.info('loaded full train data from disk')
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with accelerator.main_process_first():
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            )
     
    ################ change this ########################
    train_dataset = full_train_dataset
    eval_dataset = eval_mono_dataset

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    tokenizer = IndicXLMSentencePieceTokenizer.from_pretrained('abaw_tok', pretraining=True, src_lang=f'en')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    logger.info('Data collator ready!!')

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, num_workers=args.preprocessing_num_workers
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, num_workers=args.preprocessing_num_workers)

    logger.info('Data loaders ready!!')

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logger.info('Scheduler ready!!')

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers("mlm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            # if isinstance(checkpointing_steps, int):
            #     if completed_steps % checkpointing_steps == 0:
            #         output_dir = f"step_{completed_steps }"
            #         if args.output_dir is not None:
            #             output_dir = os.path.join(args.output_dir, output_dir)
            #         accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

            # do eval every 10k steps
            if completed_steps % int(args.eval_every) == 0:

                model.eval()
                losses = []
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)

                    loss = outputs.loss
                    losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

                losses = torch.cat(losses)
                losses = losses[: len(eval_dataset)]
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                logger.info(f"step {step}: perplexity: {perplexity}")

                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)

                    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                        json.dump({"perplexity": perplexity}, f)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
