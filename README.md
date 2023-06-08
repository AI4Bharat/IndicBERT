# Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages

This repository contains the code and other resources for the paper published at ACL 2023.

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/ACL%20-2023-blue"></a>
  <a href="https://github.com/ai4bharat/IndicBERT/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>

[**Benchmark**](#indicxtreme) | [**Corpus**](#indiccorp-v2) |
[**Models**](#indicbert) |
[**Pretraining**](#pre-train) |
[**Fine-tuning**](#run-fine-tuning) |
[**Paper**](https://arxiv.org/abs/2212.05409)

## IndicXTREME
IndicXTREME benchmark includes 9 tasks that can be broadly grouped into sentence classification (5), structure prediction (2), question answering (1), and sentence retrieval (1).

The list of tasks are as follows:
- **IndicCOPA** - [[Dataset](https://huggingface.co/datasets/ai4bharat/IndicCOPA)] - We manually translate the [COPA](https://people.ict.usc.edu/~gordon/publications/AAAI-SPRING11A.PDF) test set into 18 Indic languages to create IndicCOPA
- **IndicQA** - [[Dataset](https://huggingface.co/datasets/ai4bharat/IndicQA)] - A manually curated cloze-style reading comprehension dataset that can be used for evaluating question-answering models in 11 Indic languages
- **IndicXParaphrase** - [[Dataset](https://huggingface.co/datasets/ai4bharat/IndicXParaphrase)] - A new, multilingual, and n-way parallel dataset for paraphrase detection in 10 Indic languages
- **IndicSentiment** - [[Dataset](https://huggingface.co/datasets/ai4bharat/IndicSentiment)] - A new, multilingual, and n-way parallel dataset for sentiment analysis in 13 Indic languages
- **IndicXNLI** - [[Dataset](https://huggingface.co/datasets/Divyanshu/indicxnli)] - Automatically translated version of XNLI in 11 Indic languages. Created by Divyanshu et. al. in [this paper](https://aclanthology.org/2022.emnlp-main.755/)
- **Naamapadam** - [[Dataset](https://huggingface.co/datasets/ai4bharat/naamapadam)] - NER dataset with manually curated testsets for 9 Indic languages. Created by Arnav et. al in [this paper](https://arxiv.org/abs/2212.10168)
- **MASSIVE** - [[Dataset](https://github.com/alexa/massive#accessing-and-processing-the-data)] - This in an intent classification and slot-filling dataset created using user queries collected by Amazon Alexa for 7 Indic languages. Created by FitzGerald et. al. in [this paper](https://arxiv.org/abs/2204.08582)
- **FLORES** - [[Dataset](https://huggingface.co/datasets/facebook/flores)] - To evaluate the retrieval capabilities of models, we include the Indic parts of the FLORES-101 dataset. Available in 18 Indic languages. Created by NLLB Team et. al. in [this paper](https://arxiv.org/abs/2207.04672)

## IndicCorp v2
| Language | Download Link                                                                                         | Language  | Download Link                                                                                         |
|----------|-------------------------------------------------------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------------|
| Assamese | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/as.txt) | Malayalam | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/ml.txt) |
| Bodo     | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/bd.txt) | Manipuri  | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/mni.txt) |
| Bengali  | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/bn.txt) | Marathi   | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/mr.txt) |
| Dogri    | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/dg.txt) | Nepali    | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/ne.txt) |
| English  | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/en.txt) | Odia      | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/or.txt) |
| Konkani  | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/gom.txt) | Punjabi   | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/pa.txt) |
| Gujarati | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/gu.txt) | Sanskrit  | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/sa.txt) |
| Hindi    | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/hi.txt) | Santali   | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/sat.txt) |
| Khasi    | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/kha.txt) | Sindhi    | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/sd.txt) |
| Kannada  | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/kn.txt) | Tamil     | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/ta.txt) |
| Kashmiri | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/ks.txt) | Telugu    | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/te.txt) |
| Maithili | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/mai.txt) | Urdu      | [Download](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/ur.txt) |

## IndicBERT
A multilingual language model trained on IndicCorp v2 and evaluated on IndicXTREME benchmark. The model has 278M parameters and is available in 23 Indic languages and English. The models are trained with various objectives and datasets. The list of models are as follows:

- IndicBERT-MLM [[Model](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only)] - A vanilla BERT style model trained on IndicCorp v2 with the MLM objective
    - +Samanantar [[Model](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-Sam-TLM)] - TLM as an additional objective with Samanantar Parallel Corpus [[Paper](https://aclanthology.org/2022.tacl-1.9)] | [[Dataset](https://huggingface.co/datasets/ai4bharat/samanantar)]
    - +Back-Translation [[Model](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-Back-TLM)] - TLM as an additional objective by translating the Indic parts of IndicCorp v2 dataset into English w/ IndicTrans model [[Model](https://github.com/AI4Bharat/indicTrans#download-model)]
- IndicBERT-SS [[Model](https://huggingface.co/ai4bharat/IndicBERTv2-SS)] - To encourage better lexical sharing among languages we convert the scripts from Indic languages to Devanagari and train a BERT style model with the MLM objective

## Run Pretraining

### Setup
The current BERT Preprocessig code needs to run in Tensorflow v2. Create a new conda environment and set it up as follows:
```shell
conda create -n tpu_data_prep python=3.7

pip install tokenizers transformers tqdm joblib indic-nlp-library
conda install tensorflow==2.3.0
```

### Train Tokenizer
Train a WordPiece Tokenizer to preprocess the data. The following command trains a tokenizer and saves it in the specified path.

Arguments:
- INPUT: /path/to/input.txt
- OUTPUT: /path/to/output
- VOCAB_SIZE: size of the vocabulary

```shell
python IndicBERT/tokenization/build_tokenizer.py \
    --input_file=$INPUT \
    --output_dir=$OUTPUT \
    --vocab_size=$VOCAB_SIZE
```

### Preprocess Data

Run the following command after update the required paths in the script:

```shell
python IndicBERT/process_data/create_mlm_data.py \
    --input_file=$INPUT \
    --output_file=$OUTPUT \
    --input_file_type=$DATA_TYPE \
    --tokenizer=$TOKENIZER_PATH \
    --max_seq_length=$MAX_SEQ_LEN \
    --max_predictions_per_seq=$MAX_PRED \
    --do_whole_word_mask=$WHOLE_WORD_MASK \
    --masked_lm_prob=$MASK_PROB \
    --random_seed=$SEED \
    --dupe_factor=$DUPE_FACTOR \
```
- DATA_TYPE: `monolingual` or `parallel`
    - `monolingual`: if the input file is a monolingual corpus
        - INPUT:/path/to/input.txt 
            - (one sentence per line, empty line between documents)
            - can take multiple files as input, separated by comma
    - `parallel`: if the input file is a parallel corpus
        - INPUT:/path/to/input
            - (one sentence per line)
            - the input directory should contain two files, `input.en` and `input.lang`
- OUTPUT:/path/to/output.tfrecord
- TOKENIZER_PATH: /path/to/tokenizer/config.json
- MAX_SEQ_LEN: maximum sequence length, generally 512
- MAX_PRED: maximum number of tokens to mask in a sequence
- WHOLE_WORD_MASK: whether to mask whole words or not
- MASK_PROB: probability of masking a token
- DUPE_FACTOR: number of times to duplicate the input data

### Pre-train
The BERT Pretraining code is a modified version of [Google BERT Repo](https://github.com/google-research/bert), without NSP and customisation to support parallel data. The training code need to run on Tensorflow v1. Create a new conda environment and set it up as follows:
```shell
conda env create --name bert_pretraining
conda activate bert_pretraining
conda install -c conda-forge tensorflow==1.14

```


Run the following command for pretraining:
```shell
python IndicBERT/train/run_pretraining.py \
--input_file=$INPUTS \
--output_dir=$OUTPUTS \
--do_train=True \
--bert_config_file=$BERT_CONFIG \
--train_batch_size=$BS \
--max_seq_length=$MAX_SEQ_LEN \
--max_predictions_per_seq=$MAX_PRED \
--num_train_steps=$TRAIN_STEPS \
--num_warmup_steps=$WARMUP \
--learning_rate=$LR \
--save_checkpoints_steps=$SAVE_EVERY \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--tpu_zone=$TPU_ZONE \
--num_tpu_cores=$TPU_CORES
```
> ***Note that to run the pretraining on TPUs, the input data and output directory should be on Google Cloud Storage***

Arguments:
- INPUTS: gs://path/to/input/tfrecords/*
- OUTPUTS: gs://path/to/output
- BERT_CONFIG: /path/to/bert_config.json
- BS: batch size, usually `4096`
- MAX_SEQ_LEN: `should be same as the preprocessing step`
- MAX_PRED: `should be same as the preprocessing step`
- TRAIN_STEPS: number of training steps, usually `1000000`
- WARMUP: number of warmup steps, usually `10000`
- LR: learning rate, usually `5e-4`
- SAVE_EVERY: save checkpoints every `n` steps
- TPU_NAME: name of the TPU
- TPU_ZONE: zone of the TPU
- TPU_CORES: number of TPU cores

## Run Fine-tuning
Fine-tuning scripts are based on transformers library. Create a new conda environment and set it up as follows:
```shell
conda create -n finetuning python=3.9
pip install -r requirements.txt
```

All the tasks follow the same structure, please check individual files for detailed hyper-parameter choices. The following command runs the fine-tuning for a task:
```shell
python IndicBERT/fine-tuning/$TASK_NAME/$TASK_NAME.py \
    --model_name_or_path=$MODEL_NAME \
    --do_train
```
Arguments:
- MODEL_NAME: name of the model to fine-tune, can be a local path or a model from the [HuggingFace Model Hub](https://huggingface.co/models)
- TASK_NAME: one of [`ner, paraphrase, qa, sentiment, xcopa, xnli, flores`]

> For MASSIVE task, please use the instrction provided in the [official repository](https://github.com/alexa/massive) 

## LICENSE
All the datasets created as part of this work will be released under a [CC-0 license](https://creativecommons.org/publicdomain/zero/1.0) and all models \& code will be release under an [MIT license](https://github.com/ai4bharat/IndicBERT/blob/main/LICENSE)


## Citation

```
@article{Doddapaneni2022towards,
  title={Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages},
  author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.05409}
}
```
