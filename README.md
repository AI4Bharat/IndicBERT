# IndicXTREME: A Multi-Task Benchmark For Evaluating Indic Languages

This repository contains the code of the paper titled "IndicXTREME: A Multi-Task Benchmark For Evaluating Indic Languages"

<p align="center">
  <!-- <a href=""><img src="https://img.shields.io/badge/XXX%20-2023-blue"></a> -->
  <a href="https://github.com/ai4bharat/IndicBERT/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>

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

## LICENSE
All the datasets created as part of this work will be released under a [CC-0 license](https://creativecommons.org/publicdomain/zero/1.0) and all models \& code will be release under an [MIT license](https://github.com/ai4bharat/IndicBERT/blob/main/LICENSE)


## Citation

```
@article{Doddapaneni2022IndicXTREMEAM,
  title={IndicXTREME: A Multi-Task Benchmark For Evaluating Indic Languages},
  author={Sumanth Doddapaneni and Rahul Aralikatte and Gowtham Ramesh and Shreyansh Goyal and Mitesh M. Khapra and Anoop Kunchukuttan and Pratyush Kumar},
  journal={ArXiv},
  year={2022},
  volume={abs/2212.05409}
}
```