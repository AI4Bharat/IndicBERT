# data prep needs to run on tf 2.0 (based on Relate LM repo)
conda create -n tpu_data_prep python=3.7

pip install tokenizers transformers
pip install tqdm joblib indic-nlp-library
conda install tensorflow==2.3.0