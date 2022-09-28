import os
import argparse
import collections
from tqdm import tqdm
import numpy as np
import faiss
import torch
from datasets import load_dataset, load_metric
from transformers import set_seed, AutoConfig, AutoModel, AutoTokenizer

import wandb

wandb.init(project="indicxtreme")

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--src_file", type=str, default="hin_Deva")
parser.add_argument("--tgt_file", type=str, default="eng_Latn")
parser.add_argument("--task_name", type=str, default="flores")
parser.add_argument("--metric", type=str, default="squad")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--fp16", type=bool, default=True)
parser.add_argument("--num_layers", type=int, default=12)
parser.add_argument("--specific_layers", type=str, default="11")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--embed_size", type=int, default=768)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--pool_type", type=str, default="mean")
parser.add_argument("--dist", type=str, default="cosine")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

def similarity_search(x, y, dim, normalize=False):
    num = x.shape[0]
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
        faiss.normalize_L2(y)
    idx.add(x)
    scores, prediction = idx.search(y, 1)
    return prediction

def cls_pool_embedding(all_layer_outputs):
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = embeds[:, 0, :]
        sent_embeds.append(embeds)
    return sent_embeds


def mean_pool_embedding(all_layer_outputs):
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = embeds.mean(dim=1)
        sent_embeds.append(embeds)
    return sent_embeds

def extract_embeddings(lines):
    num_embeds = args.num_layers
    num_batch = int(np.ceil(len(lines) * 1.0 / args.batch_size))
    num_sents = len(lines)

    model.to(args.device)
    model.eval()

    all_embeds = [np.zeros(shape=(num_sents, args.embed_size), dtype=np.float32) for _ in range(num_embeds)]
    for i in tqdm(range(num_batch), desc='Batch'):
        start_index = i * args.batch_size
        end_index = min((i + 1) * args.batch_size, num_sents)

        # Tokenize the inputs
        tokenized_inputs = tokenizer(
            lines[start_index:end_index],
            max_length=args.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_inputs.to(args.device)

        # Get the embeddings
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            last_hidden_state, pooler_output, hidden_states = outputs['last_hidden_state'], outputs['pooler_output'], outputs['hidden_states']

            if args.pool_type == 'cls':
                all_batch_embeds = cls_pool_embedding(hidden_states[-args.num_layers:])
            elif args.pool_type == 'mean':
                all_batch_embeds = mean_pool_embedding(hidden_states[-args.num_layers:])
            elif args.pool_type == 'pooler_output':
                all_batch_embeds = [pooler_output] * num_embeds

        for embeds, batch_embeds in zip(all_embeds, all_batch_embeds):
            embeds[start_index: end_index] = batch_embeds.cpu().numpy().astype(np.float32)

        del last_hidden_state, pooler_output, hidden_states

        torch.cuda.empty_cache()
    return all_embeds

config = AutoConfig.from_pretrained(args.model_name, output_hidden_states=True, use_auth_token=True)
tokenizer = AutoTokenizer.from_pretrained(f"{args.model_name}", use_auth_token=True)
model = AutoModel.from_pretrained(f"{args.model_name}", config=config, use_auth_token=True)

if args.do_predict:

    if args.task_name == "flores":
        src = load_dataset("facebook/flores", f"{args.src_file}")
        tgt = load_dataset("facebook/flores", f"{args.tgt_file}")

        src_embeddings = extract_embeddings(src['devtest']['sentence'])
        tgt_embeddings = extract_embeddings(tgt['devtest']['sentence'])

    elif args.task_name == "tatoeba":
        tatoeba = load_dataset("xtreme", f"tatoeba.{args.src_file}")

        src_embeddings = extract_embeddings(tatoeba['validation']['source_sentence'])
        tgt_embeddings = extract_embeddings(tatoeba['validation']['target_sentence'])
    
    print('finished extracting embeddings')

    for i in args.specific_layers.split():
        x, y = src_embeddings[int(i)], tgt_embeddings[int(i)]

        predictions = similarity_search(x, y, args.embed_size, normalize=(args.dist == 'cosine'))

        count = 0
        for idx, pred in enumerate(predictions):
            if idx == pred:
                count += 1

        print(f"Accuracy for {args.src_file} @ Layer {int(i)+1}: {count / predictions.shape[0]}")
