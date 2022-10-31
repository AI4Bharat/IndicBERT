from transformers import set_seed, AutoModelForMaskedLM, AutoTokenizer, pipeline
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="ai4bharat/IndicBERT-MLM-SAM-ForMaskedLM")
parser.add_argument("--src_file", type=str, default="hin_Deva")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

map_ = {
    "asm_Beng": "<as>",
    "ben_Beng": "<bn>",
    "guj_Gujr": "<gu>",
    "hin_Deva": "<hi>",
    "kan_Knda": "<kn>",
    "kas_Arab": "<ks>",
    "mai_Deva": "<mai>",
    "mal_Mlym": "<ml>",
    "mar_Deva": "<mr>",
    "mni_Beng": "<mni>",
    "npi_Deva": "<ne>",
    "ory_Orya": "<or>",
    "pan_Guru": "<pa>",
    "san_Deva": "<sa>",
    "sat_Olck": "<sat>",
    "tam_Taml": "<ta>",
    "tel_Telu": "<te>",
    "urd_Arab": "<ur>",
}

def add_mask(example):
    example["sentence"] = f"[MASK] {example['sentence']}"
    return example

if args.do_predict:
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)

    src = load_dataset("facebook/flores", f"{args.src_file}")
    src = src.map(add_mask)

    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0)

    correct_preds = 0
    with torch.no_grad():
        for i in tqdm(range(len(src['devtest']))):
            pred = pipe(src['devtest'][i]['sentence'])
            if pred[0]['token_str'] == map_[args.src_file]:
                correct_preds += 1
            else:
                print("################# ERROR #################")
                print(pred)

    print(f"Correct Predictions: {correct_preds}")
    print(f"Accuracy for {args.src_file}: {correct_preds/len(src['devtest'])}")
