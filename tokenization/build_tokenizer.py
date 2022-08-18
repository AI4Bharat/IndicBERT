import os
import json
import glob
import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="")
parser.add_argument("--vocab_size", type=int, default=250000)
parser.add_argument("--output", type=str, default="")
args = parser.parse_args()

special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<as>", "<bd>", "<bn>", "<dg>", "<en>", "<gom>", "<gu>", "<hi>", "<kha>", "<kn>", "<ks>", "<mai>", "<ml>", "<mni>", "<mr>", "<ne>", "<or>", "<pa>", "<sa>", "<sd>", "<sat>", "<ta>", "<te>", "<ur>"]

def main(args):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    trainer = WordPieceTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)

    tokenizer.pre_tokenizer = Whitespace()

    files = glob.glob(f"{args.input}/*")
    tokenizer.train(files, trainer)

    os.mkdir(args.output)
    tokenizer.save(f"{args.output}/config.json")

    with open(f"{args.output}/config.json", 'r') as f:
        data = json.load(f)
        vocab = list(data['model']['vocab'].keys())

    with open(f"{args.output}/vocab.txt", 'w') as f:
        for word in vocab:
            f.write(word + '\n')

if __name__ == "__main__":
    main(args)