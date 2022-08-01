# this code is to convert paragraphs to sentences where the input file is one paragraph per line and the output file is one sentence per line with line break between each paragraph.

import os
import argparse
from tqdm import tqdm
from indicnlp.tokenize.sentence_tokenize import sentence_split

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="")
parser.add_argument("--output", type=str, default="")
parser.add_argument("--lang", type=str, default="")
args = parser.parse_args()

indicnlp_supported  =['as', 'bn', 'en', 'gu', 'hi', 'kn', 'ml', 'mr', 'ne', 'or', 'pa', 'sa', 'sd', 'ta', 'te', 'ur']

def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines

def main(args):
    lines = read_file(args.input)
    lang = args.lang

    if lang == 'sd':
        lang = 'ur'

    if lang in indicnlp_supported:
        split_lines = [sentence_split(line, lang) for line, lang in tqdm(zip(lines, [lang] * len(lines)))]
    else:
        split_lines = [map(lambda x: x + '.', line.split('.')) for line in tqdm(lines)]

    with open(args.output, 'w') as f:
        for paragraph in split_lines:
            for line in paragraph:
                if len(line) > 5:
                    f.write(line.strip() + '\n')
            f.write('\n')

if __name__ == "__main__":
    main(args)