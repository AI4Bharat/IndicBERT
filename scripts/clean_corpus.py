# take a folder as input and gather all the text files
# make a csv with content of text file with a column to indicate source which is the filename

# pip3 install pandarallel pycld3 langid
import pandas as pd
import os
import argparse
from tqdm import tqdm
import langid

# import string
from cld3 import get_language
from pandarallel import pandarallel
import re
import time


pandarallel.initialize()


LANGS = "as bn en gu hi kn ks gom mai ml mni mr ne or pa sa sat sd ta te ur".split()
LOW_RESOURCE_LANGS = ["ks", "gom", "mai", "mni", "sat", "sa"]


def check_path_and_folder_or_file(path):
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(f"{path} is not a valid path")


def read_txt(file_path):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()

    return lines


def write_txt(file_path, data, shuffle=False):
    def shuffle_list(l):
        import random

        random.shuffle(l)
        return l

    if shuffle:
        data = shuffle_list(data)
    with open(file_path, "w") as f:
        f.write("\n".join(data))


def get_fname_and_ext(file_path):
    fname, ext = os.path.splitext(file_path)
    return fname, ext


def strip_punctuation_and_numbers(text):

    text = text.lower()

    # remove punctuations
    text = text.translate(
        str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "\u0964")
    )

    # remove english numbers
    text = re.sub(r"[0-9]+", "", text)

    # remove space
    text = text.replace(" ", "")

    return text


def detect_lang(text, lang):
    def pycld3_func(text):
        try:
            language_resuls = get_language(text)
            if language_resuls.is_reliable:
                return language_resuls.language
            else:
                return "undetermined"
        except Exception:
            return "undetermined"

    def langid_func(text):
        return langid.classify(text)[0]

    if lang in ["as", "or"]:
        func = langid_func
    elif lang in [
        "bn",
        "en",
        "gu",
        "hi",
        "kn",
        "ml",
        "mr",
        "ne",
        "pa",
        "sd",
        "ta",
        "te",
        "ur",
    ]:
        func = pycld3_func
    elif lang in LOW_RESOURCE_LANGS:  # unsupported languages
        # we can still use pycld3 to get some outputs and filter out if there
        # are english sentences in these language files
        func = pycld3_func
    else:
        raise ValueError(f"Unsupported language {lang}")

    return func(text)


def match_pattern(text, pattern):
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.group(0)
    else:
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--bad_words_file_path", type=str, default=None)
    parser.add_argument(
        "--shuffle", action="store_true"
    )  # whether to shuffle the sentences before saving
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    assert args.lang in LANGS, f"{args.lang} is not supported"
    assert args.input_folder, "Please specify the input folder"

    if args.output_dir == None:
        args.output_dir = "./"
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    lang = args.lang
    output_dir = args.output_dir

    input_folder = check_path_and_folder_or_file(args.input_folder)
    output_csv = f"{output_dir}/{lang}_processed_sents.csv"

    if args.shuffle:
        output_processed_sents_file = f"{output_dir}/{lang}.shuffled.final"
    else:

        output_processed_sents_file = f"{output_dir}/{lang}.final"
    if args.bad_words_file_path:
        bad_words_file_path = check_path_and_folder_or_file(args.bad_words_file_path)

        # https://stackoverflow.com/a/17973255/4954882
        # use regular expression to find if sentence contains bad words
        bad_words_list = list(set(read_txt(bad_words_file_path)))
        pattern = "|".join(bad_words_list)

        # df.apply(lambda col: col.str.contains('f  oo|bar', na=False), axis=1)
    bad_sources = ["eferrit", "birmiss", "atomiyme"]

    start = time.time()
    data = []
    sources = []
    for filename in os.listdir(input_folder):
        fname, ext = get_fname_and_ext(filename)
        # check if fname is in bad_sources or bad_sources are in fname
        if (
            any(bad_source in fname for bad_source in bad_sources)
            or fname in bad_sources
        ):
            continue
        if ext == ".txt":
            file_path = os.path.join(input_folder, filename)
            lines = read_txt(file_path)
            # data.extend(lines)
            data = lines
            sources = [fname] * len(lines)

        df = pd.DataFrame({"source": sources, "sent": data})
        # deduplicate based on sentence
        df = df.drop_duplicates("sent")
        num_sents = len(df)
        print(f"{num_sents} unique sentences in total")
        end = time.time()
        print(f"Time taken to read all files: {end - start}")
        if not num_sents:
            continue

        start = time.time()
        try:
            df["lang"] = df.sent.parallel_apply(lambda x: detect_lang(x, lang))
        except:
            print('Error!!!')
            continue
        print("Language detection done")
        if lang in LOW_RESOURCE_LANGS:
            # count number of english sentences
            en_count = len(df[df.lang == "en"])
            print(f"{en_count} english sentences detected")
            remaining_sents = num_sents - en_count

        else:
            # count number of lang sentences
            lang_count = len(df[df.lang == lang])
            print(f"{lang_count} {lang} sentences detected")
            remaining_sents = lang_count

        print(f"{remaining_sents} remaining sentences")
        end = time.time()
        print(f"Time taken to detect language: {end - start}")

        if args.bad_words_file_path:
            start = time.time()
            df["bad_word"] = df.sent.parallel_apply(lambda x: match_pattern(x, pattern))
            print("Bad word detection done")

            bad_word_count = len(df[df.bad_word != ""])
            print(f"{bad_word_count} bad word sentences detected")
            remaining_sents = remaining_sents - bad_word_count
            print(f"{remaining_sents} remaining sentences")
            end = time.time()
            print(f"Time taken to detect bad words: {end - start}")

        start = time.time()
        df["punct_num_stripped_sent"] = df.sent.parallel_apply(
            strip_punctuation_and_numbers
        )
        print("Punctuation and numbers stripped")
        end = time.time()
        print(f"Time taken to strip punctuation and numbers: {end - start}")
        # save this csv
        output_csv = f"{output_dir}/{filename}_processed_sents.csv"
        df.to_csv(output_csv, index=False)

        # deduplicate dataframe based on stripped sentences
        stripped_df = df.drop_duplicates(subset="punct_num_stripped_sent")
        print(f"{len(stripped_df)} unique sentences")

        if lang in LOW_RESOURCE_LANGS:
            proper_sentences = stripped_df[
                (stripped_df.lang != "en") & (stripped_df.bad_word == "")
            ].sent.tolist()
        else:
            proper_sentences = stripped_df[
                (stripped_df.lang == lang) & (stripped_df.bad_word == "")
            ].sent.tolist()

        # save stripped sentences as txt
        if args.shuffle:
            output_processed_sents_file = f"{output_dir}/{filename}.shuffled.final"
        else:
            output_processed_sents_file = f"{output_dir}/{filename}.final"

        write_txt(output_processed_sents_file, proper_sentences, args.shuffle)
        print(f"{len(proper_sentences)} sentences saved to {output_processed_sents_file}")