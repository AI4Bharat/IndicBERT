# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import itertools
import collections
from operator import index
import random
from turtle import back
import tokenization
import tensorflow as tf
from absl import logging
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from transformers import PreTrainedTokenizerFast

flags = tf.compat.v1.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files). if input_file_type is parallel then expected input format is src-tgt where src-tgt.src and src-tgt.tgt exist")

flags.DEFINE_string("input_file_type", None,
                    "monolingual or parallel")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("tokenizer", None,
                    "The trained tokenizer path")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

flags.DEFINE_integer("num_workers", 128, "Num. of CPU workers for parallel processing")

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            logging.info("*** Example ***")
            logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
                    
    for writer in writers:
        writer.close()

    logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def create_parallel_instances(src_files, tgt_files, tokenizer, max_seq_length,
                                dupe_factor, short_seq_prob, masked_lm_prob,
                                max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from parallel text."""

    # Input file format:
    # 2 text files one each with source and target data
    # each line containing one sentence and a \n separating 2 paragraphs
    # num. lines source == num. lines of target

    src_documents = [[]]
    tgt_documents = [[]]
    start = time.time()
    count = 0
    for src_file, tgt_file in zip(src_files, tgt_files):
        with tf.io.gfile.GFile(src_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                if not line:
                    src_documents.append([])

                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(line))
                if tokens:
                    src_documents[-1].append(tokens)

                count += 1
                if count % 1000000 == 0:
                    print(count)
        src_documents.append([])

        count = 0
        with tf.io.gfile.GFile(tgt_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                if not line:
                    tgt_documents.append([])

                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(line))
                if tokens:
                    tgt_documents[-1].append(tokens)

                count += 1
                if count % 1000000 == 0:
                    print(count)
        tgt_documents.append([])
    end = time.time()
    logging.info(f'*** time to tokenize: {end-start}')

    src_documents = [x for x in src_documents if x]
    tgt_documents = [x for x in tgt_documents if x]

    all_documents = list(zip(src_documents, tgt_documents))
    random.shuffle(all_documents)
    src_documents, tgt_documents = zip(*all_documents)

    assert len(src_documents) == len(tgt_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []

    # parallel implementation
    for _ in range(dupe_factor):
        document_instance = Parallel(n_jobs=FLAGS.num_workers, prefer="threads", verbose=1)(
                delayed(create_parallel_instances_from_document) \
                    (src_documents, tgt_documents, doc_index, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, vocab_words, rng) \
                    for doc_index in range(len(src_documents))
            )
        x = [l for l in document_instance if l]
        instances += list(itertools.chain.from_iterable(x))

    return instances

def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text.
    # (2) Blank lines between documents. Document boundaries are needed so
    # that we don't use 2 unrelated sentences for attending to each other
    start = time.time()
    count = 0
    for input_file in input_files:
        with tf.io.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(line))
                if tokens:
                    all_documents[-1].append(tokens)

                count += 1
                if count % 1000000 == 0:
                    print(count)
        all_documents.append([])
    end = time.time()
    logging.info(f'*** time to tokenize: {end-start}')

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []

    # parallel implementation
    for _ in range(dupe_factor):
        document_instance = Parallel(n_jobs=FLAGS.num_workers, prefer="threads", verbose=1)(
                delayed(create_instances_from_document) \
                    (all_documents, doc_index, max_seq_length, short_seq_prob, masked_lm_prob, max_predictions_per_seq, vocab_words, rng) \
                    for doc_index in range(len(all_documents))
            )  
        instances += list(itertools.chain.from_iterable(document_instance))

    return instances

def create_parallel_instances_from_document(
    src_documents, tgt_documents, idx, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):

    src_doc = src_documents[idx]
    tgt_doc = tgt_documents[idx]

    if len(src_doc) != len(tgt_doc):
        return

    assert len(src_doc) == len(tgt_doc)

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # check `create_instances_from_document` for documentation
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    instances = []
    source_chunk = []
    target_chunk = []
    source_length = 0
    target_length = 0

    i = 0
    while i < len(src_doc):
        src_segment = src_doc[i]
        tgt_segment = tgt_doc[i]

        if source_length + len(src_segment) + target_length + len(tgt_segment) < target_seq_length:
            source_chunk.append(src_segment)
            source_length += len(src_segment)

            target_chunk.append(tgt_segment)
            target_length += len(tgt_segment)

        if i == len(src_doc) - 1 or source_length + len(src_segment) + target_length + len(tgt_segment) > target_seq_length:
            if source_chunk and target_chunk:
                tokens_a = []
                for j in range(len(source_chunk)):
                    tokens_a.extend(source_chunk[j])

                tokens_b = []
                for j in range(len(target_chunk)):
                    tokens_b.extend(target_chunk[j])

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                # sanity check
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append('[SEP]')
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

                tokens.append('[SEP]')
                segment_ids.append(1)

                (token, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng
                )

                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels
                )
                instances.append(instance)

            source_chunk = []
            target_chunk = []

            if i == len(src_doc) - 1:
                source_chunk.append(src_segment)
                source_length = len(src_segment)

                target_chunk.append(tgt_segment)
                target_length = len(tgt_segment)
            else:
                source_length = 0
                target_length = 0
        i += 1

    return instances


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""

    document = all_documents[document_index]

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)


    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # we change this to single sentence input with a single [CLS] and [SEP]
                a_end = len(current_chunk)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                # for token in tokens_b:
                #     tokens.append(token)
                #     segment_ids.append(1)
                # tokens.append("[SEP]")
                # segment_ids.append(1)

                (tokens, masked_lm_positions,
                masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
            token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("tokenizer")
    logging.info("do_whole_word_mask: %s" % FLAGS.do_whole_word_mask)
    logging.set_verbosity(logging.INFO)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=FLAGS.tokenizer)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

    if FLAGS.input_file_type == "monolingual":
        logging.info("*** Reading MLM from input files ***")
        for input_file in input_files:
            logging.info("  %s", input_file)

        rng = random.Random(FLAGS.random_seed)
        output_files = FLAGS.output_file.split(",")
        instances = create_training_instances(
            input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
            FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
            rng)
        logging.info("*** Writing MLM to output files ***")
        write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                        FLAGS.max_predictions_per_seq, output_files)

    elif FLAGS.input_file_type == "parallel":
        # expected file pattern: `file` so that we can extract file.en, file.lang
        src_files = []
        tgt_files = []
        for parallel_pattern in FLAGS.input_file.split(","):

            # basename = os.path.basename(parallel_pattern)

            # to handle splits
            # basename, split = basename.split('.')[0], basename.split('.')[1]
            # src, tgt = basename.split(',')[0].split('-')

            # 50% make `en` as sent1 and 50% time make `lang` as sent1
            if random.uniform(0,1) > 0.5:
                src_files.extend(tf.compat.v1.gfile.Glob(f'{parallel_pattern}.en'))
                tgt_files.extend(tf.compat.v1.gfile.Glob(f'{parallel_pattern}.lang'))
            else:
                src_files.extend(tf.compat.v1.gfile.Glob(f'{parallel_pattern}.lang'))
                tgt_files.extend(tf.compat.v1.gfile.Glob(f'{parallel_pattern}.en'))

        logging.info("*** Reading TLM from input files ***")
        for src, tgt in zip(src_files, tgt_files):
            logging.info(f'src: {src}, tgt: {tgt}')

        rng = random.Random(FLAGS.random_seed)
        output_files = FLAGS.output_file.split(",")
        parallel_instances = create_parallel_instances(
            src_files, tgt_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
            FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
            rng)
        logging.info("*** Writing TLM to output files ***")
        write_instance_to_example_files(parallel_instances, tokenizer, FLAGS.max_seq_length,
                                        FLAGS.max_predictions_per_seq, output_files)

    else:
        print("Please pass correct argument for input_file_type: monolingual or parallel")