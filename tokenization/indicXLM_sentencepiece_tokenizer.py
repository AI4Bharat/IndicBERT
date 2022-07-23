import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm

from transformers.file_utils import is_sentencepiece_available
from transformers.tokenization_utils import AddedToken, BatchEncoding
from transformers.tokenization_utils_fast import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)
from transformers.utils import logging

from tokenizers import processors
from transformers import XLMRobertaTokenizer


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

# VOCAB_FILES_NAMES = {"vocab_file": "/nlsasfs/home/ai4bharat/gramesh/fine-tuning/IndicXLM/tools/indicXLM_sentencepiece_hf2_og_base_paragraphs/spm.model"}
VOCAB_FILES_NAMES = {"vocab_file": "/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/tokenizer/spm_250k_unigram_abaw2/spm_250k_unigram_abaw2.model"}
FAIRSEQ_LANGUAGE_CODES = [
    "<as>",
    "<bd>",
    "<bn>",
    "<dg>",
    "<en>",
    "<gom>",
    "<gu>",
    "<hi>",
    "<kha>",
    "<kn>",
    "<ks>",
    "<mai>",
    "<ml>",
    "<mni>",
    "<mr>",
    "<ne>",
    "<or>",
    "<pa>",
    "<sa>",
    "<sat>",
    "<sd>",
    "<ta>",
    "<te>",
    "<ur>",
]


class IndicXLMSentencePieceTokenizer(PreTrainedTokenizerFast):
    """
    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            .. note::
                When building a sequence using special tokens, this is not the token that is used for the beginning of
                sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
            .. note::
                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<s>NOTUSED", "</s>NOTUSED"]`):
            Additional special tokens used by the tokenizer.
        sp_model_kwargs (:obj:`dict`, `optional`):
            Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
            <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:
            - ``enable_sampling``: Enable subword regularization.
            - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.
              - ``nbest_size = {0,1}``: No sampling is performed.
              - ``nbest_size > 1``: samples from the nbest_size results.
              - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
            - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
    Attributes:
        sp_model (:obj:`SentencePieceProcessor`):
            The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []
    # slow_tokenizer_class = XLMRobertaTokenizer

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        src_lang=None,
        tgt_lang=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs=None,
        **kwargs,
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", [])
        # kwargs["additional_special_tokens"] += [
        #     code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        # ]

        super().__init__(
            vocab_file,
            src_lang=src_lang,
            tokenizer_file=tokenizer_file,
            tgt_lang=tgt_lang,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}

        # In our case there is no fairesq offset
        self.fairseq_offset = 0

        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {
            v: k for k, v in self.fairseq_tokens_to_ids.items()
        }

        self.can_save_slow_tokenizer = False if not self.vocab_file else True

        if src_lang is not None:
            src_lang = f"<{src_lang}>"

        if tgt_lang is not None:
            tgt_lang = f"<{tgt_lang}>"

        self.lang_code_to_id = {
            lang_code: self.convert_tokens_to_ids(lang_code)
            for lang_code in FAIRSEQ_LANGUAGE_CODES
        }

        self._src_lang = src_lang if src_lang is not None else "<en>"
        self.tgt_lang = tgt_lang
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.set_src_lang_special_tokens(self._src_lang, self.tgt_lang)

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def set_src_lang_special_tokens(self, src_lang: str, tgt_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        # self.cur_lang_code_id = self.convert_tokens_to_ids(src_lang)
        # self.cls_tokens = [self.cls_token_id]
        # self.prefix_tokens = [self.cur_lang_code_id]
        # self.suffix_tokens = [self.eos_token_id]

        # cls_token_str = self.convert_ids_to_tokens(self.cls_tokens)
        # prefix_tokens_str = self.convert_ids_to_tokens(self.prefix_tokens)
        # suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        # self._tokenizer.post_processor = processors.TemplateProcessing(
        #     single=cls_token_str + prefix_tokens_str + ["$A"] + suffix_tokens_str,
        #     pair=cls_token_str
        #     + prefix_tokens_str
        #     + ["$A"]
        #     + suffix_tokens_str
        #     + suffix_tokens_str
        #     + prefix_tokens_str
        #     + ["$B"]
        #     + suffix_tokens_str,
        #     special_tokens=list(
        #         zip(
        #             cls_token_str + prefix_tokens_str + suffix_tokens_str,
        #             self.cls_tokens + self.prefix_tokens + self.suffix_tokens,
        #         )
        #     ),
        # )
        self.src_lang_code_id = self.convert_tokens_to_ids(src_lang)
        self.cls_tokens = [self.cls_token_id]
        self.src_lang_tokens = [self.src_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]
        suffix_tokens_str = self.convert_ids_to_tokens(self.suffix_tokens)

        cls_token_str = self.convert_ids_to_tokens(self.cls_tokens)
        src_lang_tokens_str = self.convert_ids_to_tokens(self.src_lang_tokens)

        if self.tgt_lang:
            self.tgt_lang_code_id = self.convert_tokens_to_ids(tgt_lang)
            self.tgt_lang_tokens = [self.tgt_lang_code_id]
            tgt_lang_tokens_str = self.convert_ids_to_tokens(self.tgt_lang_tokens)
            

            self._tokenizer.post_processor = processors.TemplateProcessing(
                single=cls_token_str + src_lang_tokens_str + ["$A"] + suffix_tokens_str,
                pair=cls_token_str
                + src_lang_tokens_str
                + ["$A"]
                + suffix_tokens_str
                + suffix_tokens_str
                + tgt_lang_tokens_str
                + ["$B"]
                + suffix_tokens_str,
                special_tokens=list(
                    zip(
                        cls_token_str + src_lang_tokens_str + tgt_lang_tokens_str + suffix_tokens_str,
                        self.cls_tokens + self.src_lang_tokens + self.tgt_lang_tokens + self.suffix_tokens,
                    )
                ),
            )
        else:
            self._tokenizer.post_processor = processors.TemplateProcessing(
            single=cls_token_str + src_lang_tokens_str + ["$A"] + suffix_tokens_str,
            pair=cls_token_str
            + src_lang_tokens_str
            + ["$A"]
            + suffix_tokens_str
            + suffix_tokens_str
            + src_lang_tokens_str
            + ["$B"]
            + suffix_tokens_str,
            special_tokens=list(
                zip(
                    cls_token_str + src_lang_tokens_str + suffix_tokens_str,
                    self.cls_tokens + self.src_lang_tokens + self.suffix_tokens,
                )
            ),
        )


    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An XLM-RoBERTa sequence has the following format:
        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        # if token_ids_1 is None:
        #     return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        # cls = [self.cls_token_id]
        # sep = [self.sep_token_id]
        # return cls + token_ids_0 + sep + sep + token_ids_1 + sep

        # print("cls", self.cls_token_id)
        # print("sep", self.sep_token_id)
        # print("eos", self.eos_token_id)
        # print("mask", self.mask_token_id)
        # print("pad", self.pad_token_id)
        # print("unk", self.unk_token_id)
        # print("prefix tokens", self.prefix_tokens)
        # print("tokens_0", token_ids_0)

        # if token_ids_1 is not None:
        #     print("tokens_1", token_ids_1)

        if token_ids_1 is None:
            return (
                self.cls_tokens + self.src_lang_tokens + token_ids_0 + self.suffix_tokens
            )
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return (
            self.cls_tokens
            + self.src_lang_tokens
            + token_ids_0
            + self.suffix_tokens
            + self.suffix_tokens
            + self.tgt_lang_tokens
            + token_ids_1
            + self.suffix_tokens
        )

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        # if already_has_special_tokens:
        #     return super().get_special_tokens_mask(
        #         token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
        #     )

        if token_ids_1 is None:
            # cls + lang_id + seq_tokens + sep token
            return [1, 1] + ([0] * len(token_ids_0)) + [1]
        # cls + lang_id + seq_tokens + sep token, sep token, lang_id + seq_tokens + sep token
        else:
            return (
                [1, 1]
                + ([0] * len(token_ids_0))
                + [1, 1, 1]
                + ([0] * len(token_ids_1))
                + [1]
            )


    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. XLM-RoBERTa does
        not make use of token type ids, therefore a list of zeros is returned.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of zeros.
        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        src_lang_id = [self.src_lang_code_id]
        tgt_lang_id = [self.tgt_lang_code_id]

        if token_ids_1 is None:
            return len(cls + src_lang_id + token_ids_0 + sep) * [0]
        return len(
            cls + src_lang_id + token_ids_0 + sep + sep + tgt_lang_id + token_ids_1 + sep
        ) * [0]

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset + 1  # Add the <mask> token

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
