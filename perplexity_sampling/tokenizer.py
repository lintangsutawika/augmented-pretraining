import seqio
import argparse

import sentencepiece as spm


parser = argparse.ArgumentParser()
parser.add_argument("--build_tokenizer", default=False, type=bool)
parser.add_argument("--tokenizer_input", default=None, type=str)
parser.add_argument("--tokenizer_prefix", default=None, type=str)
parser.add_argument("--num_threads", default=16, type=int)
parser.add_argument("--vocab_size", default=500_000, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    if args.build_tokenizer:
        spm.SentencePieceTrainer.train(
            input=args.tokenizer_input,
            model_prefix=args.tokenizer_prefix,
            num_threads=args.num_threads,
            vocab_size=args.vocab_size,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            max_sentence_length=64_000,
            hard_vocab_limit=False,
            model_type="bpe",
            )