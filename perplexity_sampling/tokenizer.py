import seqio
import argparse

import sentencepiece as spm


parser = argparse.ArgumentParser()
parser.add_argument("--build_tokenizer", default=False, type=bool)
parser.add_argument("--tokenizer_input", default=None, type=str)
parser.add_argument("--tokenizer_prefix", default=None, type=str)
args = parser.parse_args()

if __name__ == '__main__':
    if args.build_tokenizer:
        spm.SentencePieceTrainer.train(
            input=args.tokenizer_input,
            model_prefix=args.tokenizer_prefix,
            pad_id=0, bos_id=1,
            eos_id=2, unk_id=3,
            model_type="word",
            vocab_size=1000000,
            character_coverage=1.0
            )