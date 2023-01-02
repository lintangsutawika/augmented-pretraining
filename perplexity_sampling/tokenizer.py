import seqio
import argparse

import sentencepiece as spm

from perplexity_sampling.util import StringTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--build_tokenizer", type=bool)
parser.add_argument("--tokenizer_input", default=None, type=str)
parser.add_argument("--tokenizer_prefix", default=None, type=str)
parser.add_argument("--num_threads", default=16, type=int)
parser.add_argument("--vocab_size", default=25_000, type=int)
args = parser.parse_args()

if __name__ == '__main__':

    if args.build_tokenizer:
        # spm.SentencePieceTrainer.train(
        #     input=args.tokenizer_input,
        #     model_prefix=args.tokenizer_prefix,
        #     num_threads=args.num_threads,
        #     vocab_size=args.vocab_size,
        #     pad_id=0,
        #     unk_id=1,
        #     bos_id=2,
        #     eos_id=3,
        #     max_sentence_length=1_000_000,
        #     hard_vocab_limit=False,
        #     model_type="word",
        #     )

        import io
        import sentencepiece as spm
        from lm_dataformat import Reader

        rdr = Reader(args.tokenizer_input)
        sentence_iterator = rdr.stream_data()

        tokenizer = StringTokenizer()
        sentence_iterator = tokenizer.yield_split(rdr.stream_data())

        model = io.BytesIO()
        spm.SentencePieceTrainer.train(
            model_prefix=args.tokenizer_prefix,
            num_threads=args.num_threads,
            vocab_size=args.vocab_size,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            max_sentence_length=1_000_000,
            hard_vocab_limit=False,
            model_type="word",
            sentence_iterator=sentence_iterator,
            # model_writer=model,
            )