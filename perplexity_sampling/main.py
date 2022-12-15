import os
import seqio
import argparse

import jax
import jax.numpy as jnp

from perplexity_sampling import task
from perplexity_sampling import util
from perplexity_sampling import matrix

from tqdm import tqdm
from functools import partial
from jax_smi import initialise_tracking
initialise_tracking()

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

def PP(w):
    sbs = StupidBackoffSmoothing(matrix=matrix, k=args.k, N=args.N)
    log_score = sbs.score(w, k=5)
    return 10**(-log_score)

def gaussian_sampling(alpha, beta, W, X):

    exponent = (-1/beta) * jnp.power(((PP(W) - X)/X), 2)
    return alpha * jnp.exp(exponent)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.95'
os.environ["xla_force_host_platform_device_count"] = '64'

if __name__ == '__main__':

    import sentencepiece as spm

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seq_length", default=4096, type=int)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--checkpoint_prefix", default="checkpoint_", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--matrix_path", default=None, type=str)
    parser.add_argument("--ngram", default=5, type=int)
    parser.add_argument("--build_matrix", default=True, type=bool)
    args = parser.parse_args()

    seqio.add_global_cache_dirs(["/fsx/lintangsutawika/data/"])
    sequence_length = {"text": args.seq_length}
    dataset = seqio.get_mixture_or_task(args.task).get_dataset(
        sequence_length=sequence_length,
        split="train",
        shuffle=False,
        num_epochs=1,
        shard_info=seqio.ShardInfo(index=0, num_shards=10),
        use_cached=True,
        seed=42
    )

    dataset = seqio.utils.trim_and_pad_dataset(dataset, sequence_length)
    dataset = dataset.batch(args.batch_size, drop_remainder=True)

    tokenizer = spm.SentencePieceProcessor(model_file=args.vocab_path)

    k = args.ngram
    vocab_size = tokenizer.vocab_size()
    
    pad_fn = jax.vmap(util.pad, [0, None])

    if args.build_matrix:
        
        matrix = util.init_matrix(vocab_size, k)
        increment_fn = jax.vmap(util.increment_at_coordinate, in_axes=[None, 0, None])
        ngram_fn = jax.vmap(util.ngrams, [0,None])

        total_tokens = 0

        for idx, ex in tqdm(enumerate(dataset.as_numpy_iterator())):

            total_tokens += ex['text'][jnp.where(ex['text'] != 0)].shape[0]
            
            all_x = []
            for _k in range(1, k+1):
                x = ngram_fn(ex['text'], _k)
                x = jnp.reshape(x, (-1,_k))
                x = pad_fn(x, k)

                all_x.append(x)

            all_x = jnp.concatenate(all_x, axis=0)
            all_x = all_x[jnp.where(all_x.sum(1) != 0)]

            matrix = matrix + increment_fn(matrix, all_x, k).sum(0)
            jax.clear_backends()

            if (idx > 0) and (idx%args.save_interval == 0):
                jnp.save("checkpoint_{}.npy".format(idx), matrix)

        jnp.save("checkpoint_{}_finished.npy".format(idx), matrix)