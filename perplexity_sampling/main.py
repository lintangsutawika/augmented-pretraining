import os
import seqio
import argparse

import jax
import jax.numpy as jnp

from perplexity_sampling import task
from perplexity_sampling import util
from perplexity_sampling import model

from tqdm import tqdm
from functools import partial
from jax_smi import initialise_tracking
initialise_tracking()

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")

# def PP(w):
#     sbs = StupidBackoffSmoothing(matrix=matrix, k=args.k, N=args.N)
#     log_score = sbs.score(w, k=5)
#     return 10**(-log_score)

# def gaussian_sampling(alpha, beta, W, X):

#     exponent = (-1/beta) * jnp.power(((PP(W) - X)/X), 2)
#     return alpha * jnp.exp(exponent)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.95'

if __name__ == '__main__':

    import sentencepiece as spm

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--build_matrix", type=bool)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seq_length", default=512, type=int)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--checkpoint_prefix", default="checkpoint_", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--matrix_path", default=None, type=str)
    parser.add_argument("--ngram", default=5, type=int)
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

        def get_k_ngrams(sequence, k):

            for _k in range(1, k+1):
                x = ngram_fn(sequence, _k)
                x = jnp.reshape(x, (-1,_k))
                x = pad_fn(x, k)

                if _k == 1:
                    all_x = x
                else:
                    all_x = jnp.concatenate([all_x, x], axis=0)

            return all_x

        get_k_ngram_fn = jax.pmap(
            partial(get_k_ngrams, k=k),
            in_axes=(0)
        )

        total_tokens = 0
        for idx, ex in tqdm(enumerate(dataset.as_numpy_iterator())):

            seq = ex['text']
            _, seq_length = seq.shape

            num_devices = len(jax.devices())
            seq = seq.reshape(num_devices, -1, seq_length)

            total_tokens += seq[jnp.where(seq != 0)].shape[0]
            
            all_x = get_k_ngram_fn(seq)
            all_x = jnp.reshape(all_x, (-1, k))
            all_x = all_x[jnp.where(all_x.sum(1) != 0)]

            matrix = matrix + increment_fn(matrix, all_x, k).sum(0)
            jax.clear_backends()

            if (idx > 0) and (idx%args.save_interval == 0):
                jnp.save("checkpoint_{}.npy".format(idx), matrix)

        jnp.save("checkpoint_{}_finished.npy".format(idx), matrix)

    else:

        matrix = jnp.load(args.matrix_path, allow_pickle=True).tolist()
        sbs = model.StupidBackoffSmoothing(matrix=matrix, k=args.ngram, N=18693964)

        # sbs.score(jnp.zeros(num_device, bs, 8, k))

        # fn = jax.vmap(partial(sbs.score, k=k), (0))

        for idx, ex in tqdm(enumerate(dataset.as_numpy_iterator())):

            seq = ex['text']
            _, seq_length = seq.shape

            num_devices = len(jax.devices())
            # seq = seq.reshape(num_devices, -1, seq_length)
            # seq = jnp.expand_dims(seq[:,0,:10], 1)

            # log_scores = sbs.score(sentence, k)
            break

        # fn(sentence)
