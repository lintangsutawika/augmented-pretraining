import os
import seqio
import argparse

import jax
import jax.numpy as jnp

from perplexity_sampling import task
from perplexity_sampling import util
from perplexity_sampling import counter

from tqdm import tqdm
from functools import partial
from jax_smi import initialise_tracking
initialise_tracking()

import tensorflow as tf

from jax.experimental.sparse import bcoo_reduce_sum

tf.config.experimental.set_visible_devices([], "GPU")
# tf.config.experimental.set_visible_devices([], "CPU")

# def PP(w):

#     def kneser_ney()

#     return 10**()

# def gaussian_sampling(W):

#     exponent = (-1/beta) * ((PP(W) - X)/X)**2
#     return alpha * e ** exponent

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.95'
os.environ["xla_force_host_platform_device_count"] = '64'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seq_length", default=512, type=int)
    parser.add_argument("--save_interval", default=500, type=int)
    parser.add_argument("--checkpoint_prefix", default="checkpoint_", type=str)
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

    k = 5
    vocab_size = 1_000_000
    cnt = counter.NGramCounter(vocab_size, k)
    matrix = cnt.init_matrix()

    increment_fn = jax.vmap(cnt.increment_at_coordinate, in_axes=[None, 0])
    ngram_fn = jax.vmap(util.ngrams, [0,None])
    pad_fn = jax.vmap(util.pad, [0, None])

    for idx, ex in tqdm(enumerate(dataset.as_numpy_iterator())):

        all_x = []
        for _k in [1,2,3,4,5]:
            x = ngram_fn(ex['text'], _k)
            x = jnp.reshape(x, (-1,_k))
            x = pad_fn(x, k)

            all_x.append(x)
        all_x = jnp.concatenate(all_x, axis=0)

        _matrix = increment_fn(cnt.init_matrix(), all_x)
        matrix = matrix + _matrix.sum(0)

        jax.clear_backends()

        if (idx > 0) and (idx%args.save_interval == 0):
            jnp.save("checkpoint_{}.npy".format(idx), matrix)
        # print("Num of ngram for {}\n Count: {}".format(x[0], cnt.get_value(matrix, x[0])))

    jnp.save("checkpoint_{}_finished.npy".format(idx), matrix)