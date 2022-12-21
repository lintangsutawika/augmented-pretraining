import jax
import jax.numpy as jnp

from functools import partial

from perplexity_sampling import util

@partial(jax.jit, static_argnames=['alpha', 'seq_length', 'k'])
def get_alpha_matrix(alpha, k, seq_length):
    exponent_matrix = jnp.expand_dims(jnp.flip(jnp.arange(k)), 1).repeat(seq_length, 1)
    base_matrix = jnp.ones((k, seq_length)) * alpha

    return jnp.power(base_matrix, exponent_matrix)


class StupidBackoffSmoothing:

    def __init__(
        self, matrix, k, N, alpha=0.4, micro_bs=8
        ):

        self.matrix = matrix
        self.k = k
        self.N = N
        self.alpha = alpha
        self.micro_bs = micro_bs

    @partial(jax.jit, static_argnums=(0,))
    def score(self, w_seq):

        k = self.k
        N = self.N
        alpha = self.alpha
        matrix = self.matrix
        micro_bs = self.micro_bs
        
        pmap_get_k_ngram = jax.pmap(
            partial(util.get_k_ngrams, k=k),
            in_axes=(0)
        )

        bs, seq_length = w_seq.shape

        @partial(jax.jit, static_argnames=['k', 'seq_length'])
        def pad_ngram_table(carry, k, seq_length):

            carry = util.get_k_ngrams(carry, k)

            idx = 0
            for _k in range(k):

                part_a = jax.lax.slice(carry, [0,0], [idx+seq_length-_k,k])
                part_b = jax.lax.slice(carry, [idx+seq_length-_k,0], [carry.shape[0],k])
                part_insert = jnp.zeros((_k,k), dtype=jnp.int32)
                idx = idx + (seq_length-_k) + part_insert.shape[0]

                carry = jnp.concatenate([part_a, part_insert, part_b], 0)
            
            return carry

        fn_pad_ngram_table = partial(pad_ngram_table, k=k, seq_length=seq_length)
        vmap_pad_ngram_table = jax.vmap(
            fn_pad_ngram_table,
            in_axes=(0)
            )
        pmap_pad_ngram_table = jax.pmap(
            vmap_pad_ngram_table,
            in_axes=(0)
            )

        num_device = jax.device_count()
        seq = w_seq.reshape(num_device, -1, 1, seq_length)
        padded_seq_ngrams = pmap_pad_ngram_table(seq).reshape(num_device, -1, k)

        # return padded_seq_ngrams

        get_index_fn = partial(util.get_by_multiplication_jit, matrix=matrix)
        vmap_get_index = jax.vmap(
            get_index_fn,
            in_axes=(0)
            )

        pmap_get_index = jax.pmap(
            vmap_get_index,
            in_axes=(0)
            )

        def process_index(indices):
            return jax.lax.map(vmap_get_index, indices)


        pmap_process_index = jax.pmap(
            process_index,
            in_axes=(0)
            )

        # jax.clear_backends()
        xs = padded_seq_ngrams.reshape(num_device, -1, micro_bs, k) # bs, -1, k -> num_device, -1, 2, k
        # return xs

        # score_table = jnp.logical_and(score_table, w_seq.reshape((bs, 1, seq_length)).repeat(k, 1))
        score_table = pmap_process_index(xs).reshape(bs, k, -1)
        return score_table

        # score_table = pmap_get_index().reshape(k, -1)

        # # Todo, need to make this more efficient
        # score_table = jnp.stack(
        #     [util.get_by_indexing_jit(matrix, x) for x in padded_seq_ngrams],
        #     axis=0,
        #     dtype=jnp.float32
        #     ).reshape(k, -1)

        denominator = jnp.roll(score_table, 1, 0)
        denominator = denominator.at[0,:].set(N)

        alpha_matrix = get_alpha_matrix(alpha, k, seq_length)
        score_table = jnp.multiply(score_table, alpha_matrix)

        score = jnp.nanmax(score_table, axis=0)

        # return score
        return jnp.log10(score).sum() * (-1/seq_length)

    # # @partial(jax.jit, static_argnums=(0))
    # def score(self, w_seq, k):

    #     N = len(w_seq)
    #     log_score = 0
    #     for idx in range(0, N):
    #         # w_i | w_{i-k+1}^{i}
    #         log_score += jnp.log10(self.S(w_seq, idx, k=k))

    #     return log_score * (-1/N)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--N", default=25_000_000, type=int)
    args = parser.parse_args()
    # 18693964
    # sentence = jnp.asarray([64, 10, 8, 6224, 35, 542, 71, 66, 11916])
    sentence = jnp.asarray([67306, 12793, 6880, 4977, 259, 14037, 263, 259, 262, 259])

    matrix = jnp.load(args.matrix, allow_pickle=True).tolist()
    sbs = StupidBackoffSmoothing(matrix=matrix, k=args.k, N=args.N)
    # log_score = sbs.score(sentence, k=5)

    # print("log_score: {}".format(log_score))