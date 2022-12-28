import jax
import jax.numpy as jnp

from functools import partial

from perplexity_sampling import util

@partial(jax.jit, static_argnames=['alpha', 'seq_length', 'k'])
def get_alpha_matrix(alpha, k, seq_length):
    exponent_matrix = jnp.expand_dims(jnp.flip(jnp.arange(k)), 1).repeat(seq_length, 1)
    base_matrix = jnp.ones((k, seq_length)) * alpha

    return jnp.power(base_matrix, exponent_matrix)


def shift_left(matrix, i):

    return jnp.multiply(matrix, jnp.roll(matrix, -1*i))


class StupidBackoffSmoothing:

    def __init__(
        self, matrix, k, N, alpha=0.4
        ):

        self.matrix = matrix
        self.k = k
        self.N = N
        self.alpha = alpha

        self.get_ngram_count = partial(util.get_ngram_count, matrix=matrix)

    # @partial(jax.jit, static_argnums=(0,))
    def score(self, seq):

        k = self.k
        N = self.N
        alpha = self.alpha
        matrix = self.matrix
        
        bs, seq_length = seq.shape

        fn_get_k_ngrams = partial(util.get_k_ngrams, k=k)
        pmap_pad_ngram_table = jax.pmap(
            fn_get_k_ngrams,
            in_axes=(0)
            )

        num_device = jax.device_count()
        _seq = seq.reshape(num_device, -1, seq_length)
        padded_seq_ngrams = pmap_pad_ngram_table(_seq).reshape(-1, k)

        # score_table = jnp.stack([self.get_ngram_count(i) for i in padded_seq_ngrams], 0).reshape(bs, k, -1)
        score_table = jnp.asarray([self.get_ngram_count(i) for i in padded_seq_ngrams]).reshape(bs, k, -1)

        base = (seq != 0).sum(1).astype(jnp.int32)
        # mask = jnp.stack(jnp.asarray([[[1]*(sl-i)+[0]*(seq_length-sl+i) for i in range(k)] for sl in base.tolist()]), 0)
        mask = jnp.asarray([[[1]*(sl-i)+[0]*(seq_length-sl+i) for i in range(k)] for sl in base.tolist()])
        score_table = jnp.multiply(score_table, mask)

        denominator = jnp.roll(score_table, 1, 1)
        denominator = denominator.at[:, 0,:].set(N)
        score_table = jnp.divide(score_table, denominator)

        alpha_matrix = get_alpha_matrix(alpha, k, seq_length)
        score_table = jnp.multiply(score_table, alpha_matrix)

        score = jnp.nanmax(score_table, axis=1)

        score = jnp.log10(score)
        score = jnp.nan_to_num(score, posinf=0.0, neginf=0.0)

        return - score.sum(1) / base

    # def calculate_log_score(score_table, alpha, k, seq_length):
    #     alpha_matrix = get_alpha_matrix(alpha, k, seq_length)
    #     score_table = jnp.multiply(score_table, alpha_matrix)

    #     score = jnp.nanmax(score_table, axis=1)

    #     score = jnp.log10(score)
    #     score = jnp.nan_to_num(score, posinf=0.0, neginf=0.0)

    #     return score

    # # @partial(jax.jit, static_argnums=(0))
    # def score(self, seq, k):

    #     N = len(seq)
    #     log_score = 0
    #     for idx in range(0, N):
    #         # w_i | w_{i-k+1}^{i}
    #         log_score += jnp.log10(self.S(seq, idx, k=k))

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