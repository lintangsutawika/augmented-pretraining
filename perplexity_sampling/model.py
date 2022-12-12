import jax
import jax.numpy as jnp

from functools import partial

from perplexity_sampling import util

class StupidBackoffSmoothing:

    def __init__(
        self, matrix, k, N, alpha=0.4
        ):

        self.matrix = matrix
        self.k = k
        self.N = N #8700*512*4
        self.alpha = alpha

    # @partial(jax.jit, static_argnums=(0,4,5))
    def S(self, w_seq, idx, k=None, offset=1, alpha=0.4):

        # k = k if k else self.k
        alpha = alpha if alpha else self.alpha

        # f(w_{i-k+1}^{i})
        numerator = self.freq(w_seq, idx-k+offset, idx)
        # print("Count Numerator")
        # print(
        #     "from idx: {} to idx: {}, numerator: {}".format(
        #         idx-k+offset, idx, numerator
        #     )
        # )

        if numerator > 0:
            # print("Count Denominator")
            if (idx) == (idx-k+offset):
                denominator = self.N
            else:
                # f(w_{i-k+1}^{i-1})
                denominator = self.freq(w_seq, idx-k+offset, idx-1)

            return numerator/denominator
        else:
            offset += 1
            return alpha * self.S(w_seq, idx, k, offset)


    # @partial(jax.jit, static_argnums=(0,4))
    # @partial(jax.jit, static_argnums=(0,2,3,4))
    def freq(self, w_seq, from_idx, to_idx, k=None):

        k = k if k else self.k

        ngram = util.get_slice(w_seq, from_idx, to_idx)

        # print(
        #     "CORRECTED: from idx: {} to idx: {}".format(
        #         from_idx, to_idx
        #     )
        # )            
        # print("Current ngram: {}".format(ngram))

        if type(ngram) == list:
            ngram = jnp.asarray(ngram)

        count = util.get_value(self.matrix, ngram, k)
        return count

    # @partial(jax.jit, static_argnums=(0))
    def score(self, w_seq, k):

        k = k if k else self.k

        if type(w_seq) == list:
            w_seq = jnp.asarray(w_seq)

        log_score = 0
        for idx in range(0, len(w_seq)):
            # w_i | w_{i-k+1}^{i}
            log_score += jnp.log10(self.S(w_seq, idx, k=k))

        return log_score

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str)
    parser.add_argument("--k", default=5, type=int)
    parser.add_argument("--N", default=25_000_000, type=int)
    args = parser.parse_args()

    sentence = jnp.asarray([64, 10, 8, 6224, 35, 542, 71, 66, 11916])
    matrix = jnp.load(args.matrix, allow_pickle=True).tolist()
    sbs = StupidBackoffSmoothing(matrix=matrix, k=args.k, N=args.N)
    log_score = sbs.score(sentence, k=5)

    print("log_score: {}".format(log_score))