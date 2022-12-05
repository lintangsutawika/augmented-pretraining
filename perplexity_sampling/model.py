import jax
import jax.numpy as jnp

from functools import partial

class StupidBackoffSmoothing:

    def __init__(
        self,
        k,
        alpha=0.4
        ):

        self.k = k
        self.alpha = alpha

    @partial(jit, static_argnums=(0,))
    def S(self, w, idx, k, offset=1, alpha=0.4):

        # f(w^(i)_(i-k+1))
        numerator = self.freq(w_seq, idx, idx-self.k+1)

        if numerator > 0:
            # f(w^(i-1)_(i-k+1))
            denominator = self.freq(w_seq, idx-1, idx-self.k+1)
            return numerator/denominator
        else:
            offset += 1
            alpha * self.S(w_seq, idx, self.k, offset)

        jax.lax.cond(
            numerator
        )

    def freq(w, i, L=None):
        return w[i:L]

    def sequence(w_seq, idx, L=None):

        if L == None:
            return w_seq[idx]
        else:
            return w_seq[idx:L]


    def score(w_seq):
        # use scan here.
        scores = jax.lax.scan(self.S, carry, range(len(w_seq)))
        # product of scores

        return scores

    def freq(w_seq)
