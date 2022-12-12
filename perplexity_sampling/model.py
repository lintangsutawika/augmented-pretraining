import jax
import jax.numpy as jnp

from functools import partial

class StupidBackoffSmoothing:

    def __init__(
        self, matrix, k, N, alpha=0.4
        ):

        self.matrix = matrix
        self.k = k
        self.N = N #8700*512*4
        self.alpha = alpha

    # @partial(jit, static_argnums=(0,))
    def S(self, w_seq, idx, k=None, offset=1, alpha=0.4):

        k = k if k else self.k
        alpha = alpha if alpha else self.alpha

        # f(w_{i-k+1}^{i})
        numerator = self.freq(w_seq, idx-k+offset, idx)

        print(
            "from idx: {} to idx: {}, numerator: {}".format(
                idx-k+offset, idx, numerator
            )
        )

        if numerator > 0:
            if (idx) == (idx-k+offset):
                denominator = self.N
            else:
                # f(w_{i-k+1}^{i-1})
                denominator = self.freq(w_seq, idx-1, idx-k+offset)

            return numerator/denominator
        else:
            offset += 1
            return alpha * self.S(w_seq, idx, k, offset)

    def freq(self, w_seq, from_idx, to_idx):

        if from_idx < 0:
            from_idx = 0
        
        if to_idx < 0:
            to_idx = 0
        to_idx += 1 #Offset by 1 for proper indexing in python

        ngram = w_seq[from_idx:to_idx]

        print("Current ngram: {}".format(ngram))

        if type(ngram) == list:
            ngram = jnp.asarray(ngram)

        k = k if k else self.k
        assert k >= len(ngram)
        if len(ngram) < k:
            indices = jnp.append(ngram, jnp.array([0]*(k-len(ngram))))
        else:
            indices = jnp.asarray(ngram)

        def _f(element, token):
            return element[int(token)], 0
        
        element = self.matrix
        for token in indices:
            element, _ = _f(element, token)

        return element.todense()


    def score(w_seq, k=None):

        k = k if k else self.k

        log_score = 0
        for idx in range(0, len(w_seq)):
            # w_i | w_{i-k+1}^{i}
            log_score += jnp.log10(self.S(w_seq, idx, k=k))

        # L = jnp.array(range(len(w_seq)))
        # for i 
        # score, _ = jax.lax.scan(self.S, score, L)

        return score

    # def freq(w_seq)
# In [114]: s = []
#      ...: for i in range(len(sentence)):
#      ...:     print(i)
#      ...:     s.append(sbs.S(sentence, i))
#      ...:     print(sp.decode([sentence[i]]))