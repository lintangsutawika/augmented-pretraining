import jax
import jax.numpy as jnp

from functools import partial
from jax.experimental.sparse import BCOO

# ~5x faster than NLTK on CPU.
# TODO: Always 1 ngram left behind
@partial(jax.jit, static_argnames=['n'])
def ngrams(sequence, n):

    history = jax.lax.slice(sequence, [0], [n])
    sequence = jax.lax.slice(sequence, [n], [len(sequence)])

    def _f(init, x):
        y = jnp.append(init, x)
        return y[1:], y[1:]
    
    _, ys = jax.lax.scan(_f, history, sequence)

    return ys


class NGramCounter:

    def __init__(self, num_vocab, k) -> None:
        self.num_vocab = num_vocab
        self.k = k

    def init_matrix(self, num_vocab=None, k=None):

        num_vocab = num_vocab if num_vocab else self.num_vocab
        k = k if k else self.k

        seed_matrix = jnp.array([0])
        root_coordinate = jnp.array([[0]*k])
        matrix = BCOO((seed_matrix, root_coordinate), shape=tuple([num_vocab]*k))

        return matrix

    # set arg 'self' and 'k' as static
    @partial(jax.jit, static_argnums=(0,3))
    def increment_at_coordinate(self, matrix, ngram, k=None):

        if type(ngram) == list:
            ngram = jnp.asarray(ngram)

        k = k if k else self.k

        assert k >= len(ngram)
        indices = jnp.append(jnp.expand_dims(ngram, 0), jnp.array([[0]*(k-len(ngram))]), axis=1)
        increment = jnp.array([1])

        matrix = matrix + BCOO((increment, indices), shape=matrix.shape)

        return matrix

    # @partial(jax.jit, static_argnames=['k'])
    def get_value(self, matrix, ngram, k=None):

        if type(ngram) == list:
            ngram = jnp.asarray(ngram)

        k = k if k else self.k
        assert k >= len(ngram)
        indices = jnp.append(ngram, jnp.array([0]*(k-len(ngram))))

        def _f(element, token):
            return element[int(token)], 0
            # return element[jnp.array(token, int)], 0
            # return element[token], 0
        
        element = matrix
        for token in indices:
            element, _ = _f(element, token)
        # element, _ = jax.lax.scan(_f, matrix, indices)

        return element.todense()

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




