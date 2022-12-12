import jax
import jax.numpy as jnp

from functools import partial
from jax.experimental.sparse import BCOO


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

        increment = jnp.array([1])

        if len(ngram) < k:
            indices = jnp.append(jnp.expand_dims(ngram, 0), jnp.array([[0]*(k-len(ngram))], dtype=jnp.int32), axis=1, dtype=jnp.int32)
        else:
            indices = jnp.expand_dims(ngram, 0)

        return matrix + BCOO((increment, indices), shape=matrix.shape)

    # @partial(jax.jit, static_argnames=['k'])
    def get_value(self, matrix, ngram, k=None):

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
            # return element[jnp.array(token, int)], 0
            # return element[token], 0
        
        element = matrix
        for token in indices:
            element, _ = _f(element, token)
        # element, _ = jax.lax.scan(_f, matrix, indices)

        return element.todense()
