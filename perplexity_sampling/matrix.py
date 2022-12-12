import jax
import jax.numpy as jnp

from functools import partial
from jax.experimental.sparse import BCOO


class NGramMatrix:

    def __init__(self, num_vocab, k) -> None:
        self.num_vocab = num_vocab
        self.k = k



    # set arg 'self' and 'k' as static
    @partial(jax.jit, static_argnums=(0,3))
    def increment_at_coordinate(self, matrix, ngram, k=None):

        if type(ngram) == list:
            ngram = jnp.asarray(ngram)

        k = k if k else self.k
        assert k >= len(ngram)

        increment = jnp.array([1])

        if len(ngram) < k:
            indices = jnp.append(
                jnp.expand_dims(ngram, 0),
                jnp.array([[0]*(k-len(ngram))], dtype=jnp.int32),
                axis=1, dtype=jnp.int32
                )
        else:
            indices = jnp.expand_dims(ngram, 0)

        return matrix + BCOO((increment, indices), shape=matrix.shape)

    @partial(jax.jit, static_argnums=(0,3))
    def get_value(self, matrix, ngram, k=None):

        if type(ngram) == list:
            ngram = jnp.asarray(ngram)

        k = k if k else self.k
        assert k >= len(ngram)

        if len(ngram) < k:
            indices = jnp.append(
                jnp.expand_dims(ngram, 0),
                jnp.array([[0]*(k-len(ngram))], dtype=jnp.int32),
                axis=1,
                # dtype=jnp.int32
                )
        else:
            indices = jnp.expand_dims(ngram, 0)
        
        identity = jnp.array([1])

        x = BCOO((identity, indices), shape=matrix.shape)

        return jax.experimental.sparse.bcoo_multiply_sparse(matrix, x)

    # # @partial(jax.jit, static_argnums=(0,3))
    # def get_value(self, matrix, ngram, k=None):

    #     x = self.get_identity(matrix, ngram, k)

    #     return _fn(matrix, x)

    # # @partial(jax.jit, static_argnames=['k'])
    # def get_value(self, matrix, ngram, k=None):

    #     if type(ngram) == list:
    #         ngram = jnp.asarray(ngram)

    #     k = k if k else self.k
    #     assert k >= len(ngram)
    #     if len(ngram) < k:
    #         indices = jnp.append(ngram, jnp.array([0]*(k-len(ngram))))
    #     else:
    #         indices = jnp.asarray(ngram)

    #     def _f(element, token):
    #         return element[int(token)], 0
    #         # return element[jnp.array(token, int)], 0
    #         # return element[token], 0
        
    #     element = matrix
    #     for token in indices:
    #         element, _ = _f(element, token)
    #     # element, _ = jax.lax.scan(_f, matrix, indices)

    #     return element.todense()

