import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from functools import partial

@partial(jax.jit, static_argnames=['n', 'k'])
def get_ngrams(sequence, n, k):

    history = jax.lax.slice(sequence, [0], [n])
    sequence = jax.lax.slice(sequence, [n], [len(sequence)])

    def _f(init, x):
        y = jnp.append(init, x)
        return y[1:], y[1:]
    
    _, ys = jax.lax.scan(_f, history, sequence)

    ngram = jnp.append(jnp.expand_dims(history, 0), ys, axis=0)
    ngram = jnp.reshape(ngram, (-1,n))
    ngram = pad_fn(ngram, k)

    return ngram


vmap_get_ngrams = jax.vmap(get_ngrams, [0, None, None])


def get_k_ngrams(sequence, k):

    num, *_ = sequence.shape

    return jnp.concatenate(
        # [ngram_fn(sequence, n, k) for n in range(1,k+1)],
        [jnp.concatenate(
            [vmap_get_ngrams(sequence, n, k), jnp.zeros((num, n-1, k), dtype=jnp.int32)], axis=1
            ) for n in range(1,k+1)],
        axis=1
    )


def init_matrix(num_vocab, k):

    seed_matrix = jnp.array([1])
    root_coordinate = jnp.array([[0]*k])
    matrix = BCOO((seed_matrix, root_coordinate), shape=tuple([num_vocab]*k))

    return matrix


@partial(jax.jit, static_argnames=['k'])
def increment_at_coordinate(matrix, ngram, k):

    if type(ngram) == list:
        ngram = jnp.asarray(ngram)

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

    return BCOO((increment, indices), shape=matrix.shape)


@partial(jax.jit, static_argnames=['k'])
def get_value(matrix, ngram, k):

    if type(ngram) == list:
        ngram = jnp.asarray(ngram)

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

    return jax.experimental.sparse.bcoo_multiply_sparse(matrix, x).sum()

## @partial(jax.jit, static_argnames=['matrix'])
def get_by_indexing(indices, matrix):

    i0, i1, i2, i3, i4 = indices

    return matrix[i0, i1, i2, i3, i4].todense()

get_by_indexing_jit = jax.jit(get_by_indexing)


def get_by_multiplication(indices, matrix):

    identity = jnp.array([1])
    x = BCOO((identity, jnp.expand_dims(indices, 0)), shape=matrix.shape)

    count = jax.experimental.sparse.bcoo_multiply_sparse(matrix, x).sum()
    # jax.clear_backends()
    return count

get_by_multiplication_jit = jax.jit(get_by_multiplication)


def get_ngram_count(indices, matrix):

    indices = indices.tolist()
    _m = matrix
    for j, i in enumerate(indices):
        if (j == (len(indices)-1)) or (i == 0):
            break
        elif i not in _m:
            return 0
        else:
            _m = _m[i]

    if -1 in _m:
        return int(_m[-1])
    else:
        return 0


def pad(sequence, n):

    if len(sequence) < n:
        if type(sequence) == list:
            sequence = jnp.asarray(sequence)

        indices = jnp.append(sequence, jnp.array([0]*(n-len(sequence)), dtype=jnp.int32))
        return indices
    else:
        return sequence

# vmap_ngram = jax.vmap(ngrams, [0,None])
pad_fn = jax.vmap(pad, [0, None])
