import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from functools import partial

@partial(jax.jit, static_argnames=['n'])
def ngrams(sequence, n):

    # assert len(sequence) >= n

    # if type(sequence) == list:
    #     sequence = jnp.asarray(sequence)

    history = jax.lax.slice(sequence, [0], [n])
    sequence = jax.lax.slice(sequence, [n], [len(sequence)])

    def _f(init, x):
        y = jnp.append(init, x)
        return y[1:], y[1:]
    
    _, ys = jax.lax.scan(_f, history, sequence)

    return jnp.append(jnp.expand_dims(history, 0), ys, axis=0)


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

def pad(sequence, n):

    if len(sequence) < n:
        if type(sequence) == list:
            sequence = jnp.asarray(sequence)

        indices = jnp.append(sequence, jnp.array([0]*(n-len(sequence)), dtype=jnp.int32))
        return indices
    else:
        return sequence


vmap_ngram = jax.vmap(ngrams, [0,None])
pad_fn = jax.vmap(pad, [0, None])

def get_k_ngrams(sequence, k):

    for _k in range(1, k+1):
        x = vmap_ngram(sequence, _k)
        x = jnp.reshape(x, (-1,_k))
        x = pad_fn(x, k)

        if _k == 1:
            all_x = x
        else:
            all_x = jnp.concatenate([all_x, x], axis=0)

    return all_x



# @partial(jax.jit, static_argnums=(1,2,))
# @partial(jax.vmap, in_axes=(0, None, None))
def get_slice(sequence, from_idx, to_idx):

    print("before")
    print("from_idx", from_idx)
    print("to_idx", to_idx)
    from_idx = jnp.array(from_idx * ~jnp.less(from_idx, 0), int)
    to_idx = jnp.array(to_idx * ~jnp.less(to_idx, 0), int)

    # from_idx = jnp.expand_dims(from_idx, 0)

    # def true_fun(*operands):
    #     return (1000)

    # def false_fun_from_idx(*operands):
    #     return (from_idx)

    # def false_fun_to_idx(*operands):
    #     return (to_idx)

    # from_idx = jax.lax.cond(
    #     from_idx < 0,
    #     true_fun,
    #     false_fun_from_idx,
    #     [zero, from_idx]
    # )

    # to_idx = jax.lax.cond(
    #     to_idx < 0,
    #     true_fun,
    #     false_fun_to_idx,
    #     [to_idx]
    # )


    print("after")
    print("from_idx", from_idx)
    print("to_idx", to_idx)
    return jax.lax.dynamic_slice(sequence, (from_idx,), (to_idx-from_idx+1,))