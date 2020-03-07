import tensorflow as tf
import numpy as np
import torch


def _bl_matmul(A, B):
    return tf.einsum('mij,jk->mik', A, B)


def compute_permu_matrix(s, tau=1):
    A_s = s - tf.transpose(s, perm=[0, 2, 1])
    A_s = tf.abs(A_s)
    n = tf.shape(s)[1]
    one = tf.ones((n, 1))
    B = _bl_matmul(A_s, one @ tf.transpose(one))
    K = tf.range(n) + 1
    C = _bl_matmul(s, tf.expand_dims(
        tf.cast(n+1-2*K, dtype=tf.float32), 0))
    P = tf.transpose(C - B, perm=[0, 2, 1])
    P = tf.nn.softmax(P / tau, -1)
    print(P)


torch.manual_seed(0)
batch_size = 2
seq_len = 5
stitch_len = 4


s = tf.convert_to_tensor(torch.randn(batch_size, seq_len, 1).numpy())

foo = compute_permu_matrix(s)
