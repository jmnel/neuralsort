import numpy as np
import torch
import torch.nn.functional as F

import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

#import tf.nn

p_true = torch.eye(5).unsqueeze(0)
p_hat = torch.tensor([[[0.1970, 0.2009, 0.1993, 0.2016, 0.2012],
                       [0.1982, 0.2007, 0.1997, 0.2005, 0.2009],
                       [0.1993, 0.2006, 0.2002, 0.1994, 0.2005],
                       [0.2004, 0.2004, 0.2007, 0.1984, 0.2001],
                       [0.2016, 0.2002, 0.2012, 0.1973, 0.1997]]])

#p_hat = torch.randn((5, 5)).unsqueeze(0)
# print(p_hat)

p_true_tf = tf.constant(p_true.tolist())
p_hat_tf = tf.constant(p_hat.tolist())

#p_true_tf = tf.convert_to_tensor(p_true_tf)
#p_hat_tf = tf.convert_to_tensor(p_hat_tf)

tf_l = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=p_true_tf, logits=tf.log(tf.add(p_hat_tf, 1e-20)), dim=2)

# print(tf_l)

y_hat_softmax = tf.nn.softmax(p_hat_tf)
tl = -tf.reduce_sum(p_true_tf * tf.log(y_hat_softmax), [1])

foo = F.softmax(p_hat, dim=1)
bar = -torch.sum(p_true * torch.log(foo), dim=1)

print(tl)
print(bar)
# print(tl)
# print(bar)

# foo =
# print(foo)
# print(losses.numpy())

#foo = np.exp(p_hat.numpy())
#colsum = np.sum(foo, axis=1)
# print(colsum)

#sm = foo / colsum

# print(sm)
#ce = np.sum(np.multiply(-np.log(sm), p_hat.numpy()))

# print(ce)
#foo = F.softma
#foo = F.log_softmax(p_hat+1e-20, 2)
#foo = F.softmax(p_hat, -1)
#bar = (-torch.sum(p_true * foo, 2))
#bar = -sum(p_true * torch.log(foo), 1)
# print(foo)
# print(bar)

#sm = F.softmax(p_hat+1e-20)
#ce = F.cross_entropy(sm, p_true)

# print(ce)

#logits = torch.log(p_hat+1e-20)
#loss = -p_true + F.log_softmax(logits, 1)
#loss = torch.mean(loss, -2)
#loss = torch.sum(-p_true + F.log_softmax(logits, -1), 2)

# print(loss)
# print(loss.mean())
#l = tf.reduce_mean(losses).numpy()

# print(l)
