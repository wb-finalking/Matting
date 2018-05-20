from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
import tensorflow as tf

@ops.RegisterGradient("Matting")
def _matting_grad(op, grad):
  matting_module = tf.load_op_library('./matting.so')
  input = op.inputs[0]
  fb = op.inputs[1]
  # b = op.inputs[2]
  lamb = op.inputs[2]

  fb_grad, lamb_grad = matting_module.matting_grad(input, fb, lamb, grad)

  return [None, fb_grad, lamb_grad]

@ops.RegisterGradient("MattingLoss")
def _matting_loss_grad(op, grad):
  loss_module = tf.load_op_library('./loss.so')
  alpha = op.inputs[0]
  ground = op.inputs[1]

  alpha_grad = loss_module.matting_loss_grad(alpha, ground, grad)

  return [alpha_grad, None]

# @ops.RegisterGradient("Unpack")
# def _unpack_grad(op, grad):
#   list = tf.stack(grad)
#
#   return [list]