import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.image as mpimg
import tensorflow as tf
import grad

if __name__ == "__main__":
  lam = tf.Variable(tf.constant(100, dtype=tf.double, shape=[2]), dtype=tf.double)
  A, B = tf.unstack(lam, axis=0)
  # check the graph with its op name
  print(tf.get_default_graph().as_graph_def())