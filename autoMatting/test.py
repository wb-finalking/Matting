#!/usr/bin/env python

import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

import FCN
import dataset
import grad


IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# img1 = scp.misc.imread("./test_data/tabby_cat.png")
img1 = scp.misc.imread("./input.png")
ground = scp.misc.imread("./ground.png")
model_dir = "model/"

matting_module = tf.load_op_library('./matting.so')
loss_module = tf.load_op_library('./loss.so')
#
# img, alpha = dataset.get_file("F:/dataset/training")
# img_batch, alpha_batch = dataset.get_batch(img, alpha, IMAGE_WIDTH, IMAGE_HEIGHT, 1, 5)

with tf.Session() as sess:
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # imgs = sess.run(img_batch)
    # print(imgs.shape)
    images = tf.placeholder("float32")
    grounds = tf.placeholder("float32")
    batch_images = tf.expand_dims(images, 0)
    batch_ground = tf.expand_dims(grounds, 0)
    # vgg_fcn = FCN8s.FCN8s()
    vgg_fcn = FCN.FCN8s(load=False)
    with tf.name_scope("content_vgg"):
        vgg_fcn.build(batch_images, debug=True,load=False)

    with tf.name_scope("matting"):
        init = tf.constant_initializer(value=100,
                                       dtype=tf.float32)
        lam = tf.get_variable(name="lamb",initializer=init,
                              shape=[1],dtype=tf.float32)
        alpha = matting_module.matting(batch_images, vgg_fcn.softmax, lam)
    loss = tf.reduce_mean(loss_module.matting_loss(alpha, batch_ground))
    # loss = tensors-batch_ground
    Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    TrainOp = Optimizer.minimize(loss)
    print('Finished building Network.')

    # img = sess.run(img_batch)
    feed_dict = {images: img1,grounds: ground}

    # logging.info("Start Initializing Variabels.")
    # init = tf.global_variables_initializer()
    # sess.run(init)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    print('Running the Network')
    tensors = [TrainOp, vgg_fcn.softmax, alpha]
    _, softmax, res = sess.run(tensors, feed_dict=feed_dict)

    scp.misc.imshow(softmax[0])
    scp.misc.imshow(res[0,:,:])


    # saver.save(sess, model_dir + "model.ckpt")

    # coord.request_stop()
    # coord.join(threads)



