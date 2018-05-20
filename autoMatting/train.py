import os
import scipy as scp
import scipy.misc

import numpy as np
import logging
import tensorflow as tf
import sys

from FCN8s import FCN8s, logger
from data import BatchDatset
import utils


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "5", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 2
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800

def inference(image):
    FCN8s.logger.info("inference")

    # FCN
    net = FCN8s()
    with tf.name_scope("FCN8s"):
        net.build(image, debug=True)
    fub = tf.softmax(net.upscore32, axis=3)

    # Matting
    matting_module = tf.load_op_library("matting.so")
    with tf.name_scope("matting"):
        init = tf.constant_initializer(value=100.0, dtype=tf.double)
        lamb = tf.get_variable(name="lambda", initializer=init, shape=[1])
        pred_annotation = matting_module.matting(image, fub, lamb)

    return pred_annotation

def main():
    keep_probability = tf.placeholder(tf.double, name="keep_probabilty")
    image = tf.placeholder(tf.double, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 6], name="input_image")
    annotation = tf.placeholder(tf.double, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation = inference(image)

    matting_loss_module = tf.load_op_library("loss.so")
    loss = matting_loss_module.matting_loss(pred_annotation, annotation)
    loss_mean = tf.reduce_mean(loss)

    init = tf.global_variables_initializer()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss_mean)

    # print("Setting up summary op...")
    summary_op = tf.merge_all_summaries()

    '''
    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    '''
    train_dataset_reader = BatchDatset('data/trainlist.mat', batch_size=FLAGS.batch_size)

    with tf.Session() as sess:
        # initializing parameters
        sess.run(init)

        print("Setting up Saver...")
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

        # restore the parameters
        # sess.run(tf.initialiBatchDatsetze_all_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")
            itr = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        else:
            itr = 0
        # if FLAGS.mode == "train":
        # itr = 0
        train_images, train_annotations = train_dataset_reader.next_batch()
        trloss = 0.0
        while len(train_annotations) > 0:
            # train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            # print('==> batch data: ', train_images[0][100][100], '===', train_annotations[0][100][100])
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.5}
            _, summary_str, rloss = sess.run([optimizer, summary_op, loss], feed_dict=feed_dict)
            trloss += rloss

            if itr % 100 == 0:
                # train_loss, rpred = sess.run([loss, pred_annotation], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%f" % (itr, trloss / 100))
                trloss = 0.0
                saver.save(sess, FLAGS.logs_dir + "plus_model.ckpt", itr)
                summary_writer.add_summary(summary_str, itr)

            # if itr % 10000 == 0 and itr > 0:
            '''
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                           keep_probability: 1.0})
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))'''
            itr += 1

            train_images, train_annotations = train_dataset_reader.next_batch()
        saver.save(sess, FLAGS.logs_dir + "plus_model.ckpt", itr)


def test():
    net = FCN8s()
    print(net.data_dict["conv1_1"][0].shape)


if __name__ == '__main__':
    # main()
    test()