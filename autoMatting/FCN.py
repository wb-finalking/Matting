from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers

# setting log format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# setting log outstream
# file = logging.FileHandler("log/1.log")
# file.setLevel(logging.DEBUG)
# file.setFormatter(formatter)

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# console.setFormatter(formatter)
# add outstream to logger
# logger = logging.getLogger("FCN8s")
# logger.addHandler(console)
# logger.addHandler(file)

VGG_MEAN = [103.939, 116.779, 123.68]

class FCN8s:

    def __init__(self, vgg16_npy_path=None, load = False):

        if(load):
            if vgg16_npy_path is None:
                path = sys.modules[self.__class__.__module__].__file__
                # print path
                path = os.path.abspath(os.path.join(path, os.pardir))
                # print path
                path = os.path.join(path, "vgg16.npy")
                vgg16_npy_path = path
                logging.info("Load npy file from '%s'.", vgg16_npy_path)
            if not os.path.isfile(vgg16_npy_path):
                logging.error(("File '%s' not found. Download it from "
                               "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
                               "models/vgg16.npy"), vgg16_npy_path)
                sys.exit(1)

            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
            logging.info("npy file loaded")
        self.wd = 5e-4
        logging.info("init")


    def build(self, rgb, train=False, num_classes=3, load=False, random_init_fc8=False,
              debug=False, use_dilated=False, is_bn=True):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        # Convert RGB to BGR

        with tf.name_scope('Processing'):

            red, green, blue = tf.split(rgb, 3, 3)
        #     # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        #     # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        #     # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)
        #
        #     if debug:
        #         bgr = tf.Print(bgr, [tf.shape(bgr)],
        #                        message='Shape of input image: ',
        #                        summarize=4, first_n=1)
        # bgr = rgb
        self.conv1_1 = self._conv_layer(bgr, "conv1_1", [3,3,3,64], load, is_bn, train)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2", [3,3,64,64], load, is_bn, train)
        self.pool1 = self._max_pool(self.conv1_2, 'pool1', debug)

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1", [3,3,64,128], load, is_bn, train)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2", [3,3,128,128], load, is_bn, train)
        self.pool2 = self._max_pool(self.conv2_2, 'pool2', debug)

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1", [3,3,128,256], load, is_bn, train)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2", [3,3,256,256], load, is_bn, train)
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3", [3,3,256,256], load, is_bn, train)
        self.pool3 = self._max_pool(self.conv3_3, 'pool3', debug)

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1", [3,3,256,512], load, is_bn, train)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2", [3,3,512,512], load, is_bn, train)
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3", [3,3,512,512], load, is_bn, train)

        if use_dilated:
            pad = [[0, 0], [0, 0]]
            self.pool4 = tf.nn.max_pool(self.conv4_3, ksize=[1, 2, 2, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool4')
            self.pool4 = tf.space_to_batch(self.pool4,
                                           paddings=pad, block_size=2)
        else:
            self.pool4 = self._max_pool(self.conv4_3, 'pool4', debug)

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1", [3,3,512,512], load, is_bn, train)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2", [3,3,512,512], load, is_bn, train)
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3", [3,3,512,512], load, is_bn, train)
        if use_dilated:
            pad = [[0, 0], [0, 0]]
            self.pool5 = tf.nn.max_pool(self.conv5_3, ksize=[1, 2, 2, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='SAME', name='pool5')
            self.pool5 = tf.space_to_batch(self.pool5,
                                           paddings=pad, block_size=2)
        else:
            self.pool5 = self._max_pool(self.conv5_3, 'pool5', debug)

        self.fc6 = self._fc_layer(self.pool5, "fc6", shape=[7, 7, 512, 4096],
                                  load=load, is_bn=is_bn, train=train)


        if train:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)
        self.fc7 = self._fc_layer(self.fc6, "fc7", shape=[1, 1, 4096, 4096],
                                  load=load, is_bn=is_bn, train=train)
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        if use_dilated:
            self.pool5 = tf.batch_to_space(self.pool5, crops=pad, block_size=2)
            self.pool5 = tf.batch_to_space(self.pool5, crops=pad, block_size=2)
            self.fc7 = tf.batch_to_space(self.fc7, crops=pad, block_size=2)
            self.fc7 = tf.batch_to_space(self.fc7, crops=pad, block_size=2)
            return

        if random_init_fc8:
            self.score_fr = self._score_layer(self.fc7, "score_fr",
                                              num_classes, is_bn=is_bn, train=train)
        else:
            self.score_fr = self._fc_layer(self.fc7, "score_fr",
                                           shape = [1, 1, 4096, 1000],
                                           num_classes=num_classes,
                                           load = load,
                                           relu=False, is_bn=is_bn, train=train)

        # self.pred = tf.argmax(self.score_fr, dimension=3)

        self.upscore2 = self._upscore_layer(self.score_fr,
                                            shape=tf.shape(self.pool4),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore2',
                                            ksize=4, stride=2,
                                            is_bn = is_bn, train = train)
        self.score_pool4 = self._score_layer(self.pool4, "score_pool4",
                                             num_classes=num_classes,
                                             is_bn=is_bn, train=train)
        self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)

        self.upscore4 = self._upscore_layer(self.fuse_pool4,
                                            shape=tf.shape(self.pool3),
                                            num_classes=num_classes,
                                            debug=debug, name='upscore4',
                                            ksize=4, stride=2,
                                            is_bn=is_bn, train=train)
        self.score_pool3 = self._score_layer(self.pool3, "score_pool3",
                                             num_classes=num_classes,
                                             is_bn=is_bn, train=train)
        self.fuse_pool3 = tf.add(self.upscore4, self.score_pool3)

        self.upscore32 = self._upscore_layer(self.fuse_pool3,
                                             shape=tf.shape(bgr),
                                             num_classes=num_classes,
                                             debug=debug, name='upscore32',
                                             ksize=16, stride=8,
                                             is_bn=is_bn, train=train)

        # self.pred_up = tf.argmax(self.upscore32, dimension=3)

        self.softmax = tf.nn.softmax(self.upscore32, axis=3)


    #########################################################################
    ######################### convolutional layer ###########################
    #########################################################################

    def get_conv_filter(self, name, shape, load=False):
        if(load):
            init = tf.constant_initializer(value=self.data_dict[name][0],
                                           dtype=tf.float32)
            var = tf.get_variable(name="filter", initializer=init,shape=shape)
        else:
            init = tf.random_normal(shape=shape,
                                 mean=0.0,
                                 stddev=1.0,
                                 dtype=tf.float32,
                                 seed=None, name=None)
            var = tf.get_variable(name="filter", initializer=init)
        # init = tf.constant_initializer(value=0,
        #                                dtype=tf.float32)
        logging.info('Layer name: %s' % name)
        logging.info('Layer shape: %s' % str(shape))

        return var

    def _bias_reshape(self, bweight, num_orig, num_new):
        """ Build bias weights for filter produces with `_summary_reshape`

        """
        n_averaged_elements = num_orig//num_new
        avg_bweight = np.zeros(num_new)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
        return avg_bweight

    def get_bias(self, name, shape, load=False, num_classes=None):
        if(load):
            bias_wights = self.data_dict[name][1]
            if name == 'fc8':
                bias_wights = self._bias_reshape(bias_wights, bias_wights.shape[0],
                                                 num_classes)
            init = tf.constant_initializer(value=bias_wights,
                                            dtype=tf.float32)
        else:
            init = tf.constant_initializer(value=0,
                                           dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        # _variable_summaries(var)
        return var

    def _conv_layer(self, bottom, name, shape, load=False, is_bn=True, train=True):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(name, shape, load)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, shape[3], load=load)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            # _activation_summary(relu)
            if (is_bn):
                # return utils.bn_layer(relu, is_training)
                return layers.batch_norm(relu, scope='afternorm', is_training=train)
            else:
                return relu

    #########################################################################
    ############################## pool layer ###############################
    #########################################################################

    def _max_pool(self, bottom, name, debug):
        pool = tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

        if debug:
            pool = tf.Print(pool, [tf.shape(pool)],
                            message='Shape of %s' % name,
                            summarize=4, first_n=1)
        return pool

    #########################################################################
    ################### fc layer to convolutional layer #####################
    #########################################################################

    def _summary_reshape(self, fweight, shape, num_new):
        """ Produce weights for a reduced fully-connected layer.

        FC8 of VGG produces 1000 classes. Most semantic segmentation
        task require much less classes. This reshapes the original weights
        to be used in a fully-convolutional layer which produces num_new
        classes. To archive this the average (mean) of n adjanced classes is
        taken.

        Consider reordering fweight, to perserve semantic meaning of the
        weights.

        Args:
          fweight: original weights
          shape: shape of the desired fully-convolutional layer
          num_new: number of new classes


        Returns:
          Filter weights for `num_new` classes.
        """
        num_orig = shape[3]
        shape[3] = num_new
        print("num_orig:"+ str(num_orig))
        print("num_new:" + str(num_new))
        assert(num_new < num_orig)
        n_averaged_elements = num_orig//num_new
        avg_fweight = np.zeros(shape)
        for i in range(0, num_orig, n_averaged_elements):
            start_idx = i
            end_idx = start_idx + n_averaged_elements
            avg_idx = start_idx//n_averaged_elements
            if avg_idx == num_new:
                break
            avg_fweight[:, :, :, avg_idx] = np.mean(
                fweight[:, :, :, start_idx:end_idx], axis=3)
        return avg_fweight

    def get_fc_weight_reshape(self, name, shape, num_classes=None, load=False):
        if(load):
            weights = self.data_dict[name][0]
            weights = weights.reshape(shape)
            if num_classes is not None:
                weights = self._summary_reshape(weights, shape,
                                                num_new=num_classes)
            init = tf.constant_initializer(value=weights,
                                           dtype=tf.float32)
            var = tf.get_variable(name="filter", initializer=init, shape=shape)
        else:
            if num_classes is not None:
                shape[3] = num_classes
            init = tf.random_normal(shape=shape,
                                    mean=0.0,
                                    stddev=1.0,
                                    dtype=tf.float32,
                                    seed=None, name=None)
            var = tf.get_variable(name="filter", initializer=init)
        print('Layer name: %s' % name)
        print('Layer shape: %s' % shape)
        return var

    def _fc_layer(self, bottom, name, shape, load=False, num_classes=None,
                  relu=True, debug=False, is_bn = True, train = True):
        with tf.variable_scope(name) as scope:
            if name == 'score_fr':
                name = 'fc8'
                filt = self.get_fc_weight_reshape(name, shape, load=load,
                                                  num_classes=num_classes)
            else:
                filt = self.get_fc_weight_reshape(name, shape, load=load)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, shape[3], load=load,
                                        num_classes=num_classes)
            bias = tf.nn.bias_add(conv, conv_biases)

            if relu:
                bias = tf.nn.relu(bias)
            # _activation_summary(bias)

            if debug:
                bias = tf.Print(bias, [tf.shape(bias)],
                                message='Shape of %s' % name,
                                summarize=4, first_n=1)
            if (is_bn):
                # return utils.bn_layer(relu, is_training)
                return layers.batch_norm(bias, scope='afternorm', is_training=train)
            else:
                return bias

    #########################################################################
    ############################## score layer ##############################
    #########################################################################

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.

        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.

        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.

        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        # collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        # if wd and (not tf.get_variable_scope().reuse):
        #     weight_decay = tf.multiply(
        #         tf.nn.l2_loss(var), wd, name='weight_loss')
        #     tf.add_to_collection(collection_name, weight_decay)
        # _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        # _variable_summaries(var)
        return var

    def _score_layer(self, bottom, name, num_classes,is_bn=True, train=True):
        with tf.variable_scope(name) as scope:
            # get number of input channels
            in_features = bottom.get_shape()[3].value
            shape = [1, 1, in_features, num_classes]
            # He initialization Sheme
            if name == "score_fr":
                num_input = in_features
                stddev = (2 / num_input)**0.5
            elif name == "score_pool4":
                stddev = 0.001
            elif name == "score_pool3":
                stddev = 0.0001
            # Apply convolution
            w_decay = self.wd

            weights = self._variable_with_weight_decay(shape, stddev, w_decay,
                                                       decoder=True)
            conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
            # Apply bias
            conv_biases = self._bias_variable([num_classes], constant=0.0)
            bias = tf.nn.bias_add(conv, conv_biases)

            # _activation_summary(bias)

            if (is_bn):
                # return utils.bn_layer(relu, is_training)
                return layers.batch_norm(bias, scope='afternorm', is_training=train)
            else:
                return bias

    #########################################################################
    ############################# upscore layer #############################
    #########################################################################

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.double)
        var = tf.get_variable(name="up_filter", initializer=init,
                              shape=weights.shape)
        return var

    def _upscore_layer(self, bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2,is_bn=True, train=True):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            if shape is None:
                # Compute shape out of Bottom
                in_shape = tf.shape(bottom)

                h = ((in_shape[1] - 1) * stride) + 1
                w = ((in_shape[2] - 1) * stride) + 1
                new_shape = [in_shape[0], h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]
            output_shape = tf.stack(new_shape)

            logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]

            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            weights = self.get_deconv_filter(f_shape)
            # weights = tf.Print(weights, [weights],
            #                    message='weights: ',
            #                    summarize=20, first_n=5)
            # self._add_wd_and_summary(weights, self.wd, "fc_wlosses")
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')


            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)

            # _activation_summary(deconv)
            if (is_bn):
                # return utils.bn_layer(relu, is_training)
                return layers.batch_norm(deconv, scope='afternorm', is_training=train)
            else:
                return deconv



def main():
    net = FCN8s()

if __name__ == '__main__':
    main()