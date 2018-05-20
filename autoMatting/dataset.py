import os
import tensorflow as tf
import numpy as np
import scipy as sp
import scipy.misc
from PIL import Image

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 800

# TFRecorde
def creatTFRecord(path):
    # filename = os.listdir(path)
    writer = tf.python_io.TFRecordWriter("train.tfrecords")

    image_list, alpha_list = get_file(path)

    for image_name, alpha_name in [image_list, alpha_list]:
        example = tf.train.Example(features=tf.train.Feature(features={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_name])),
            "alpha": tf.train.Feature(bytes_list=tf.train.BytesList(value=[alpha_name]))}))
        writer.write(example.SerializeToString())

    # for img_name in os.listdir(path):
    #     img_path = path + os.sep + img_name
    #     img = Image.open(img_name)
    #     img_raw = img.tobytes()
    #     example = tf.train.Example(features=tf.train.Feature(features={
    #         "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
    #     writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename, img_width, img_height, batch_size, capacity):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "iamge": tf.FixedLenFeature([], tf.string),
        "alpha": tf.FixedLenFeature([], tf.string)})

    # img = tf.decode_raw(features['iamge'], tf.uint8)
    # img = tf.reshape(img, [IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    # img = tf.cast(img, tf.double)/255
    # label = tf.cast(features['iamge'], tf.string)
    image_contents = tf.read_file(features['iamge'])
    alpha_contents = tf.read_file(features['alpha'])
    image = tf.image.decode_png(image_contents, channels=3)
    alpha = tf.image.decode_png(alpha_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    alpha = tf.image.resize_image_with_crop_or_pad(alpha, img_width, img_height)
    # image = tf.image.per_image_standardization(image)
    # alpha = tf.image.per_image_standardization(alpha)

    image_batch, alpha_batch = tf.train.shuffle_batch([image, alpha],
                                                      batch_size=batch_size,
                                                      num_threads=1,
                                                      min_after_dequeue=3,
                                                      capacity=capacity)

    return image_batch, alpha_batch

def readTFRecord(filename):
    img = read_and_decode(filename)
    img_batch = tf.train.shuffle_batch([img], batch_size=1,
                                       capacity=10,
                                       min_after_dequeue=6)

    return img_batch

# filename
def get_file(file_dir):
    images = []
    alphas = []

    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            # print(name)
            if(name.find("matte")!=-1):
                images.append(os.path.join(root, name))
            else:
                alphas.append(os.path.join(root, name))

    # print(len(images))
    # print(len(alphas))
    if(len(images)!=len(alphas)):
        print("data not matching!!!")
        image_list = []
        alpha_list = []
    else:
        tmp = np.array([images, alphas])
        tmp = tmp.transpose()
        np.random.shuffle(tmp)

        image_list = list(tmp[:, 0])
        alpha_list = list(tmp[:, 1])

    return image_list, alpha_list

def get_batch(image_list, alpha_list, img_width, img_height, batch_size, capacity):
    iamge = tf.cast(image_list, tf.string)
    alpha = tf.cast(alpha_list, tf.string)

    input_queue = tf.train.slice_input_producer([iamge, alpha])

    image_contents = tf.read_file(input_queue[0])
    alpha_contents = tf.read_file(input_queue[1])
    image = tf.image.decode_png(image_contents, channels=3)
    alpha = tf.image.decode_png(alpha_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    alpha = tf.image.resize_image_with_crop_or_pad(alpha, img_width, img_height)
    image = tf.image.per_image_standardization(image)
    alpha = tf.image.per_image_standardization(alpha)

    image_batch, alpha_batch = tf.train.shuffle_batch([image, alpha],
                                 batch_size=batch_size,
                                 num_threads=1,
                                 min_after_dequeue=3,
                                 capacity=capacity)

    return image_batch, alpha_batch

def test():
    img, alpha = get_file("F:/dataset/training")
    img_batch, alpha_batch = get_batch(img, alpha, IMAGE_WIDTH, IMAGE_HEIGHT, 5, 50)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image = sess.run(img_batch)

        # sp.misc.imsave("test.png",image[0,:,:,:])
        # print(type(image[0,0,0,0]))
        print(image.shape)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    test()
    # get_file("F:/dataset/training")