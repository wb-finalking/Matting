import numpy as np
import scipy as sp
from PIL import Image
import matplotlib.image as mpimg
import tensorflow as tf
import grad

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  # tf.test.main()
  # print(" ".join(tf.sysconfig.get_compile_flags()))
  # print(" ".join(tf.sysconfig.get_link_flags()))
  matting_module = tf.load_op_library('./matting.so')
  loss_module = tf.load_op_library('./loss.so')
  # zero_out_module = tf.load_op_library('./zero_out.so')
  input = sp.misc.imread("./input.png").astype(np.double)
  input=input;
  trimap = sp.misc.imread("./trimap.png").astype(np.double)
  trimap = trimap/255;
  trimap = trimap[:, :, 0]
  ground = sp.misc.imread("./ground.png").astype(np.double)
  ground = ground/255;
  # ground = ground[:, :, 0]
  # f = np.zeros([input.shape[0],input.shape[1]])
  # f[trimap > 0.98] = 1
  # b = np.zeros([input.shape[0], input.shape[1]])
  # b[trimap < 0.02] = 1

  lamb_in=np.zeros([2,1])
  input_in=np.zeros([2,input.shape[0],input.shape[1],input.shape[2]])
  trimap_in=np.zeros([2,trimap.shape[0],trimap.shape[1]])
  f_in = np.zeros([2, trimap.shape[0], trimap.shape[1]])
  b_in = np.zeros([2, trimap.shape[0], trimap.shape[1]])
  ground_in = np.zeros([2, ground.shape[0], ground.shape[1]])

  lamb_in[0,0]=100;
  input_in[0,:,:,:]=input
  trimap_in[0, :, :] = trimap
  # f_in[0, :, :] = f
  # b_in[0, :, :] = b
  ground_in[0, :, :] = ground
  lamb_in[1, 0] = 100;
  input_in[1, :, :, :] = input
  trimap_in[1, :, :] = trimap
  # f_in[1, :, :] = f
  # b_in[1, :, :] = b
  ground_in[1, :, :] = ground

  print(input.shape)
  print(ground.shape)
  img1=np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3).astype(np.float)
  with tf.Session() as sess:
    images = tf.placeholder("double")
    trimap_img = tf.placeholder("double")
    lamb = tf.placeholder("double")
    images_f = tf.placeholder("double")
    images_b = tf.placeholder("double")
    images_ground = tf.placeholder("double")

    batch_images = tf.expand_dims(images, 0)
    batch_trimaps = tf.expand_dims(trimap_img, 0)
    batch_lamb = tf.expand_dims(lamb, 0)
    batch_f = tf.expand_dims(images_f, 0)
    batch_b = tf.expand_dims(images_b, 0)
    batch_ground = tf.expand_dims(images_ground, 0)
    feed_dict = {batch_images: input_in, batch_f: f_in, batch_b: b_in,
                 batch_lamb: lamb_in, batch_ground: ground_in}

    # init = tf.global_variables_initializer()
    # sess.run(init)
    lam = tf.Variable(tf.constant(100, dtype=tf.double, shape=[1]), dtype=tf.double)
    # [A,_] = tf.unstack(lam,axis=0)
    # A, _ = tf.split(lam, num_or_size_splits=2, axis=0)
    # tf.reshape(A, [1])
    f = tf.Variable(tf.constant(0.5,dtype=tf.double,shape=[2,497,800]),dtype=tf.double)
    b = tf.Variable(tf.constant(0.5, dtype=tf.double, shape=[2, 497, 800]), dtype=tf.double)
    u = tf.Variable(tf.constant(0.5, dtype=tf.double, shape=[2, 497, 800]), dtype=tf.double)
    fb = tf.stack([f, u, b],axis=3)
    # tf.to_double(lam, name='ToDouble')
    print('Running the Network')
    tensors = matting_module.matting(batch_images,fb,lam)
    loss = tf.reduce_mean(loss_module.matting_loss(tensors,batch_ground))
    # loss = tensors-batch_ground
    Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    TrainOp = Optimizer.minimize(loss)
    # res, loss_res = sess.run([tensors, loss], feed_dict=feed_dict)
    sess.run(tf.global_variables_initializer())
    # print(tf.get_default_graph().as_graph_def())
    for i in range(1):
      _, res_lam,res_loss, res_fb= sess.run([TrainOp, lam,loss, f], feed_dict=feed_dict)
    # res = sess.run(lam, feed_dict=feed_dict)
    sp.misc.imshow(res_fb[0,:,:])

    # sp.misc.imsave("./tf_alpha.png",res[0,:,:,0])
    # print("loss is : ",loss_res)
    # print(res.shape)
    print(res_lam)
    print(res_loss)
    # res=res*255;
    # print(res.shape)
    # A = res[0, :, :, 0]
    # A[A > 255] = 255
    # A[A < 0] = 0
    # print(A[A > 225])
    # input_in=input_in*255
    # im = Image.fromarray(A.astype(np.uint8))
    # im.show()


    # np.set_printoptions(threshold=np.inf)
    # print(res[0,:,:,0])
    # sp.misc.imshow(A)

  # alpha=sp.misc.imread("./alpha.png").astype(np.double)
  # tf_alpha=sp.misc.imread("./tf_alpha.png").astype(np.double)
  # sp.misc.imshow(tf_alpha-alpha)
  # print(res)




