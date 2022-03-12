from PIL import Image
import os
import tensorflow as tf
import cv2
import numpy as np
from scipy.misc import imread
def batch_norm(input):
    return tf.contrib.layers.batch_norm(inputs=input, epsilon=1e-5, decay=0.9, updates_collections=None, scale=True,
                                        data_format='NCHW')


def conv2d(input, num_output, kernel=[4, 4], stride=[2, 2], padding='SAME'):
    return tf.layers.conv2d(inputs=input,
                            filters=num_output,
                            kernel_size=kernel,
                            strides=stride,
                            padding=padding)


def leaky_relu(x, leak=0.2):
    return tf.nn.leaky_relu(x)


def deconv2d(input, num_output, kernel=[4, 4], stride=[2, 2], padding='SAME'):
    return tf.layers.conv2d_transpose(inputs=input,
                                      filters=num_output,
                                      kernel_size=kernel,
                                      strides=stride,
                                      padding=padding
                                      )
def Dilation(img):
    _,_,_,channel = img.shape.as_list()
    kernel = tf.ones([5,5,channel])
    erode = tf.nn.erosion2d(img,kernel,[1,1,1,1],[1,1,1,1],'SAME')
    return erode

def ELU(x):
    return tf.nn.elu(x)

def max_pool(x):
    net = tf.nn.max_pool(x,3,2,padding="SAME")
    return net
def relu(x):
    return tf.nn.relu(x)


def LBC(input, num_output, kernel=[2, 2], stride=[2, 2], padding='SAME'):
    net = conv2d(input, num_output=num_output, kernel=kernel, stride=stride, padding=padding)
    net = batch_norm(net)
    net = leaky_relu(net)
    return net


def RBC(input, num_output, kernel=[2, 2], stride=[2, 2], padding='SAME'):
    net = conv2d(input, num_output=num_output, kernel=kernel, stride=stride, padding=padding)
    net = batch_norm(net)
    net = relu(net)
    return net


def RBD(input, num_output, kernel=[2, 2], stride=[2, 2], padding='SAME'):
    net = deconv2d(input, num_output=num_output, kernel=kernel, stride=stride, padding=padding)
    net = batch_norm(net)
    net = relu(net)
    return net


def EBC(input, num_output, kernel=[2, 2], stride=[2, 2], padding='SAME'):
    net = conv2d(input, num_output, kernel=kernel, stride=stride, padding=padding)
    net = batch_norm(net)
    net = ELU(net)
    return net


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def hv_transform(image):
    '''
    '''
    orig_shp = [64,64]


    image = tf.cast(image, tf.float32)

    rd = tf.random.uniform([])
    less_than_025 = tf.less(rd, tf.constant(0.25))
    less_than_050 = tf.less(rd, tf.constant(0.50))
    less_than_075 = tf.less(rd, tf.constant(0.75))

    def func_less_than_025():
        x = tf.image.resize_images(image, [orig_shp[0] // 2, orig_shp[1]])

        h_off = int(np.random.uniform(0,32))
        x = tf.image.pad_to_bounding_box(
            x,
            offset_height=h_off,
            offset_width=0,
            target_height=orig_shp[0],
            target_width=orig_shp[1])
        return x

    def func_less_than_050():
        x = tf.image.resize_images(image, [orig_shp[0], orig_shp[1] // 2])

        w_off = int(np.random.uniform(0,32))
        np.random.uniform()
        x = tf.image.pad_to_bounding_box(
            x,
            offset_height=0,
            offset_width=w_off,
            target_height=orig_shp[0],
            target_width=orig_shp[1])
        return x

    def func_less_than_075():
        x = tf.image.resize_images(image, [orig_shp[0] // 2, orig_shp[1] // 2])

        h_off = int(np.random.uniform(0,32))
        w_off = int(np.random.uniform(0,32))
        x = tf.image.pad_to_bounding_box(
            x,
            offset_height=h_off,
            offset_width=w_off,
            target_height=orig_shp[0],
            target_width=orig_shp[1])
        return x

    def func_less_than_100():
        return image

    x = tf.case(
        {less_than_025: func_less_than_025, less_than_050: func_less_than_050, less_than_075: func_less_than_075},
        default=func_less_than_100, exclusive=False)
    return x


def Adaptive_deform(image,target_root):
    r1,r2 = calcul_r(target_root)
    image = tf.image.resize_image_with_pad(image,int(64 * r2),64)
    image = tf.image.resize(image,[64,64])
    image = r1* image
    return image

def calcul_hw(image):
    h_min = np.where(image < 127.5)[0].min()
    h_max = np.where(image < 127.5)[0].max()
    w_min = np.where(image < 127.5)[1].min()
    w_max = np.where(image < 127.5)[1].max()

    h= h_max - h_min
    w = w_max - w_min
    return h,w

def calcul_r(target_root):
    r2 = []
    r1 = []
    imgs = os.listdir(target_root)
    imgs = [os.path.join(target_root,img) for img in imgs]
    for img_path in imgs:
        img = image_read(img_path)

        h,w = calcul_hw(img)
        r1.append(h*w/64/64)
        r2.append(h/w)
    return np.mean(np.asarray(r1)),np.mean(np.asarray(r2))

def image_read(path):
    img = Image.open(path)
    img = img.resize((64,64))
    img = img.convert('L')
    img = np.asarray(img)
    img = np.expand_dims(img,3)
    img = img / 255.0
    return img
