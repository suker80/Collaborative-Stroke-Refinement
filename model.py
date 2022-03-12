from glob import glob
import matplotlib.pyplot as plt
from ops import *

import argparse

class Model(object):

    def __init__(self, batch__size, lr, epoch):
        self.batch_size = batch__size
        self.lr = lr
        self.epochs = epoch

    def build_model(self):

        print('build model...')
        self.x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 64, 64, 1])
        self.target = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 64, 64, 1])
        self.bold = Dilation(self.target)
        net = Adaptive_deform(self.x,'target')
        net = hv_transform(net)
        self.CNN_layers(net, reuse=False)
        self.aux_branch()
        self.refine()
        self.dominated()
        self.loss()
        print('model build complete! ')

    def dominated(self):
        net = self.CNN_outptut
        with tf.variable_scope('dominated'):
            net = RBC(net, 64, stride=[1, 1])
            concated = tf.concat([net, self.net_conv], axis=3)
            net = RBC(concated, 64, stride=[1, 1])
            concated = tf.concat([net, self.netflow_right], axis=3)

            net = RBD(concated, 64)
            self.output = RBD(net, 1, kernel=[3, 3], stride=[1, 1])

    def aux_branch(self):
        net = self.CNN_outptut
        net = RBD(net, 64)
        net = RBD(net, 1, kernel=[3, 3], stride=[1, 1])
        self.bold_output = Dilation(net)

    def refine(self):
        net = self.bold_output
        with tf.variable_scope('refine'):
            net = max_pool(net)
            self.net_conv = RBC(net, 64, kernel=[3, 3], stride=[1, 1])

            netflow_right = max_pool(net)
            self.netflow_right = RBD(netflow_right, 64, kernel=[3, 3], stride=[2, 2])

    def CNN_layers(self, input, reuse=False):
        with tf.variable_scope('coarse_generator', reuse=reuse):
            self.first_layer = EBC(input, 64, kernel=[3, 3], stride=[1, 1])  # 64 * 64 - > 64 * 64

            conv_2 = EBC(self.first_layer, 64)  # 64 * 64 - > 32 * 32
            conv_3 = EBC(conv_2, 128, kernel=[3, 3], stride=[1, 1])

            conv_4 = EBC(conv_3, 128)  # 32 *32  - > 16 * 16
            conv_5 = EBC(conv_4, 256, kernel=[3, 3], stride=[1, 1])

            conv_6 = EBC(conv_5, 256)  # 16 * 16 - > 8 * 8
            conv_7 = EBC(conv_6, 512, kernel=[3, 3], stride=[1, 1])

            conv_8 = ELU(conv2d(conv_7, 512, kernel=[2, 2]))  # 8 * 8 - > 4 * 4

            deconv_1 = RBD(conv_8, 512)  # 4 * 4 -> 8 * 8
            deconv_2 = RBD(tf.concat([deconv_1, conv_7], axis=3), 512, stride=[1, 1])
            deconv_3 = RBD(deconv_2, 512, stride=[1, 1])

            deconv_4 = RBD(deconv_3, 256)  # 8 * 8 -> 16 * 16
            deconv_5 = RBD(tf.concat([deconv_4, conv_5], axis=3), 256, stride=[1, 1])
            deconv_6 = RBD(deconv_5, 128, stride=1)

            deconv_7 = RBD(deconv_6, 128)  # 16 * 16 -> 32 * 32
            deconv_8 = RBD(tf.concat([deconv_7, conv_3], axis=3), 128, stride=1)
            self.CNN_outptut = RBD(deconv_8, 64, stride=1)

    def disc_1(self, y, y_, reuse=False):
        with tf.variable_scope('disc_1', reuse=reuse):
            input = tf.concat([y, y_], axis=3)

            conv_1 = RBC(input, 64)
            conv_2 = RBC(conv_1, 128)
            conv_3 = RBC(conv_2, 256)
            conv_4 = RBC(conv_3, 512)

            output = tf.layers.dense(tf.reshape(conv_4, [self.batch_size, -1]), 1)
            return tf.nn.sigmoid(output), output

    def disc_2(self, y, y_, reuse=False):
        with tf.variable_scope('disc_2', reuse=reuse):
            input = tf.concat([y, y_], axis=3)

            conv_1 = RBC(input, 64)
            conv_2 = RBC(conv_1, 128)
            conv_3 = RBC(conv_2, 256)
            conv_4 = RBC(conv_3, 512)

            output = tf.layers.dense(tf.reshape(conv_4, [self.batch_size, -1]), 1)
            return output

    def loss(self):
        d1_real = self.disc_1(self.x, self.target)
        d1_fake = self.disc_1(self.x, self.output, reuse=True)

        d2_real = self.disc_2(self.x, self.bold)
        d2_fake = self.disc_2(self.x, self.output, reuse=True)

        L1_pixel = tf.reduce_mean(
            tf.multiply(-self.target, tf.log(self.output)) +
            tf.multiply((tf.ones_like(self.target) - self.target),
                        tf.log(tf.ones_like(self.output) - self.output))
        )
        L1_pixel = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target,logits=self.output))
        # L1_pixel = tf.reduce_mean(tf.abs(self.target - self.output))
        L1_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_fake, labels=tf.ones_like(d1_fake)))
        L1_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_fake, labels=tf.zeros_like(d1_fake)))
        L1_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d1_real, labels=tf.ones_like(d1_real)))

        self.d1 = L1_fake + L1_real

        L2_pixel = tf.reduce_mean(
            tf.multiply(-self.bold , tf.log(self.bold_output)) +
            tf.multiply(- (tf.ones_like(self.bold) - self.bold) , tf.log(tf.ones_like(self.bold_output) - self.bold_output))
        )
        L2_pixel = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.bold,logits=self.bold_output))
        # L2_pixel = tf.reduce_mean(tf.abs(self.bold - self.bold_output))
        L2_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_fake, labels=tf.ones_like(d2_fake)))
        L2_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_fake, labels=tf.zeros_like(d2_fake)))
        L2_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_real, labels=tf.ones_like(d2_real)))

        self.d2 = L2_fake + L2_real

        d1_vars = [var for var in tf.trainable_variables() if 'disc_1' in var.name]
        d2_vars = [var for var in tf.trainable_variables() if 'disc_2' in var.name]
        g2_vars = [var for var in tf.trainable_variables() if 'refine' in var.name]
        g1_vars = [var for var in tf.trainable_variables() if 'coarse_generator' in var.name] + g2_vars


        self.g1_loss = L1_g + L1_pixel
        self.g2_loss = L2_g + L2_pixel
        self.d1_optim = tf.train.AdamOptimizer(self.lr).minimize(self.d1, var_list=d1_vars)
        self.d2_optim = tf.train.AdamOptimizer(self.lr).minimize(self.d2, var_list=d2_vars)
        self.g1_optim = tf.train.AdamOptimizer(self.lr).minimize(self.g1_loss, var_list=g1_vars)
        self.g2_optim = tf.train.AdamOptimizer(self.lr).minimize(self.g2_loss, var_list=g2_vars)

        self.total_loss = self.g1_loss + self.g2_loss + self.d1 + self.d2

        self.Entire_optim = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

    def train(self):

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            train_data = glob('train/*.png')
            target_data = glob('target/*.png')
            data = [[train_data[i], target_data[i]] for i in range(len(train_data))]
            np.random.shuffle(data)
            batch_idxs = len(data) // self.batch_size
            saver = tf.train.Saver()
            for epoch in range(self.epochs):
                for idx in range(batch_idxs):
                    batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                    x = [image_read(batch_file[0]) for batch_file in batch_files]
                    y = [image_read(batch_file[1]) for batch_file in batch_files]

                    total_loss, d1_loss, d2_loss, g1_loss, g2_loss ,output= sess.run(
                        [self.total_loss, self.d1, self.d2, self.g1_loss, self.g2_loss,self.output],
                        feed_dict={self.x: x, self.target: y})
                    _, _, _, _, _ = sess.run(
                        [self.d1_optim, self.d2_optim, self.g1_optim, self.g2_optim, self.Entire_optim],
                        feed_dict={self.x: x, self.target: y})
                    print(
                        'epoch : {} iter : {} total loss : {} d1_loss : {} d2_loss : {} g1_loss : {} g2_loss : {}] '.format(
                            epoch, idx, total_loss, d1_loss, d2_loss, g1_loss, g2_loss))

                saver.save(sess, 'checkpoint/model', global_step=epoch)
    def test(self):
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint('checkpoint')

        with tf.Session() as sess:

            saver.restore(sess,checkpoint)
            train_data = glob('train/*.png')
            target_data = glob('target/*.png')
            data = [[train_data[i], target_data[i]] for i in range(len(train_data))]
            np.random.shuffle(data)
            batch_idxs = len(data) // self.batch_size
            for epoch in range(self.epochs):
                for idx in range(batch_idxs):
                    batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                    x = [image_read(batch_file[0]) for batch_file in batch_files]
                    output = sess.run(self.output, feed_dict={self.x: x})
                    for i in range(16):
                        save_path = 'result/' + batch_files[i][1].split('/')[1]
                        out = output[i].reshape((64,64))
                        plt.imsave(save_path,out)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        model = Model(16, 0.001, 10000000)
        model.build_model()
        model.train()

    else:

        model = Model(16, 0.001, 10000000)
        model.build_model()
        model.test()
