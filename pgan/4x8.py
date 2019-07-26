#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Florian Grimm, Vanessa Kirchner, Andreas Specker


import os 
import sys
import argparse
import tensorflow as tf
import cv2
import numpy as np
from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
sys.path.append('shared')
sys.path.append('pgan/shared')
from GAN import GANTrainer, GANModelDesc
from myConv2D import myConv2D


# Define batch size (parallel processed iterations), channel size of images and images current width and height
BATCH_SIZE = 16
CHANNELS = 3
H = 8
W = 8


@layer_register()
def Upsample(x, factor=2):
    h, w = x.get_shape().as_list()[1:3]
    return tf.image.resize_nearest_neighbor(x, [h * factor, w * factor], align_corners=True)


def Downsample(name, x, factor=2):
    assert factor == 2
    return AvgPooling(name, x, factor, padding='VALID')
    
def combine_img(current, previous, alpha): # takes alpha into account
    return alpha * current + (1 - alpha) * previous

@layer_register(use_scope=None)
def PixelwiseNorm(x, eps=1e-8): # Prevents generators and discriminators magnitudes of loss getting out of control
    scale = tf.reduce_mean(x**2, axis=1, keep_dims=True) + eps
    return x / (scale ** 0.5)


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, H, W, CHANNELS), 'curInput'), InputDesc(tf.float32, (None, H // 2, W // 2, CHANNELS), 'prevInput')]
        

    def toRGB(self, x, strval):
        x = myConv2D(strval, x, 3, kernel_shape=1, stride=1, nl=tf.identity)
        return x

    def fromRGB(self, x, strval):
        x = myConv2D(strval, x, 512, kernel_shape=1, stride=1, nl=tf.identity)
        return x

    # For visualiztion of alpha in tensorboard
    def visualize_images(self, name, *imgs):
        xy = (tf.concat(imgs, axis=2) + 1.) * 128.
        xy = tf.cast(tf.clip_by_value(xy, 0, 255), tf.uint8, name='viz')
        tf.summary.image(name, xy, max_outputs=30)

    # Computes standard deviation per feature vector, averages all of them and adds it as extra layer
    # at the end of the feature vector (see report)
    def compute_stddev(self, x, eps=1e-8):
        _, h, w, _ = x.get_shape().as_list()
        _, var = tf.nn.moments(x, axes=[3], keep_dims=True)
        stddev = tf.reduce_mean(tf.sqrt(var), keep_dims=True)
        y = tf.tile(stddev, [BATCH_SIZE, h, w, 1])
        res = tf.concat([x, y], axis=3)
        return res


    # loss function for generator and discriminator
    def build_losses(self, vecpos, vecneg, vec_interp, interp):
        # the Wasserstein-GAN losses
        self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
        self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')

        # the gradient penalty loss
        gradients = tf.gradients(vec_interp, [interp])[0]
        gradients = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))
        gradients_rms = symbolic_functions.rms(gradients, 'gradient_rms')
        gradient_penalty = tf.reduce_mean(tf.square(gradients - 1), name='gradient_penalty')
        add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms)

        self.d_loss = tf.add(self.d_loss, 10 * gradient_penalty)
        
        # Drift loss
        eps_drift = 0.001
        drift_loss = tf.reduce_mean(tf.nn.l2_loss(vecpos))
        self.d_loss = tf.add_n([self.d_loss, 10 * gradient_penalty, eps_drift * drift_loss], name='total_d_loss')
        add_moving_summary(self.d_loss, drift_loss)

    def generator(self, z, alpha):
        z = tf.reshape(z, [BATCH_SIZE, 1, 1, 512])  # 1 1 512

        x = Upsample('upsample000', z, 4)  # 4 4 512

        # Convolution layers for stage 4 4 512
        x = myConv2D('conv001', x, 512, kernel_shape=4, stride=1, nl=LeakyReLU)
        x = PixelwiseNorm(x)
        x = myConv2D('conv002', x, 512, kernel_shape=3, stride=1, nl=LeakyReLU)
        x = PixelwiseNorm(x)
        
        x = Upsample('upsample001', x, 2)  # 8 8 512
        
        # Alpha without convolution
        x_left = self.toRGB(x, 'toRGBtemp_4_8')
        x_left_alpha = tf.multiply((1 - alpha), x_left)

        # Convolution layers for stage 8 8 512
        x = myConv2D('conv003', x, 512, kernel_shape=3, stride=1, nl=LeakyReLU)
        x = PixelwiseNorm(x)
        x = myConv2D('conv004', x, 512, kernel_shape=3, stride=1, nl=LeakyReLU)
        x = PixelwiseNorm(x)
        
        x = self.toRGB(x, 'toRGB_4_8')

        # Alpha with convolution and subsequent combination
        x_right_alpha = tf.multiply(alpha, x)
        x_right = x
        x = tf.add(x_left_alpha, x_right_alpha)
        
        # Visualization of alpha for tensorboard
        self.visualize_images('vizAlpha', x_left, x, x_right)
        self.visualize_images('vizNoAlpha', x_left_alpha, x, x_right_alpha)
        
        return x

    @auto_reuse_variable_scope
    def discriminator(self, x, alpha):

        # Alpha without convolution
        x_left = Downsample('disDowntemp_4_8', x)
        x_left = self.fromRGB(x_left, 'fromRGBtemp_4_8')
        x_left_alpha = tf.multiply((1 - alpha), x_left)

        x = self.fromRGB(x, 'fromRGB_4_8')

        # Convolution layers for stage 8 8 512
        x = myConv2D('conv004', x, 512, kernel_shape=3, stride=1, nl=LeakyReLU)
        x = myConv2D('conv003', x, 512, kernel_shape=3, stride=1, nl=LeakyReLU)

        x = Downsample('downsample001', x) # 4 4 512

        # Alpha with convolution and subsequent combination
        x_right_alpha = tf.multiply(alpha, x)
        x = tf.add(x_left_alpha, x_right_alpha)

        # MiniBatch discrimination
        x = self.compute_stddev(x)
        
        # Convolution layers for stage 4 4 512
        x = myConv2D('conv002', x, 512, kernel_shape=3, stride=1, nl=LeakyReLU)
        x = myConv2D('conv001', x, 512, kernel_shape=4, stride=1, padding='VALID', nl=LeakyReLU)

        logits = FullyConnected('fc', x, 1, nl=tf.identity)

        return logits  # 1 1 1

    def _build_graph(self, inputs):
    
        # CelebA HQ images for current resolution [H, W]
        real_img = inputs[0]                # [0, 255]
        real_img = real_img / 128.0 - 1.0   # map to [-1.0f, 1.0f]
        
        # CelebA HQ images for previous resolution [H/2, W/2]
        real_img_prev = Upsample('real_img_prev_4_8', inputs[1])
        real_img_prev = real_img_prev / 128.0 - 1.0

        # Latent random noise vector as initial input
        z = tf.random_uniform([BATCH_SIZE, 512], -1.0, 1.0, name='z')
        
        # Calculation of alpha for fading in new information
        seenImages = tf.multiply(get_global_step_var(),  BATCH_SIZE)
        alpha = tf.divide(seenImages, 600000)
        alpha = tf.cast(alpha, dtype=tf.float32)
        add_moving_summary(alpha, seenImages)
        
        # For Transition -> Alpha
        real_img = combine_img(real_img, real_img_prev, alpha)

        # Generator part
        with argscope(LeakyReLU, alpha=0.2):
            with tf.variable_scope('gen'):
                fake_img = self.generator(z, alpha)

        # For improved loss
        someAlpha = tf.random_uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0., maxval=1., name='someAlpha')
        interp = real_img + someAlpha * (fake_img - real_img)

        # Discriminator part
        with argscope(LeakyReLU, alpha=0.2):
            with tf.variable_scope('discrim'):
                score_fake = self.discriminator(fake_img, alpha)
                score_real = self.discriminator(real_img, alpha)
                vec_interp = self.discriminator(interp, alpha)

        self.build_losses(score_real, score_fake, vec_interp, interp)
        self.collect_variables()

        # Mapping back to [0, 255]
        real_img = 128.0 * (real_img + 1.0)
        fake_img = 128.0 * (fake_img + 1.0)

        # Visualization for tensorboard
        viz = tf.concat([fake_img, real_img], 2)
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('fake,real', viz, max_outputs=max(30, BATCH_SIZE))

    def _get_optimizer(self):
        opt = tf.train.RMSPropOptimizer(1e-4)
        return opt
        
        
class ImageDecode(MapDataComponent):
    def __init__(self, ds, dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img
        super(ImageDecode, self).__init__(ds, func, index=index)


def get_data():
    lmdb = '/graphics/scratch/datasets/celebHQ/celeb_hq_256.lmdb'
  
    from PIL import Image
    import os
    assert os.path.isfile(lmdb)

    def resize(img, shp=4):
        img = img.resize((shp, shp), Image.ANTIALIAS)
        return np.array(img)

    def bgr2rgb(x):
        return x[:, :, ::-1]

    ds = LMDBDataPoint(lmdb, shuffle=True)
    ds = ImageDecode(ds, index=0)

    ds = MapDataComponent(ds, bgr2rgb, index=0)
    ds = MapData(ds, lambda x: [resize(Image.fromarray(x[0]), 8), resize(Image.fromarray(x[0]), 8 // 2)])


    ds = PrefetchDataZMQ(ds, 12)
    ds = BatchData(ds, BATCH_SIZE)
    return ds




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    logger.auto_set_dir()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    GANTrainer(
        input=QueueInput(get_data()),
        model=Model()).train_with_defaults(
        callbacks=[ModelSaver()],
        steps_per_epoch=1875, # <- 30.000 / BATCH_SIZE
        max_epoch=20, # <- 600.000 / 30.000
        session_init=SaverRestore(args.load, 'loaded', ignore=['global_step']) if args.load else None
    )
