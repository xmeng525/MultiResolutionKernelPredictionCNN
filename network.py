"""
description: network architecture

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
"""

import tensorflow as tf
import numpy as np
import math
from kernel_ops import kernel_filter, upsampling

class MyModel(object):
    def __init__(self, input_shape, target_shape, loss_name, if_albedo_in_training=False):
        self.input_shape = input_shape
        self.target_shape = target_shape

        self.source = tf.placeholder(tf.float32, self.input_shape, name='source')
        self.target = tf.placeholder(tf.float32, self.target_shape, name='target')

        self.loss_name = loss_name
        self.if_albedo_in_training = if_albedo_in_training
        self.rsz_mtd = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    def inference(self):
        if self.if_albedo_in_training:
            noisy_1 = self.source[:,:,:, 0:3] * self.source[:,:,:,3:6]
            ae_input = tf.concat([noisy_1, self.source[:,:,:, 3:10]], axis=3)
        else:
            noisy_1 = self.source[:,:,:, 0:3]
            ae_input = self.source
        with tf.variable_scope('autoencoder'):
            kernel_1, kernel_2, kernel_3 = self.autoencoder(ae_input, net_name="ae")
            print("kernel_1.shape = ", kernel_1.shape)
            print("kernel_1.shape = ", kernel_2.shape)
            print("kernel_1.shape = ", kernel_3.shape)
            
        with tf.variable_scope('kernel_filters'):
            kernel_weight_1 = kernel_1[:,:,:,:25]
            kernel_alpha_1 = kernel_1[:,:,:,25:]
            denoised_hdr_1 = kernel_filter(noisy_1, kernel_weight_1)

            noisy_2 = self.pool(noisy_1)
            kernel_weight_2 = kernel_2[:,:,:,:25]
            kernel_alpha_2 = kernel_2[:,:,:,25:]
            denoised_hdr_2 = kernel_filter(noisy_2, kernel_weight_2)

            noisy_3 = self.pool(noisy_2)
            kernel_weight_3 = kernel_3[:,:,:,:25]
            kernel_alpha_3 = kernel_3[:,:,:,25:]
            denoised_hdr_3 = kernel_filter(noisy_3, kernel_weight_3)

            noisy_4 = self.pool(noisy_3)

            denoised_4_up = upsampling(noisy_4)
            denoised_3_du = upsampling(self.pool(denoised_hdr_3))

            denoised_3_bd = denoised_hdr_3 - kernel_alpha_3 * denoised_3_du + kernel_alpha_3 * denoised_4_up

            denoised_3_up = upsampling(denoised_3_bd)
            denoised_2_du = upsampling(self.pool(denoised_hdr_2))
            denoised_2_bd = denoised_hdr_2 - kernel_alpha_2 * denoised_2_du + kernel_alpha_2 * denoised_3_up

            denoised_2_up = upsampling(denoised_2_bd)
            denoised_1_du = upsampling(self.pool(denoised_hdr_1))
            denoised_1_bd = denoised_hdr_1 - kernel_alpha_1 * denoised_1_du + kernel_alpha_1 * denoised_2_up

        with tf.device('/gpu:0'):
            with tf.name_scope("final_loss"):
                if not self.if_albedo_in_training:
                    denoised_1_bd = denoised_1_bd * self.source[:,:,:,3:6]
                
                denoised_tm = self.tone_mapping(denoised_1_bd)
                target_tm = self.tone_mapping(self.target)
                if self.loss_name == "L1":
                    loss_denoised = tf.reduce_mean(tf.losses.absolute_difference(denoised_tm, target_tm))
                elif self.loss_name == "L2":
                    loss_denoised = tf.reduce_mean(tf.losses.mean_squared_error(denoised_tm, target_tm))
                tf.summary.scalar('loss_denoised', loss_denoised)

        return {'source': self.source, 'target': self.target,
                'denoised_hdr_1': denoised_hdr_1, 'noisy_1': noisy_1, 'denoised_1_bd': denoised_1_bd, 'kernel_alpha_1':kernel_alpha_1,
                'denoised_hdr_2': denoised_hdr_2, 'noisy_2': noisy_2, 'denoised_2_bd': denoised_2_bd, 'kernel_alpha_2':kernel_alpha_2,
                'denoised_hdr_3': denoised_hdr_3, 'noisy_3': noisy_3, 'denoised_3_bd': denoised_3_bd, 'kernel_alpha_3':kernel_alpha_3,
                'loss_denoised': loss_denoised}

    def autoencoder(self, corrupt_input, net_name=''):
        print('Creating ' + net_name + ' ...')
        current_input = corrupt_input  # input 10 channels

        #
        # corrupt_input = en_relu_out_1 [b, 736, 1280, 48]
        # en_pool_out_1 = en_relu_out_2 [b, 368,  640, 48]
        # en_pool_out_2 = en_relu_out_3 [b, 184,  320, 48]
        # en_pool_out_3 = en_relu_out_4 [b,  92,  160, 48]
        # en_pool_out_4 = en_relu_out_5 [b,  46,   80, 48]
        # en_pool_out_5 = embadding_var [b,  23,   80, 48]

        with tf.variable_scope('encoder'):
            with tf.variable_scope('en_1'):
                print("encoder - 1")
                en_conv_out_1 = self.conv_layer(current_input, 48)
                en_relu_out_1 = self.relu(en_conv_out_1)
                en_pool_out_1 = self.pool(en_relu_out_1)
                current_input = en_pool_out_1
            with tf.variable_scope('en_2'):
                print("encoder - 2")
                en_conv_out_2 = self.conv_layer(current_input, 48)
                en_relu_out_2 = self.relu(en_conv_out_2)
                en_pool_out_2 = self.pool(en_relu_out_2)
                current_input = en_pool_out_2
            with tf.variable_scope('en_3'):
                print("encoder - 3")
                en_conv_out_3 = self.conv_layer(current_input, 48)
                en_relu_out_3 = self.relu(en_conv_out_3)
                en_pool_out_3 = self.pool(en_relu_out_3)
                current_input = en_pool_out_3
            with tf.variable_scope('en_4'):
                print("encoder - 4")
                en_conv_out_4 = self.conv_layer(current_input, 48)
                en_relu_out_4 = self.relu(en_conv_out_4)
                en_pool_out_4 = self.pool(en_relu_out_4)
                current_input = en_pool_out_4
            with tf.variable_scope('en_5'):
                print("encoder - 5")
                en_conv_out_5 = self.conv_layer(current_input, 48)
                en_relu_out_5 = self.relu(en_conv_out_5)
                en_pool_out_5 = self.pool(en_relu_out_5)
                current_input = en_pool_out_5

        with tf.variable_scope('decoder'):
            with tf.variable_scope('de_5'):
                print("decoder - 5")
                de_conv_out_5 = self.deconv_layer(current_input, tf.shape(en_relu_out_5), 48)
                de_relu_out_5 = self.relu(de_conv_out_5)
                current_input = tf.concat([de_relu_out_5, en_relu_out_5], axis=3)
            with tf.variable_scope('de_4'):
                print("decoder - 4")
                de_conv_out_4 = self.deconv_layer(current_input, tf.shape(en_relu_out_4), 48)
                de_relu_out_4 = self.relu(de_conv_out_4)
                current_input = tf.concat([de_relu_out_4, en_relu_out_4], axis=3)
            with tf.variable_scope('de_3'):
                print("decoder - 3")
                de_conv_out_3 = self.deconv_layer(current_input, tf.shape(en_relu_out_3), 48)
                de_relu_out_3 = self.relu(de_conv_out_3)
                current_input = tf.concat([de_relu_out_3, en_relu_out_3], axis=3)
                kernel_3_tmp = self.conv_layer(current_input, 26)
                kernel_3 = self.relu(kernel_3_tmp)
            with tf.variable_scope('de_2'):
                print("decoder - 2")
                de_conv_out_2 = self.deconv_layer(current_input, tf.shape(en_relu_out_2), 48)
                de_relu_out_2 = self.relu(de_conv_out_2)
                current_input = tf.concat([de_relu_out_2, en_relu_out_2], axis=3)
                kernel_2_tmp = self.conv_layer(current_input, 26)
                kernel_2 = self.relu(kernel_2_tmp)
            with tf.variable_scope('de_1'):
                print("decoder - 1")
                de_conv_out_1 = self.deconv_layer(current_input, tf.shape(en_relu_out_1), 48)
                de_relu_out_1 = self.relu(de_conv_out_1)
                current_input = tf.concat([de_relu_out_1, en_relu_out_1], axis=3)
                kernel_1_tmp = self.conv_layer(current_input, 26)
                kernel_1 = self.relu(kernel_1_tmp)
        return kernel_1, kernel_2, kernel_3

    def tone_mapping(self, input_image):
        with tf.name_scope("tone_mapping"):
            tone_mapped_color = tf.clip_by_value(
                tf.math.pow(tf.math.maximum(0., input_image), 0.454545), 0., 1.)
            return tone_mapped_color

    def rgb2gray(self, input_image):
        with tf.name_scope("rgb2gray"):
            return 0.2989 * input_image[:,:,:,0] + 0.5870 * input_image[:,:,:,1] + 0.1140 * input_image[:,:,:,2]

    def conv_layer(self, batch_input, n_output_ch, filter_size=3, stride=(1,1,1,1)):
        input_shape = batch_input.get_shape().as_list()
        print('DEBUG:: InShape = ', input_shape)
        n_input_ch = input_shape[3]
        with tf.variable_scope('conv'):
            W = tf.get_variable(name='weight',shape=[filter_size, filter_size, n_input_ch, n_output_ch],
                dtype=np.float32, initializer=tf.glorot_uniform_initializer())
            tf.summary.histogram('weight', W)
            b = tf.get_variable(name='bias',shape=[n_output_ch],
                dtype=np.float32, initializer=tf.constant_initializer(0.0))
            tf.summary.histogram('bias', b)
            output = tf.add(tf.nn.conv2d(
                batch_input, W, strides=stride, padding='SAME'), b, name='Wx_p_b')
        print('DEBUG:: OutShape = ', output.get_shape().as_list())
        return output

    def deconv_layer(self, batch_input, output_shape, n_output_ch, filter_size=3, stride=(1,2,2,1)):
        input_shape = batch_input.get_shape().as_list()
        print('DEBUG:: InShape = ', input_shape)
        n_input_ch = input_shape[3]

        with tf.variable_scope('deconv'):
            W = tf.get_variable(name='weight', shape=[filter_size, filter_size, n_output_ch, n_input_ch],
                                dtype=np.float32, initializer=tf.glorot_uniform_initializer())
            tf.summary.histogram('weight', W)
            b = tf.get_variable(name='bias',shape=[n_output_ch],
                dtype=np.float32, initializer=tf.constant_initializer(0.0))
            tf.summary.histogram('bias', b)
            output = tf.add(tf.nn.conv2d_transpose(
                batch_input, W, output_shape=output_shape, strides=stride, padding='SAME'), b, name='Wx_p_b')
        print('DEBUG:: OutShape = ', output.get_shape().as_list())
        return output

    def relu(self, x, name="relu"):
        with tf.variable_scope(name):
            return tf.nn.relu(x, name)

    def pool(self, inp, k=2):
        return tf.nn.avg_pool(inp, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

