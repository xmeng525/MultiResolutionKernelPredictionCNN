"""
description: MR-KP (5-layer) Training
Github: https://github.com/xmeng525/MultiResolutionKernelPredictionCNN

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
"""

import time
import tensorflow as tf
from dataLoader import dataLoader
from network import MyModel
import os
import numpy as np
from image_utils import *
from tensorflow.python.client import timeline

# Training Parameters
learning_rate = 0.0001

total_epochs_stage1 = 60 #30
validate_interval = 1

validation_disp_list = [0, 3]

# Batch Parameters
user_batch_size = 10 # for training and validation
test_batch_size = 1 # for test

patch_width = 128
patch_height = 128

train_start_idx = 5
train_per_scene = 55

vali_start_idx = 55
vali_per_scene = 5

patch_per_img = 50

input_channels = 10
target_channels = 3

data_dir = r'../../dataset/dataset_blockwise/BMFR-dataset'
scene_train_list = ['san-miguel', 'sponza', 'living-room', 'sponza-moving-light', 'sponza-glossy']
scene_validate_list = ['san-miguel', 'sponza', 'living-room', 'sponza-moving-light', 'sponza-glossy']
scene_test_list = ['classroom']

def tone_mapping(input_image):
    tone_mapped_color = np.clip(
        np.power(np.maximum(0., input_image), 0.454545), 0., 1.)
    return tone_mapped_color

def _parse_function(proto):  # for training data
    # with tf.name_scope("parse_data"):
    features = tf.parse_single_example(
        proto, features={
            'target': tf.FixedLenFeature([], tf.string),
            'input': tf.FixedLenFeature([], tf.string)})

    train_input = tf.decode_raw(features['input'], tf.float16)
    train_input = tf.reshape(train_input, [patch_height,
                                           patch_width, input_channels])

    train_target = tf.decode_raw(features['target'], tf.float16)
    train_target = tf.reshape(train_target, [patch_height,
                                             patch_width, target_channels])
    return (train_input, train_target)

def my_mkdir(name):
    if not os.path.exists(name):
        print('creating folder', name)
        os.makedirs(name)

if __name__ == "__main__":
    # Summary log directory
    summary_dir = r'./summary_log'
    my_mkdir(summary_dir)
    stg1_summary_dir = os.path.join(summary_dir, 'stage1')
    my_mkdir(stg1_summary_dir)
    # Result directory
    result_dir = r'./result'
    my_mkdir(result_dir)
    # Error log directory
    err_log_dir = r'./errorlogs'
    my_mkdir(err_log_dir)
    model_dir = r'./model'
    my_mkdir(model_dir)
    stg1_model_dir = os.path.join(model_dir, 'stage1')
    my_mkdir(stg1_model_dir)

    seed = 1
    rng = np.random.RandomState(seed)
    train_data = dataLoader(data_dir=data_dir, subset='train',
                            patch_width=patch_width,
                            patch_height=patch_height,
                            image_start_idx=train_start_idx,
                            img_per_scene=train_per_scene,
                            patch_per_img=patch_per_img,
                            scene_list=scene_train_list, rng=rng)
    validate_data = dataLoader(data_dir=data_dir, subset='validate',
                               patch_width=patch_width,
                               patch_height=patch_height,
                               image_start_idx=vali_start_idx,
                               img_per_scene=vali_per_scene,
                               patch_per_img=patch_per_img,
                               scene_list=scene_validate_list, rng=rng)

    # Train
    train_dataset = tf.data.TFRecordDataset([train_data.dataset_name])
    # Parse the record into tensors.
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(user_batch_size)

    # Validate
    validation_dataset = tf.data.TFRecordDataset(
         [validate_data.dataset_name])
    validation_dataset = validation_dataset.map(_parse_function)
    validation_dataset = validation_dataset.batch(user_batch_size)

    handle_small = tf.placeholder(tf.string, shape=[])
    iterator_structure_small = tf.data.Iterator.from_string_handle(
        handle_small, train_dataset.output_types, train_dataset.output_shapes)
    next_element_small = iterator_structure_small.get_next()
    train_iterator = train_dataset.make_initializable_iterator()
    validate_iterator = validation_dataset.make_initializable_iterator()

    # Model
    model = MyModel(input_shape=[None, None, None, input_channels],
                    target_shape=[None, None, None, target_channels],
                    loss_name="L1", if_albedo_in_training=False)
    with tf.device("/gpu:0"):
        ae_net = model.inference()

    saver = tf.train.Saver()

    # stage 1 loss
    loss_grid_L1 = ae_net['loss_denoised']
    optimizer1 = tf.train.AdamOptimizer(learning_rate)
    output_vars1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder')
    train_step1 = optimizer1.minimize(loss_grid_L1, var_list=output_vars1)
    
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    run_metadata = tf.RunMetadata()
    opts1 = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts1)

    opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts2)

    sess.run(tf.global_variables_initializer())

    print('Stage 1 starts: ')
    min_loss = 10000
    summary_merge = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(stg1_summary_dir, graph=sess.graph)
    begin_time_train = time.time()
    
    training_handle = sess.run(train_iterator.string_handle())
    validation_handle = sess.run(validate_iterator.string_handle())

    train_psnr_epoch_mean = []
    train_loss_epoch_mean = []

    validate_psnr_epoch_mean = []
    validate_loss_epoch_mean = []

    for epoch_i in range(total_epochs_stage1):
        should_validate = ((epoch_i + 1) % validate_interval == 0)

        epoch_avg_loss_train = 0.0
        epoch_avg_psnr_train = 0.0

        batch_cnt = 0
        # Training
        sess.run(train_iterator.initializer)
        while True:
            try:
                src_hdr, tgt_hdr = sess.run(next_element_small,
                    feed_dict={handle_small: training_handle})
                feed_dict = {ae_net['source']: src_hdr, ae_net['target']: tgt_hdr}
                summary, denoised_1_bd, batch_loss, _ = sess.run(
                    [summary_merge, ae_net['denoised_1_bd'], loss_grid_L1, train_step1], feed_dict)
                summary_writer.add_summary(summary, epoch_i)

                denoised_1_tm = tone_mapping(denoised_1_bd)
                tgt = tone_mapping(tgt_hdr)

                _, batch_psnr_val = batch_psnr(denoised_1_tm, tgt)

                epoch_avg_psnr_train += batch_psnr_val
                epoch_avg_loss_train += batch_loss

                batch_cnt += 1
            except tf.errors.OutOfRangeError:
                epoch_avg_psnr_train /= batch_cnt
                epoch_avg_loss_train /= batch_cnt

                train_psnr_epoch_mean.append(epoch_avg_psnr_train)
                train_loss_epoch_mean.append(epoch_avg_loss_train)

                print('S1, Epoch %d Train\npsnr = %.4f\nloss = %.4f'% (
                    epoch_i, epoch_avg_psnr_train, epoch_avg_loss_train))
                break
        # Validate
        if should_validate:
            epoch_avg_loss_validate = 0.0
            epoch_avg_psnr_validate = 0.0
            batch_cnt = 0

            sess.run(validate_iterator.initializer)
            while True:
                try:
                    src_hdr, tgt_hdr = sess.run(next_element_small,
                        feed_dict={handle_small: validation_handle})
                    feed_dict = {ae_net['source']: src_hdr, ae_net['target']: tgt_hdr}
                    denoised_1_bd, noisy_1, denoised_hdr_1, kernel_alpha_1, \
                    denoised_2_bd, noisy_2, denoised_hdr_2, kernel_alpha_2, \
                    denoised_3_bd, noisy_3, denoised_hdr_3, kernel_alpha_3, \
                    batch_loss = sess.run(
                        [
                            ae_net['denoised_1_bd'], ae_net['noisy_1'], ae_net['denoised_hdr_1'], ae_net['kernel_alpha_1'],
                            ae_net['denoised_2_bd'], ae_net['noisy_2'], ae_net['denoised_hdr_2'], ae_net['kernel_alpha_2'],
                            ae_net['denoised_3_bd'], ae_net['noisy_3'], ae_net['denoised_hdr_3'], ae_net['kernel_alpha_3'],
                            loss_grid_L1
                        ], feed_dict)

                    denoised_1_bd_tm = tone_mapping(denoised_1_bd)
                    denoised_2_bd_tm = tone_mapping(denoised_2_bd)
                    denoised_3_bd_tm = tone_mapping(denoised_3_bd)

                    noisy_1_tm = tone_mapping(noisy_1)
                    noisy_2_tm = tone_mapping(noisy_2)
                    noisy_3_tm = tone_mapping(noisy_3)

                    denoised_1_tm = tone_mapping(denoised_hdr_1)
                    denoised_2_tm = tone_mapping(denoised_hdr_2)
                    denoised_3_tm = tone_mapping(denoised_hdr_3)

                    tgt = tone_mapping(tgt_hdr)
                    src = tone_mapping(src_hdr)

                    _, batch_psnr_val = batch_psnr(denoised_1_bd_tm, tgt)

                    epoch_avg_psnr_validate += batch_psnr_val
                    epoch_avg_loss_validate += batch_loss

                    if batch_cnt in validation_disp_list:
                        for k in range(tgt.shape[0]):
                            save_image_pil(result_dir, tgt[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_tgt')
                            
                            save_image_pil(result_dir, noisy_1_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_1_1_noisy')
                            save_image_pil(result_dir, noisy_2_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_2_1_noisy')
                            save_image_pil(result_dir, noisy_3_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_3_1_noisy')

                            save_image_pil(result_dir, denoised_1_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_1_2_denoised')
                            save_image_pil(result_dir, denoised_2_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_2_2_denoised')
                            save_image_pil(result_dir, denoised_3_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_3_2_denoised')

                            save_image_pil(result_dir, denoised_1_bd_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_1_3_final')
                            save_image_pil(result_dir, denoised_2_bd_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_2_3_final')
                            save_image_pil(result_dir, denoised_3_bd_tm[k, :, :, :],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_3_3_final')

                            save_image_gray(result_dir, kernel_alpha_1[k, :, :, 0],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_1_4_alpha')
                            save_image_gray(result_dir, kernel_alpha_2[k, :, :, 0],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_2_4_alpha')
                            save_image_gray(result_dir, kernel_alpha_3[k, :, :, 0],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_3_4_alpha')

                            save_image_pil(result_dir, src[k, :, :, 0:3],
                                'e' + str(epoch_i) + '_b' + str(batch_cnt) + '_idx' + str(k) + '_src')
                    batch_cnt += 1
                except tf.errors.OutOfRangeError:
                    epoch_avg_psnr_validate /= batch_cnt
                    epoch_avg_loss_validate /= batch_cnt

                    validate_psnr_epoch_mean.append(epoch_avg_psnr_validate)
                    validate_loss_epoch_mean.append(epoch_avg_loss_validate)

                    print('S1, Epoch %d Valid\npsnr = %.4f\nloss = %.4f'% (
                        epoch_i, epoch_avg_psnr_validate, epoch_avg_loss_validate))
                    if epoch_avg_loss_validate < min_loss:
                        saver.save(sess, os.path.join(stg1_model_dir, 'best_model'))
                        min_loss = epoch_avg_loss_validate
                    break             
        # Validate epoch finished.
    # Stage 1 Training finished.
    stage1_total_time = time.time() - begin_time_train
    print("Stage 1 done, total training time = %.4fs " % (stage1_total_time))

    # saver.save(sess, os.path.join(stg1_model_dir, 'my_model'))

    np.savetxt(err_log_dir + '/S1_psnr_train.txt',
               train_psnr_epoch_mean, fmt='%.8f', delimiter=',')
    np.savetxt(err_log_dir + '/S1_loss_train.txt',
               train_loss_epoch_mean, fmt='%.8f', delimiter=',')

    np.savetxt(err_log_dir + '/S1_psnr_valid.txt',
               validate_psnr_epoch_mean, fmt='%.8f', delimiter=',')
    np.savetxt(err_log_dir + '/S1_loss_valid.txt',
               validate_loss_epoch_mean, fmt='%.8f', delimiter=',')

    summary_writer.close()

