"""
description: MR-KP (5-layer) Training
Github: https://github.com/xmeng525/MultiResolutionKernelPredictionCNN

@author: Xiaoxu Meng (xiaoxumeng1993@gmail.com)
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from data_loader import dataLoader
from network import MyModel
from image_utils import save_image, batch_psnr

INPUT_CHANNEL = 10
TARGET_CHANNEL = 3

# Export some validation results for debugging.
VALID_DISPLAY_LIST = [0, 3]

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--data-dir', type=str, 
    default='../../dataset/dataset_blockwise/BMFR-dataset')
parser.add_argument('-train', '--train-dataset-name', type=str, 
    default='living-room san-miguel sponza sponza-glossy sponza-moving-light')
parser.add_argument('-test', '--test-dataset-name', type=str, 
    default='classroom')
parser.add_argument('-bs', '--batch-size', type=int, default=1)
parser.add_argument('-ps', '--patch-size', type=int, default=128)
parser.add_argument('-ppi', '--patch-per-image', type=int, default=50)
parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001)
parser.add_argument('-ep', '--total-epochs', type=int, default=100)
parser.add_argument('-vi', '--valid-interval', type=int, default=1)
parser.add_argument('--export_all',action='store_true')
args = parser.parse_args()

data_dir = args.data_dir
scene_train_list = args.train_dataset_name.split(' ')
scene_valid_list = scene_train_list
scene_test_list = args.test_dataset_name.split(' ')
user_batch_size = args.batch_size
patch_width = args.patch_size
patch_height = args.patch_size
patch_per_img = args.patch_per_image
learning_rate = args.learning_rate
valid_interval = args.valid_interval
total_epochs = args.total_epochs

train_start_idx = 0
train_per_scene = 55

valid_start_idx = 55
valid_per_scene = 5

def tone_mapping(input_image):
    tone_mapped_color = np.clip(
        np.power(np.maximum(0., input_image), 0.454545), 0., 1.)
    return tone_mapped_color

def _parse_function(proto):  # for training data
    features = tf.parse_single_example(
        proto, features={
            'target': tf.FixedLenFeature([], tf.string),
            'input': tf.FixedLenFeature([], tf.string)})

    train_input = tf.decode_raw(features['input'], tf.float16)
    train_input = tf.reshape(train_input, [patch_height,
                                           patch_width, INPUT_CHANNEL])

    train_target = tf.decode_raw(features['target'], tf.float16)
    train_target = tf.reshape(train_target, [patch_height,
                                             patch_width, TARGET_CHANNEL])
    return (train_input, train_target)

if __name__ == "__main__":
    model_dir = os.path.join(scene_test_list[0], 'model')
    result_dir = os.path.join(scene_test_list[0], 'result', 'test_out')
    errorlog_dir = os.path.join(scene_test_list[0], 'errorlog')
    summarylog_dir = os.path.join(scene_test_list[0], 'summarylog')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(errorlog_dir, exist_ok=True)
    os.makedirs(summarylog_dir, exist_ok=True)

    train_data = dataLoader(data_dir=data_dir, subset='train',
                            patch_width=patch_width,
                            patch_height=patch_height,
                            image_start_idx=train_start_idx,
                            img_per_scene=train_per_scene,
                            patch_per_img=patch_per_img,
                            scene_list=scene_train_list)
    valid_data = dataLoader(data_dir=data_dir, subset='valid',
                            patch_width=patch_width,
                            patch_height=patch_height,
                            image_start_idx=valid_start_idx,
                            img_per_scene=valid_per_scene,
                            patch_per_img=patch_per_img,
                            scene_list=scene_valid_list)

    # Train
    train_dataset = tf.data.TFRecordDataset([train_data.dataset_name])
    # Parse the record into tensors.
    train_dataset = train_dataset.map(_parse_function)
    train_dataset = train_dataset.shuffle(buffer_size=2000)
    train_dataset = train_dataset.batch(user_batch_size)

    # Validate
    valid_dataset = tf.data.TFRecordDataset([valid_data.dataset_name])
    valid_dataset = valid_dataset.map(_parse_function)
    valid_dataset = valid_dataset.batch(user_batch_size)

    handle_small = tf.placeholder(tf.string, shape=[])
    iterator_structure_small = tf.data.Iterator.from_string_handle(
        handle_small, train_dataset.output_types, train_dataset.output_shapes)
    next_element_small = iterator_structure_small.get_next()
    train_iterator = train_dataset.make_initializable_iterator()
    valid_iterator = valid_dataset.make_initializable_iterator()

    # Model
    model = MyModel(input_shape=[None, None, None, INPUT_CHANNEL],
        target_shape=[None, None, None, TARGET_CHANNEL],
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

    print('Start Training: ')
    min_loss = 10000
    summary_merge = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(summarylog_dir, graph=sess.graph)
    begin_time_train = time.time()
    
    train_handle = sess.run(train_iterator.string_handle())
    valid_handle = sess.run(valid_iterator.string_handle())

    train_psnr_epoch_mean = []
    train_loss_epoch_mean = []
    valid_psnr_epoch_mean = []
    valid_loss_epoch_mean = []

    for epoch_i in range(total_epochs):
        # Training
        epoch_avg_loss_train = 0.0
        epoch_avg_psnr_train = 0.0
        batch_cnt = 0
        sess.run(train_iterator.initializer)
        while True:
            try:
                src_hdr, tgt_hdr = sess.run(next_element_small,
                    feed_dict={handle_small: train_handle})
                feed_dict = {ae_net['source']: src_hdr, ae_net['target']: tgt_hdr}
                summary, denoised_1_bd, batch_loss, _ = sess.run(
                    [summary_merge, ae_net['denoised_1_bd'], loss_grid_L1, train_step1], feed_dict)
                # summary_writer.add_summary(summary, epoch_i)

                _, batch_psnr_val = batch_psnr(tone_mapping(denoised_1_bd), tone_mapping(tgt_hdr))
                epoch_avg_psnr_train += batch_psnr_val
                epoch_avg_loss_train += batch_loss
                batch_cnt += 1
            except tf.errors.OutOfRangeError:
                epoch_avg_psnr_train /= batch_cnt
                epoch_avg_loss_train /= batch_cnt
                train_psnr_epoch_mean.append(epoch_avg_psnr_train)
                train_loss_epoch_mean.append(epoch_avg_loss_train)

                print('Epoch %d Train\n, psnr %.8f\nloss = %.8f'% (
                    epoch_i, epoch_avg_psnr_train, epoch_avg_loss_train))
                break
        # Validate
        should_validate = ((epoch_i + 1) % valid_interval == 0)
        if should_validate:
            epoch_avg_loss_valid = 0.0
            epoch_avg_psnr_valid = 0.0
            batch_cnt = 0
            sess.run(valid_iterator.initializer)
            while True:
                try:
                    src_hdr, tgt_hdr = sess.run(next_element_small,
                        feed_dict={handle_small: valid_handle})
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
                    src = tone_mapping(src_hdr[:,:,:,0:3])

                    _, batch_psnr_val = batch_psnr(denoised_1_bd_tm, tgt)
                    epoch_avg_psnr_valid += batch_psnr_val
                    epoch_avg_loss_valid += batch_loss

                    if batch_cnt in VALID_DISPLAY_LIST:
                        for k in range(tgt.shape[0]):
                            save_image(tgt[k, :, :, :],
                                os.path.join(result_dir, 'e%d_b%d_i%d_tgt.png'%(epoch_i, batch_cnt, k)), 'RGB')
                            save_image(denoised_1_bd_tm[k, :, :, :],
                                os.path.join(result_dir, 'e%d_b%d_i%d_rcn.png'%(epoch_i, batch_cnt, k)), 'RGB')
                            save_image(src[k,:,:,:],
                                os.path.join(result_dir, 'e%d_b%d_i%d_src.png'%(epoch_i, batch_cnt, k)), 'RGB')

                            if args.export_all:
                                save_image(noisy_1_tm[k, :, :, :], 
                                    os.path.join(result_dir, 'e%d_b%d_i%d_full_noisy.png'%(epoch_i, batch_cnt, k)))
                                save_image(noisy_2_tm[k, :, :, :], 
                                    os.path.join(result_dir, 'e%d_b%d_i%d_half_noisy.png'%(epoch_i, batch_cnt, k)))
                                save_image(noisy_3_tm[k, :, :, :], 
                                    os.path.join(result_dir, 'e%d_b%d_i%d_quat_noisy.png'%(epoch_i, batch_cnt, k)))

                                save_image(denoised_1_tm[k,:, :],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_full_denoised.png'%(epoch_i, batch_cnt, k)))
                                save_image(denoised_2_tm[k,:, :],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_half_denoised.png'%(epoch_i, batch_cnt, k)))
                                save_image(denoised_3_tm[k,:, :],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_quat_denoised.png'%(epoch_i, batch_cnt, k)))

                                save_image(denoised_1_bd_tm[k,:, :],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_full_denoised_blend.png'%(epoch_i, batch_cnt, k)))
                                save_image(denoised_2_bd_tm[k,:, :],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_half_denoised_blend.png'%(epoch_i, batch_cnt, k)))
                                save_image(denoised_3_bd_tm[k,:, :],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_quat_denoised_blend.png'%(epoch_i, batch_cnt, k)))

                                save_image(kernel_alpha_1[k,:, :, 0],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_full_alpha.png'%(epoch_i, batch_cnt, k)))
                                save_image(kernel_alpha_2[k,:, :, 0],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_half_alpha.png'%(epoch_i, batch_cnt, k)))
                                save_image(kernel_alpha_3[k,:, :, 0],
                                    os.path.join(result_dir, 'e%d_b%d_i%d_quat_alpha.png'%(epoch_i, batch_cnt, k)))
                    batch_cnt += 1
                except tf.errors.OutOfRangeError:
                    epoch_avg_psnr_valid /= batch_cnt
                    epoch_avg_loss_valid /= batch_cnt
                    valid_psnr_epoch_mean.append(epoch_avg_psnr_valid)
                    valid_loss_epoch_mean.append(epoch_avg_loss_valid)

                    print('Epoch %d Valid\n, psnr %.8f\nloss = %.8f'% (
                        epoch_i, epoch_avg_psnr_valid, epoch_avg_loss_valid))
                    if epoch_avg_loss_valid < min_loss:
                        print("best model saved")
                        saver.save(sess, os.path.join(model_dir, 'best_model'))
                        min_loss = epoch_avg_loss_valid
                    break            
        # Validate epoch finished.
    # Stage 1 Training finished.
    total_time = time.time() - begin_time_train
    print("Training done, total training time = %.4fs " % (total_time))

    np.savetxt(os.path.join(errorlog_dir, 'psnr_train.txt'),
        train_psnr_epoch_mean, fmt='%.8f', delimiter=',')
    np.savetxt(os.path.join(errorlog_dir, 'loss_train.txt'),
        train_loss_epoch_mean, fmt='%.8f', delimiter=',')
    np.savetxt(os.path.join(errorlog_dir, 'psnr_valid.txt'),
        valid_psnr_epoch_mean, fmt='%.8f', delimiter=',')
    np.savetxt(os.path.join(errorlog_dir, 'loss_valid.txt'),
        valid_loss_epoch_mean, fmt='%.8f', delimiter=',')

    summary_writer.close()
    sess.close()

