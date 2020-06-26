"""
description: MR-KP (5-layer) Testing
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

# Batch Parameters
user_batch_size = 10 # for training and validation
test_batch_size = 1 # for test

patch_width = 128
patch_height = 128

original_width = 1280  # 1024
original_height = 720  # 576

whole_width = original_width
whole_height = original_height + 16

test_per_scene = 60
test_start_idx = 0

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

def _parse_function_testdata(proto):
    # with tf.name_scope("parse_test_data"):
    features = tf.parse_single_example(
        proto, features={
            'target': tf.FixedLenFeature([], tf.string),
            'input': tf.FixedLenFeature([], tf.string)})

    train_input = tf.decode_raw(features['input'], tf.float16)
    train_input = tf.reshape(train_input, [whole_height,
                                           whole_width, input_channels])

    train_target = tf.decode_raw(features['target'], tf.float16)
    train_target = tf.reshape(train_target, [whole_height,
                                             whole_width, target_channels])
    return (train_input, train_target)

def my_mkdir(name):
    if not os.path.exists(name):
        print('creating folder', name)
        os.makedirs(name)

if __name__ == "__main__":
    # Summary log directory
    summary_dir = r'./summary_log'
    result_dir = r'./result/stage1'
    my_mkdir(result_dir)
    err_log_dir = r'./errorlogs'
    model_dir = r'./model'
    stg1_model_dir = os.path.join(model_dir, 'stage1')

    seed = 1
    rng = np.random.RandomState(seed)
    test_data = dataLoader(data_dir=data_dir, subset='test',
                           patch_width=patch_width,
                           patch_height=patch_height,
                           image_start_idx=test_start_idx,
                           img_per_scene=test_per_scene,
                           patch_per_img=patch_per_img,
                           scene_list=scene_test_list, rng=rng)

    # Test
    test_dataset = tf.data.TFRecordDataset([test_data.dataset_name])
    test_dataset = test_dataset.map(_parse_function_testdata)
    test_dataset = test_dataset.batch(test_batch_size)

    handle_large = tf.placeholder(tf.string, shape=[])
    iterator_structure_large = tf.data.Iterator.from_string_handle(
         handle_large, test_dataset.output_types, test_dataset.output_shapes)
    next_element_large = iterator_structure_large.get_next()
    test_iterator = test_dataset.make_initializable_iterator()

    # Model
    model = MyModel(input_shape=[None, None, None, input_channels],
                    target_shape=[None, None, None, target_channels],
                    loss_name="L1", if_albedo_in_training=False)
    with tf.device("/gpu:0"):
        ae_net = model.inference()

    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True, graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    opts1 = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts1)

    opts2 = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    params = tf.profiler.profile(sess.graph, run_meta=run_metadata, cmd='op', options=opts2)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(stg1_model_dir, 'best_model'))
    
    # Test
    print('Start Testing...')

    epoch_avg_psnr_test = 0.0
    epoch_avg_loss_test = 0.0

    batch_cnt = 0

    sess.run(test_iterator.initializer)
    test_handle = sess.run(test_iterator.string_handle())

    while True:
        try:
            src_hdr, tgt_hdr = sess.run(next_element_large,
                feed_dict={handle_large: test_handle})
            feed_dict = {ae_net['source']: src_hdr, ae_net['target']: tgt_hdr}
            denoised_1_bd, noisy_1, denoised_hdr_1, kernel_alpha_1, \
            denoised_2_bd, noisy_2, denoised_hdr_2, kernel_alpha_2, \
            denoised_3_bd, noisy_3, denoised_hdr_3, kernel_alpha_3, \
            batch_loss = sess.run(
                [
                    ae_net['denoised_1_bd'], ae_net['noisy_1'], ae_net['denoised_hdr_1'], ae_net['kernel_alpha_1'],
                    ae_net['denoised_2_bd'], ae_net['noisy_2'], ae_net['denoised_hdr_2'], ae_net['kernel_alpha_2'],
                    ae_net['denoised_3_bd'], ae_net['noisy_3'], ae_net['denoised_hdr_3'], ae_net['kernel_alpha_3'],
                    ae_net['loss_denoised']
                ], feed_dict, options=run_options, run_metadata=run_metadata)

            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format(show_memory=True)
            with open(os.path.join(err_log_dir, 'timeline.json'),'w') as wd:
                wd.write(ctf)

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

            _, batch_psnr_val = batch_psnr(
                denoised_1_bd_tm[:, 0:original_height, 0:original_width,:], 
                tgt[:, 0:original_height, 0:original_width,:])

            epoch_avg_psnr_test += batch_psnr_val
            epoch_avg_loss_test += batch_loss

            for k in range(0, tgt.shape[0]): 
                idx_all = batch_cnt * test_batch_size + k
                save_image_pil(result_dir, tgt[k, :, :, :], str(idx_all) + '_tgt')
                
                save_image_pil(result_dir, noisy_1_tm[k, :, :, :], str(idx_all) + '_1_1_noisy')
                # save_image_pil(result_dir, noisy_2_tm[k, :, :, :], str(idx_all) + '_2_1_noisy')
                # save_image_pil(result_dir, noisy_3_tm[k, :, :, :], str(idx_all) + '_3_1_noisy')

                # save_image_pil(result_dir, denoised_1_tm[k, :, :, :], str(idx_all) + '_1_2_denoised')
                # save_image_pil(result_dir, denoised_2_tm[k, :, :, :], str(idx_all) + '_2_2_denoised')
                # save_image_pil(result_dir, denoised_3_tm[k, :, :, :], str(idx_all) + '_3_2_denoised')

                save_image_pil(result_dir, denoised_1_bd_tm[k, :, :, :], str(idx_all) + '_1_3_final')
                # save_image_pil(result_dir, denoised_2_bd_tm[k, :, :, :], str(idx_all) + '_2_3_final')
                # save_image_pil(result_dir, denoised_3_bd_tm[k, :, :, :], str(idx_all) + '_3_3_final')

                # save_image_gray(result_dir, kernel_alpha_1[k, :, :, 0], str(idx_all) + '_1_4_alpha')
                # save_image_gray(result_dir, kernel_alpha_2[k, :, :, 0], str(idx_all) + '_2_4_alpha')
                # save_image_gray(result_dir, kernel_alpha_3[k, :, :, 0], str(idx_all) + '_3_4_alpha')

                save_image_pil(result_dir, src[k, :, :, :], str(idx_all) + '_src')
            batch_cnt += 1
        except tf.errors.OutOfRangeError:
            epoch_avg_psnr_test /= batch_cnt
            epoch_avg_loss_test /= batch_cnt

            print('Test ends\npsnr = %.4f\nloss = %.4f' % (epoch_avg_psnr_test, epoch_avg_loss_test))
            break
    sess.close()



