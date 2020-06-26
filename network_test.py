"""
description: MR-KP (5-layer) Testing
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
from image_utils import save_image, save_exr

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

PADDING_HEIGHT = IMAGE_HEIGHT + 16

INPUT_CHANNEL = 10
TARGET_CHANNEL = 3

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset-name', type=str, 
	default="classroom")
parser.add_argument('-r', '--data-dir', type=str, 
	default='../../dataset/dataset_blockwise/BMFR-dataset')
parser.add_argument('-bs', '--batch-size', type=int, default=1)
parser.add_argument('-ts', '--test-size', type=int, default=60)
parser.add_argument('--export_exr',action='store_true')
parser.add_argument('--export_all',action='store_true')
args = parser.parse_args()

data_dir = args.data_dir
scene_test_list = args.dataset_name.split(' ')
test_batch_size = args.batch_size
test_per_scene = args.test_size

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
	train_input = tf.reshape(train_input, [IMAGE_HEIGHT,
										   IMAGE_WIDTH, INPUT_CHANNEL])

	train_target = tf.decode_raw(features['target'], tf.float16)
	train_target = tf.reshape(train_target, [IMAGE_HEIGHT,
											 IMAGE_WIDTH, TARGET_CHANNEL])
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

	test_data = dataLoader(data_dir=data_dir, subset='test',
						   image_start_idx=0,
						   img_per_scene=test_per_scene,
						   scene_list=scene_test_list)

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
	model = MyModel(input_shape=[None, None, None, INPUT_CHANNEL],
		target_shape=[None, None, None, TARGET_CHANNEL],
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
	saver.restore(sess, os.path.join(model_dir, 'best_model'))
	
	# Test
	print('Start Testing...')
	batch_cnt = 0
	sess.run(test_iterator.initializer)
	test_handle = sess.run(test_iterator.string_handle())

	while True:
		try:
			src_hdr_in, tgt_hdr_in = sess.run(next_element_large,
				feed_dict={handle_large: test_handle})
			src_hdr = np.zeros((test_batch_size, PADDING_HEIGHT, IMAGE_WIDTH, INPUT_CHANNEL))
			tgt_hdr = np.zeros((test_batch_size, PADDING_HEIGHT, IMAGE_WIDTH, TARGET_CHANNEL))
			src_hdr[:,0:IMAGE_HEIGHT,:,:] = src_hdr_in
			tgt_hdr[:,0:IMAGE_HEIGHT,:,:] = tgt_hdr_in

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
			with open(os.path.join(errorlog_dir, 'timeline.json'),'w') as wd:
				wd.write(ctf)

			tgt = tone_mapping(tgt_hdr_in)
			src_hdr_in = src_hdr_in[:,:,:,0:3] * src_hdr_in[:,:,:,3:6]
			src = tone_mapping(src_hdr_in)

			denoised_1_bd_tm = tone_mapping(denoised_1_bd)
			denoised_2_bd_tm = tone_mapping(denoised_2_bd)
			denoised_3_bd_tm = tone_mapping(denoised_3_bd)

			if args.export_all:
				noisy_1_tm = tone_mapping(noisy_1)
				noisy_2_tm = tone_mapping(noisy_2)
				noisy_3_tm = tone_mapping(noisy_3)

				denoised_1_tm = tone_mapping(denoised_hdr_1)
				denoised_2_tm = tone_mapping(denoised_hdr_2)
				denoised_3_tm = tone_mapping(denoised_hdr_3)

			for k in range(0, src_hdr.shape[0]): 
				idx_all = batch_cnt * test_batch_size + k
				save_image(tgt[k, :, :, :], os.path.join(result_dir, '%d_tgt.png'%idx_all), 'RGB')
				save_image(src[k, :, :, :], os.path.join(result_dir, '%d_src.png'%idx_all), 'RGB')
				save_image(denoised_1_bd_tm[k, 0:IMAGE_HEIGHT, :, :], 
					os.path.join(result_dir, '%d_rcn.png'%idx_all), 'RGB')

				if args.export_exr:
					save_exr(src_hdr_in[k,:,:,:], 
						os.path.join(result_dir, '%d_src.exr'%idx_all))
					save_exr(tgt_hdr_in[k,:,:,:], 
						os.path.join(result_dir, '%d_tgt.exr'%idx_all))
					save_exr(denoised_1_bd[k,0:IMAGE_HEIGHT,:,:], 
						os.path.join(result_dir, '%d_rcn.exr'%idx_all))

				if args.export_all:
					save_image(noisy_1_tm[k, 0:IMAGE_HEIGHT, :, :], 
						os.path.join(result_dir, '%d_full_res_noisy.png'%idx_all), 'RGB')
					save_image(noisy_2_tm[k, :, :, :], 
						os.path.join(result_dir, '%d_half_res_noisy.png'%idx_all), 'RGB')
					save_image(noisy_3_tm[k, :, :, :], 
						os.path.join(result_dir, '%d_quat_res_noisy.png'%idx_all), 'RGB')

					save_image(denoised_1_tm[k, 0:IMAGE_HEIGHT, :, :], 
						os.path.join(result_dir, '%d_full_res_denoised.png'%idx_all), 'RGB')
					save_image(denoised_2_tm[k, :, :, :], 
						os.path.join(result_dir, '%d_half_res_denoised.png'%idx_all), 'RGB')
					save_image(denoised_3_tm[k, :, :, :], 
						os.path.join(result_dir, '%d_quat_res_denoised.png'%idx_all), 'RGB')

					save_image(denoised_1_bd_tm[k, 0:IMAGE_HEIGHT, :, :], 
						os.path.join(result_dir, '%d_full_res_denoised_blended.png'%idx_all), 'RGB')
					save_image(denoised_2_bd_tm[k, :, :, :], 
						os.path.join(result_dir, '%d_half_res_denoised_blended.png'%idx_all), 'RGB')
					save_image(denoised_3_bd_tm[k, :, :, :], 
						os.path.join(result_dir, '%d_quat_res_denoised_blended.png'%idx_all), 'RGB')

					save_image(kernel_alpha_1[k, 0:IMAGE_HEIGHT, :, 0], 
						os.path.join(result_dir, '%d_full_alpha.png'%idx_all))
					save_image(kernel_alpha_2[k, :, :, 0], 
						os.path.join(result_dir, '%d_half_alpha.png'%idx_all))
					save_image(kernel_alpha_3[k, :, :, 0], 
						os.path.join(result_dir, '%d_quat_alpha.png'%idx_all))
			batch_cnt += 1
		except tf.errors.OutOfRangeError:
			print('Finish testing %d images.' % (batch_cnt * test_batch_size))
			break
	sess.close()



