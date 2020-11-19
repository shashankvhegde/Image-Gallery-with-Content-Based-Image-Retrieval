from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/media/shashank/Personal/project')
sys.path.append('/media/shashank/Personal/project/vtranse') # Location of VTramsE installation
import numpy as np
import xlwt
import h5py
import cv2
from vtranse.model.config import cfg
from vtranse.model.ass_fun import *
from darkflow.net.build import TFNet
import json


import tensorflow as tf
# import numpy as np
# from model.config import cfg
# from model.ass_fun import *
from vtranse.net.vtranse_vgg import VTranse

def create_bounding_box(bottomright, topleft):
	bbox = np.zeros(4)
	bbox[0:4] = [topleft['x'], topleft['y'], bottomright['x'], bottomright['y']]
	return bbox

print (cfg.DIR) # This line prints /media/shashank/Personal/project/ not /media/shashank/Personal/ImageRelationshipTest/
################################################################################
p = cfg.DIR
# m = p + 'vtranse/darkflow/cfg/yolov2.cfg'
# l = p + 'vtranse/darkflow/bin/yolov2.weights'
options = {"model": "/home/shashank/Desktop/project/darkflow/cfg/yolov2.cfg"
				, "load": "/home/shashank/Desktop/project/darkflow/bin/yolov2.weights", "threshold": 0.1}
tfnet = TFNet(options)

print(p + 'myfirstapp/serve/image/' + '1602315_961e6acf72_b.jpg')
imgcv = cv2.imread(p + 'myfirstapp/serve/image/' + '1602315_961e6acf72_b.jpg')
# cv2.imshow('image', imgcv)
result = tfnet.return_predict(imgcv)
# print("Result with Old Bounding Box")
# for i in range(len(result)):
# 	print(result[i])
# print(result)

result_with_req_bbox = []
epsilon = 0.000001
for i in range(len(result)):
	temp = {}
	if((result[i]['confidence'] - 0.50) > epsilon):
		temp['label'] = result[i]['label']
		temp['bbox'] = create_bounding_box(result[i]['bottomright'], result[i]['topleft'])
		result_with_req_bbox.append(temp)
print("Result with required bounding box format")
for i in range(len(result_with_req_bbox)):
	print(result_with_req_bbox[i])


# Creating pairs

data_d = []

for i in range(len(result_with_req_bbox)-1):
	for j in range(i+1, len(result_with_req_bbox)):
		temp = {}
		sub = result_with_req_bbox[i]['label']
		sbbox = result_with_req_bbox[i]['bbox']
		ob = result_with_req_bbox[j]['label']
		obbox = result_with_req_bbox[j]['bbox']

		temp['subject'] = sub
		temp['object'] = ob
		temp['sub_bbox'] = sbbox
		temp['obj_bbox'] = obbox

		data_d.append(temp)

print("Printing new 'd'")
print(data_d)


################################################################################
N_each_pred = cfg.VRD_BATCH_NUM

file_path = cfg.DIR + 'myfirstapp/serve/image/annotations_train.json'
# test_path = cfg.DIR + 'myfirstapp/serve/image/annotations_test.json'

image_path = cfg.DIR + 'myfirstapp/serve/image/'

save_input_path = cfg.DIR + 'myfirstapp/serve/save_roids/input/vrd_roidb.npz'

# Fine up till here.

file_path_use = file_path
image_path_use = image_path
save_path_use = save_input_path
# roidb = []
roidb = {}

with open(file_path_use,'r') as f:
	data=json.load(f)
	# image_name = data.keys() -> Casting to list to handle an error: Shashank

	# image_name = list(data.keys())
	image_name = '1602315_961e6acf72_b.jpg'
	# image_name = '111083560_d11369b21d_b.jpg'

	print(image_name)
	len_img = len(image_name)
	t = 0
	# image_id = '111083560_d11369b21d_b.jpg'
	# for image_id in range(len_img):
	# if (image_id+1)%1000 == 0:
	# 	print('image id is {0}'.format(image_id+1))
	roidb_temp = {}
	# image_full_path = image_path_use + image_name[image_id]
	image_full_path = image_path_use + image_name
	im = cv2.imread(image_full_path)
	im = imgcv
	if type(im) == type(None):
		# continue
		pass
	else:
		im_shape = np.shape(im)
		im_h = im_shape[0]
		im_w = im_shape[1]

		roidb_temp['image'] = image_full_path
		roidb_temp['width'] = im_w
		roidb_temp['height'] = im_h


		# filler_input_labels_image_name = '111083560_d11369b21d_b.jpg'
		filler_input_labels_image_name ='1602315_961e6acf72_b.jpg'

		d = data[filler_input_labels_image_name]
		# d = data[image_name]

		d = data_d  # This minute changes all the code below. Shashank Hegde.
		relation_length = len(d)

		if relation_length == 0:
			# continue
			pass
		else:
			sb_new = np.zeros(shape=[relation_length,4])
			ob_new = np.zeros(shape=[relation_length,4])
			rela = np.zeros(shape=[relation_length,])
			obj = np.zeros(shape=[relation_length,])
			subj = np.zeros(shape=[relation_length,])


			triplet_subject_list = []
			triplet_object_list = []
			for relation_id in range(relation_length):
				relation = d[relation_id]

				# obj[relation_id] = relation['object']['category']
				# subj[relation_id] = relation['subject']['category']
				# rela[relation_id] = relation['predicate']

				triplet_object_list.append(relation['object'])
				triplet_subject_list.append(relation['subject'])

				# ob_temp = relation['object']['bbox']
				# sb_temp = relation['subject']['bbox']

				# ob = [ob_temp[0],ob_temp[1],ob_temp[2],ob_temp[3]]
				# sb = [sb_temp[0],sb_temp[1],sb_temp[2],sb_temp[3]]

				# ob_new[relation_id][0:4] = [ob[2],ob[0],ob[3],ob[1]]
				# sb_new[relation_id][0:4] = [sb[2],sb[0],sb[3],sb[1]]

				ob_new[relation_id][0:4] = relation['obj_bbox']
				sb_new[relation_id][0:4] = relation['sub_bbox']

			roidb_temp['sub_box_gt'] = sb_new[:] + 0.0
			roidb_temp['obj_box_gt'] = ob_new + 0.0
			roidb_temp['sub_gt'] = subj + 0.0
			roidb_temp['obj_gt'] = obj + 0.0
			roidb_temp['rela_gt'] = rela + 0.0
			roidb_temp['index_pred'] = generate_batch(len(rela), N_each_pred)
			# roidb.append(roidb_temp)
			roidb = roidb_temp
np.savez(save_path_use, roidb=roidb)
print("Printing roidb")
print(roidb) #: Successful

# {'image': '/media/shashank/Personal/ImageRelationshipTest/vtranse/serve/image/111083560_d11369b21d_b.jpg',
#  'obj_box_gt': array([[ 416.,  389.,  525.,  486.],
#        [   3.,   82., 1021.,  767.],
#        [  91.,  341.,  540.,  582.]]),
#  'height': 768,
#  'obj_gt': array([96., 51., 61.]),
#  'sub_gt': array([81., 61., 28.]),
#  'sub_box_gt': array([[ 399.,  514.,  491.,  568.],
#        [  91.,  341.,  540.,  582.],
#        [ 386.,  282., 1002.,  573.]]),
#  'rela_gt': array([15., 34.,  3.]),
#  'width': 1024,
#  'index_pred': array([0, 1, 2, 1, 2, 0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 2,
#        0, 1, 2, 1, 0, 2, 1, 1], dtype=int32)}

# Obtained roidb i.e roidb_use to pass to test_predicate and also save it to serve/input


# Using test_predicate below. Copying code from test_vrd_vgg_pred.py

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import tensorflow as tf
# import numpy as np
# from model.config import cfg
# from model.ass_fun import *
# from net.vtranse_vgg import VTranse

N_cls = cfg.VRD_NUM_CLASS
N_rela = cfg.VRD_NUM_RELA
N_each_batch = cfg.VRD_BATCH_NUM

index_sp = False
index_cls = False

vnet = VTranse()
vnet.create_graph(N_each_batch, index_sp, index_cls, N_cls, N_rela)

roidb_path = cfg.DIR + 'myfirstapp/serve/save_roids/input/vrd_roidb.npz'
model_path = cfg.DIR + 'vtranse/pred_para/vrd_vgg/vrd_vgg0001.ckpt'
save_output_path = cfg.DIR + 'myfirstapp/serve/save_roids/output/vrd_pred_roidb.npz'

roidb_read = read_roidb(roidb_path)
# train_roidb = roidb_read['train_roidb']
# test_roidb = roidb_read['test_roidb']
# N_train = len(train_roidb)
# # N_test = len(test_roidb)
# # print("N_test: " + str(N_test))
# # N_test = 500
# N_test = 10

saver = tf.train.Saver()

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	saver.restore(sess, model_path)
	# pred_roidb = []
	pred_roidb = {}
	# for roidb_id in range(N_test):
		# if (roidb_id+1)%10 == 0:
		# 	print(roidb_id + 1)
		# roidb_use = test_roidb[roidb_id]
	roidb_use = roidb_read
	if len(roidb_use['rela_gt']) == 0:
		pred_roidb.append({})
		# continue
	else:
		pred_rela, pred_rela_score = vnet.test_predicate(sess, roidb_use)
		pred_roidb_temp = {}
		# pred_roidb_temp = {'pred_rela': pred_rela, 'pred_rela_score': pred_rela_score,
		# 					'sub_box_dete': roidb_use['sub_box_gt'], 'obj_box_dete': roidb_use['obj_box_gt'],
		# 					'sub_dete': roidb_use['sub_gt'], 'obj_dete': roidb_use['obj_gt']}
		pred_roidb_temp['pred_rela'] = pred_rela
		pred_roidb_temp['pred_rela_score'] = pred_rela_score
		pred_roidb_temp['sub_box_dete'] = roidb_use['sub_box_gt']
		pred_roidb_temp['obj_box_dete'] = roidb_use['obj_box_gt']
		# pred_roidb_temp['sub_dete'] = roidb_use['sub_gt']
		# pred_roidb_temp['obj_dete'] = roidb_use['obj_gt']
		pred_roidb_temp['sub_dete'] = triplet_subject_list
		pred_roidb_temp['obj_dete'] = triplet_object_list
		# pred_roidb_temp['sub_dete'] = pred_rela['sub_cls_pred'][:]
		# pred_roidb_temp['obj_dete'] = pred_rela['ob_cls_pred'][:]

		# pred_roidb.append(pred_roidb_temp)
		pred_roidb = pred_roidb_temp
roidb_output = pred_roidb

print(roidb_output)

objects_path = cfg.DIR + 'json_dataset/objects.json'
relationships_path = cfg.DIR + 'json_dataset/predicates.json'
:

print(len(roidb_output['sub_dete']))
print(len(roidb_output['obj_dete']))

with open(relationships_path,'r') as rel_file:
	# objs = json.load(obj_file)
	rels = json.load(rel_file)

	# print(len(objs))
	# print(len(rels))
	for rel in range(len(roidb_output['pred_rela'])):
		sub = roidb_output['sub_dete'][rel]
		relid = int(roidb_output['pred_rela'][rel])
		obj = roidb_output['obj_dete'][rel]
		print(sub
			+ " "
			+ str(rels[relid])
			+ " "
			+ obj
			)


np.savez(save_output_path, roidb=roidb_output)
