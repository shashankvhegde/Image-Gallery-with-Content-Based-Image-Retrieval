from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('/media/shashank/Personal/project') #Append the absolute path of project in your computer
sys.path.append('/media/shashank/Personal/project/vtranse') #Append the absolute path of project/vtranse in your computer
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

from django.shortcuts import render


# Create your views here
# print(cfg.DIR)

# def post_list(request):
#     return render(request, 'myfirstapp/post_list.html', {})
from django.shortcuts import render, redirect
from .models import Image, ImageTagRelationship
from .forms import FilterForm
from django.conf import settings
from darkflow.net.build import TFNet
import cv2

def image_list(request):
	print('Printing all Images')
	print(Image.objects.all())
	print(ImageTagRelationship.objects.all())
	if request.method == "POST":
		form  = FilterForm(request.POST)
		if form.is_valid():
			cleaned_data = form.cleaned_data;
			filterText = cleaned_data['filterText']
			print('FilterText' + str(filterText))
			image_list = []
			if len(filterText) > 0 :

				for image_tag_relationship in ImageTagRelationship.objects.filter(tag=filterText):
					image_list.append(image_tag_relationship.image)
			else:
				image_list = Image.objects.all()
	else:
		form = FilterForm()
		image_list = Image.objects.all()
	return render(request, 'myfirstapp/image_list.html', {'images':image_list, 'form':form})


def create_bounding_box(bottomright, topleft):
	bbox = np.zeros(4)
	bbox[0:4] = [topleft['x'], topleft['y'], bottomright['x'], bottomright['y']]
	return bbox

def create_and_save_roids(image_file, result):
	print (cfg.DIR)
	p = cfg.DIR
	image_full_path = p + 'myfirstapp/serve/image/' + image_file
	print(image_full_path)
	imgcv = cv2.imread(p + 'myfirstapp/serve/image/' + image_file)

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

	image_path = cfg.DIR + 'myfirstapp/serve/image/'

	save_input_path = cfg.DIR + 'myfirstapp/serve/save_roids/input/vrd_roidb.npz'

	file_path_use = file_path
	image_path_use = image_path
	save_path_use = save_input_path

	roidb = {}

	with open(file_path_use,'r') as f:
		data=json.load(f)

		roidb_temp = {}

		im = imgcv
		if type(im) == type(None):
			pass
		else:
			im_shape = np.shape(im)
			im_h = im_shape[0]
			im_w = im_shape[1]

			roidb_temp['image'] = image_full_path
			roidb_temp['width'] = im_w
			roidb_temp['height'] = im_h

			d = data_d
			relation_length = len(d)

			if relation_length == 0:
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

					triplet_object_list.append(relation['object'])
					triplet_subject_list.append(relation['subject'])

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
	print(roidb)
	return triplet_subject_list, triplet_object_list

def predicate_predictions(triplet_subject_list, triplet_object_list):
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
	saver = tf.train.Saver()

	with tf.Session() as sess:
		init = tf.global_variables_initializer()
		sess.run(init)
		saver.restore(sess, model_path)
		pred_roidb = {}
		# for roidb_id in range(N_test):
			# if (roidb_id+1)%10 == 0:
			# 	print(roidb_id + 1)
			# roidb_use = test_roidb[roidb_id]
		roidb_use = roidb_read
		if len(roidb_use['rela_gt']) == 0:
			pred_roidb.append({})
		else:
			pred_rela, pred_rela_score = vnet.test_predicate(sess, roidb_use)
			pred_roidb_temp = {}
			pred_roidb_temp['pred_rela'] = pred_rela
			pred_roidb_temp['pred_rela_score'] = pred_rela_score
			pred_roidb_temp['sub_box_dete'] = roidb_use['sub_box_gt']
			pred_roidb_temp['obj_box_dete'] = roidb_use['obj_box_gt']
			pred_roidb_temp['sub_dete'] = triplet_subject_list
			pred_roidb_temp['obj_dete'] = triplet_object_list

			pred_roidb = pred_roidb_temp
	roidb_output = pred_roidb
	print(roidb_output)

	objects_path = cfg.DIR + 'json_dataset/objects.json'
	relationships_path = cfg.DIR + 'json_dataset/predicates.json'

	print(len(roidb_output['sub_dete']))
	print(len(roidb_output['obj_dete']))

	retval = []
	ret_set = set()
	with open(relationships_path,'r') as rel_file:
		# objs = json.load(obj_file)
		rels = json.load(rel_file)

		# print(len(objs))
		# print(len(rels))
		for rel in range(len(roidb_output['pred_rela'])):
			sub = roidb_output['sub_dete'][rel]
			relid = int(roidb_output['pred_rela'][rel])
			obj = roidb_output['obj_dete'][rel]

			add_val = (sub + " " + str(rels[relid]) + " " + obj)
			print(sub
				+ " "
				+ str(rels[relid])
				+ " "
				+ obj
				)
			ret_set.add(add_val)
	np.savez(save_output_path, roidb=roidb_output)
	retval = list(ret_set)
	return retval

def insert_picture(request):
	if(request.method == 'GET'):

		print('Reached view')
		return render(request, 'myfirstapp/insert_picture.html', {})

	elif((request.method == 'POST') and (request.FILES['image_file'])):

		image_file = request.FILES['image_file']

		uploaded_image = Image.objects.create(image = image_file)
		# options = {"model": "/home/shashank/Desktop/project/darkflow/cfg/yolov2.cfg"
		# 		, "load": "/home/shashank/Desktop/project/darkflow/bin/yolov2.weights", "threshold": 0.1}
		model_path = cfg.DIR + '/darkflow/cfg/yolov2.cfg'
		load_path = cfg.DIR + '/darkflow/bin/yolov2.weights'

		options = {"model": model_path
				, "load": load_path, "threshold": 0.1}
		tfnet = TFNet(options)

		imgcv = cv2.imread(settings.MEDIA_ROOT + '/user_images/' + image_file.name)
		result = tfnet.return_predict(imgcv)
		print(result)
		labels = []
		label_set = set()
		# for desc in result:
		# 	labels.append(Tag.objects.create(label = desc['label']))
		# # print('Printing Labels Extracted from output ' + str(labels))
		# for label in labels:
		# 	ImageTagRelationship.objects.create(image = uploaded_image, tag = label)
		# # print(str(ImageTagRelationship.objects.all()))
		for desc in result:
			label_set.add(desc['label'])
		labels = list(label_set)
		print('Printing Labels Extracted from output ' + str(labels))
		for label in labels:
			ImageTagRelationship.objects.create(image = uploaded_image, tag = label)
		print(str(ImageTagRelationship.objects.all()))

		triplet_subject_list, triplet_object_list = create_and_save_roids(image_file.name, result)

		predicates = predicate_predictions(triplet_subject_list, triplet_object_list)

		print('Printing Predicate Labels Extracted from output ' + str(predicates))
		for pred in predicates:
			ImageTagRelationship.objects.create(image = uploaded_image, tag = pred)
		print(str(ImageTagRelationship.objects.all()))
		return redirect('image_list')
