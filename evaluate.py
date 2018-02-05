# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from classfile import CombinatorialTripletSet
import os.path
import time
from datetime import datetime
import numpy as np
from PIL import Image
import random
import tensorflow.contrib.slim as slim
from nets import resnet_v2
import socket

def getDist(feat,otherFeats):
    dist = (otherFeats - feat)**2
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    # dist = np.array([np.dot(feat,otherFeat) for otherFeat in otherFeats])
    return dist

test_file = './val.txt'
pretrained_net = './output/ckpts/checkpoint-201802051322_lr0pt001_outputSz128_margin0pt3-601'
# pretrained_net = './output/ckpts/TEST--90'
img_size = [256, 256]
crop_size = [227, 227]
if 'abby' in socket.gethostname().lower():
    mean_file = '/Users/abby/Documents/repos/triplepalooza/models/traffickcam/tc_mean_im.npy'
else:
    mean_file = '/project/focus/abby/triplepalooza/models/traffickcam/tc_mean_im.npy'

datestr = datetime.now().strftime("%Y%m%d%H%M")

c = tf.ConfigProto()
# c.gpu_options.visible_device_list="1,2"
c.gpu_options.visible_device_list="2"

# TODO: Fix issue where slim isn't using batch statistics -- need to save those during training
batch_size = 120

output_size = 128

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=False)

# feat = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("pool5:0"),3))
featLayer = 'resnet_v2_50/logits'
feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))

# Create data "batcher"
test_data = CombinatorialTripletSet(test_file, mean_file, img_size, crop_size, batch_size, isTraining=False)

# Create a saver for writing loading checkpoints.
saver = tf.train.Saver()

sess = tf.InteractiveSession(config=c)
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Here's where we need to load saved weights
saver.restore(sess, pretrained_net)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(len(test_data.files[ix]))]
# testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(10)]
numTestingIms = batch_size*(len(testingImsAndLabels)/batch_size)
testingImsAndLabels = testingImsAndLabels[:numTestingIms]
# numTestingIms = len(testingImsAndLabels)
random.shuffle(testingImsAndLabels)

testingFeats = np.empty((numTestingIms,feat.shape[-1]),dtype=np.float32)
testingIms = np.empty((numTestingIms),dtype=object)
testingLabels = np.empty((numTestingIms),dtype=np.int32)
num_iters = numTestingIms / batch_size

print 'Computing testing set features...'
for step in range(0,num_iters):
    print step, '/', num_iters
    if step == num_iters:
        end_ind = numTestingIms
    else:
        end_ind = step*batch_size+batch_size

    il = testingImsAndLabels[step*batch_size:end_ind]
    ims = [i[0] for i in il]
    testingIms[step*batch_size:end_ind] = ims
    labels = [i[1] for i in il]
    testingLabels[step*batch_size:end_ind] = labels
    batch = test_data.getBatchFromImageList(ims)

    while len(labels) < batch_size:
        labels += [labels[-1]]
        batch = np.vstack((batch,np.expand_dims(batch[-1],0)))

    ff = sess.run(feat, feed_dict={image_batch: batch, label_batch:labels})
    testingFeats[step*batch_size:end_ind,:] = ff[:len(il),:]

def combine_horz(ims):
    images = map(Image.open, [ims[0],ims[1],ims[2],ims[3],ims[4],ims[5],ims[6]])
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

out_dir = os.path.join('/project/focus/abby/toy_visualization/output/example_results/',datestr)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print out_dir

print 'Computing testing set distances...'
queryImsAndLabels = [(testingIms[idx],testingLabels[idx],idx) for idx in range(numTestingIms)]
ctr = 0
testingAccuracy = np.zeros((len(queryImsAndLabels),100))
for idx in range(len(queryImsAndLabels)):
    thisIm = queryImsAndLabels[idx][0]
    thisLabel = queryImsAndLabels[idx][1]
    thisFeat = testingFeats[queryImsAndLabels[idx][2],:]
    dists = getDist(thisFeat,testingFeats)
    sortedInds = np.argsort(dists)[1:]
    sortedIms = testingIms[sortedInds]
    if 'query' in thisIm:
        bad = [aa for aa in range(len(sortedIms)) if 'query' in sortedIms[aa]]
        mask = np.ones(len(sortedInds), dtype=bool)
        mask[bad] = False
        sortedInds = sortedInds[mask,...]

    sortedLabels = testingLabels[sortedInds]

    topHit = np.where(sortedLabels==thisLabel)[0][0]
    topHitIm = testingIms[sortedInds[topHit]]
    topMatchIm1 = testingIms[sortedInds[0]]
    topMatchIm2 = testingIms[sortedInds[1]]
    topMatchIm3 = testingIms[sortedInds[2]]
    topMatchIm4 = testingIms[sortedInds[3]]
    topMatchIm5 = testingIms[sortedInds[4]]
    new_im = combine_horz([thisIm,topMatchIm1,topMatchIm2,topMatchIm3,topMatchIm4,topMatchIm5,topHitIm])

    if thisLabel in sortedLabels[:100]:
        testingAccuracy[idx,topHit:] = 1

    # if ctr%10 == 0:
    #     print np.mean(testingAccuracy[:idx,:],axis=0)[0], np.mean(testingAccuracy[:idx,:]), np.mean(testingAccuracy[:idx,:],axis=0)[-1]
    save_path = os.path.join(out_dir,str(ctr)+'_'+str(topHit)+'.jpg')
    new_im.save(save_path)
    ctr += 1

# randomSuccess = np.zeros((len(queryImsAndLabels),100,1000))
# for ctr in range(1000):
#     for idx in range(len(queryImsAndLabels)):
#         thisLabel = queryImsAndLabels[idx][1]
#         randLabels = random.sample(testingLabels[np.arange(len(testingLabels))!=idx],100)
#         topHit = np.where(sortedLabels==thisLabel)[0][0]
#         if thisLabel in sortedLabels[:100]:
#             randomSuccess[idx,topHit:,ctr] = 1
#
# print '---Triplepalooza--'
# print 'Random:'
# print np.mean(np.mean(randomSuccess,axis=2),axis=0)
print 'Network: ', pretrained_net
print 'NN Test Accuracy: ',np.mean(testingAccuracy,axis=0)
print 'Example directory: ', out_dir
