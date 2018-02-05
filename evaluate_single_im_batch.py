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
pretrained_net = './output/ckpts/checkpoint-201802010945_lr0pt0001_outputSz128_margin0pt3-1264'
# pretrained_net = './output/ckpts/TEST--90'
img_size = [256, 256]
crop_size = [227, 227]
if 'abby' in socket.gethostname().lower():
    mean_file = '/Users/abby/Documents/repos/triplepalooza/models/traffickcam/tc_mean_im.npy'
else:
    mean_file = '/project/focus/abby/triplepalooza/models/traffickcam/tc_mean_im.npy'

datestr = datetime.now().strftime("%Y%m%d%H%M")

c = tf.ConfigProto()
c.gpu_options.visible_device_list="1"

# TODO: Fix issue where slim isn't using batch statistics -- need to save those during training
batch_size = 1

output_size = 128

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=False, reuse=tf.AUTO_REUSE, scope='resnet_v2_50')


feat = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"),3))

# TODO: Maybe something to do with tf.train.get_or_create_global_step? ARGGGGHHHHHH

b1_varvar1 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b1_varvar2 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_1/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b1_varvar3 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_1/bottleneck_v2/preact/moving_variance:0")
b1_varvar4 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_2/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b1_varvar5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_2/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b1_varvar6 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_2/bottleneck_v2/preact/moving_variance:0")
b1_varvar7 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_3/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b1_varvar8 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_3/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b1_varvar9 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block1/unit_3/bottleneck_v2/preact/moving_variance:0")

b2_varvar1 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_1/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b2_varvar2 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_1/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b2_varvar3 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_1/bottleneck_v2/preact/moving_variance:0")
b2_varvar4 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_2/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b2_varvar5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_2/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b2_varvar6 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_2/bottleneck_v2/preact/moving_variance:0")
b2_varvar7 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_3/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b2_varvar8 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_3/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b2_varvar9 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_3/bottleneck_v2/preact/moving_variance:0")
b2_varvar10 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_4/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b2_varvar11 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_4/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b2_varvar12 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block2/unit_4/bottleneck_v2/preact/moving_variance:0")

b3_varvar1 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_1/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b3_varvar2 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_1/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b3_varvar3 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_1/bottleneck_v2/preact/moving_variance:0")
b3_varvar4 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_2/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b3_varvar5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_2/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b3_varvar6 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_2/bottleneck_v2/preact/moving_variance:0")
b3_varvar7 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_3/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b3_varvar8 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_3/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b3_varvar9 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_3/bottleneck_v2/preact/moving_variance:0")
b3_varvar10 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_4/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b3_varvar11 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_4/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b3_varvar12 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_4/bottleneck_v2/preact/moving_variance:0")
b3_varvar13 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_5/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b3_varvar14 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_5/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b3_varvar15 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_5/bottleneck_v2/preact/moving_variance:0")
b3_varvar16 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_6/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b3_varvar17 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_6/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b3_varvar18 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block3/unit_6/bottleneck_v2/preact/moving_variance:0")

b4_varvar1 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_1/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b4_varvar2 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_1/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b4_varvar3 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_1/bottleneck_v2/preact/moving_variance:0")
b4_varvar4 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_2/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b4_varvar5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_2/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b4_varvar6 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_2/bottleneck_v2/preact/moving_variance:0")
b4_varvar7 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_3/bottleneck_v2/conv1/BatchNorm/moving_variance:0")
b4_varvar8 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/BatchNorm/moving_variance:0")
b4_varvar9 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/block4/unit_3/bottleneck_v2/preact/moving_variance:0")

pn_varvar = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/postnorm/moving_variance:0")

# Create data "batcher"
test_data = CombinatorialTripletSet(test_file, mean_file, img_size, crop_size, batch_size, isTraining=False)

# Create a saver for writing loading checkpoints.
saver = tf.train.Saver()

sess = tf.Session(config=c)
init_op = tf.global_variables_initializer()
sess.run(init_op)
# Here's where we need to load saved weights
saver.restore(sess, pretrained_net)

b1_vv1_1, b1_vv2_1, b1_vv3_1, b1_vv4_1, b1_vv5_1, b1_vv6_1, b1_vv7_1, b1_vv8_1, b1_vv9_1 = sess.run([b1_varvar1,b1_varvar2,b1_varvar3,b1_varvar4,b1_varvar5,b1_varvar6,b1_varvar7,b1_varvar8,b1_varvar9])
b2_vv1_1, b2_vv2_1, b2_vv3_1, b2_vv4_1, b2_vv5_1, b2_vv6_1, b2_vv7_1, b2_vv8_1, b2_vv9_1, b2_vv10_1, b2_vv11_1, b2_vv12_1 = sess.run([b2_varvar1,b2_varvar2,b2_varvar3,b2_varvar4,b2_varvar5,b2_varvar6,b2_varvar7,b2_varvar8,b2_varvar9,b2_varvar10,b2_varvar11,b2_varvar12])
b3_vv1_1, b3_vv2_1, b3_vv3_1, b3_vv4_1, b3_vv5_1, b3_vv6_1, b3_vv7_1, b3_vv8_1, b3_vv9_1, b3_vv10_1, b3_vv11_1, b3_vv12_1,b3_vv13_1,b3_vv14_1,b3_vv15_1,b3_vv16_1,b3_vv17_1,b3_vv18_1 = sess.run([b3_varvar1,b3_varvar2,b3_varvar3,b3_varvar4,b3_varvar5,b3_varvar6,b3_varvar7,b3_varvar8,b3_varvar9,b3_varvar10,b3_varvar11,b3_varvar12,b3_varvar13,b3_varvar14,b3_varvar15,b3_varvar16,b3_varvar17,b3_varvar18])
b4_vv1_1, b4_vv2_1, b4_vv3_1, b4_vv4_1, b4_vv5_1, b4_vv6_1, b4_vv7_1, b4_vv8_1, b4_vv9_1 = sess.run([b4_varvar1,b4_varvar2,b4_varvar3,b4_varvar4,b4_varvar5,b4_varvar6,b4_varvar7,b4_varvar8,b4_varvar9])
pn_vv_1 = sess.run(pn_varvar)

# testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(len(test_data.files[ix]))]
testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(10)]
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
    il = testingImsAndLabels[step]
    testingIms[step] = il[0]
    testingLabels[step] = il[1]
    batch = test_data.getBatchFromImageList([il[0]])
    ff = sess.run([feat], feed_dict={image_batch: batch, label_batch:[il[1]]})
    b1_vv1_2, b1_vv2_2, b1_vv3_2, b1_vv4_2, b1_vv5_2, b1_vv6_2, b1_vv7_2, b1_vv8_2, b1_vv9_2 = sess.run([b1_varvar1,b1_varvar2,b1_varvar3,b1_varvar4,b1_varvar5,b1_varvar6,b1_varvar7,b1_varvar8,b1_varvar9])
    b2_vv1_2, b2_vv2_2, b2_vv3_2, b2_vv4_2, b2_vv5_2, b2_vv6_2, b2_vv7_2, b2_vv8_2, b2_vv9_2, b2_vv10_2, b2_vv11_2, b2_vv12_2 = sess.run([b2_varvar1,b2_varvar2,b2_varvar3,b2_varvar4,b2_varvar5,b2_varvar6,b2_varvar7,b2_varvar8,b2_varvar9,b2_varvar10,b2_varvar11,b2_varvar12])
    b3_vv1_2, b3_vv2_2, b3_vv3_2, b3_vv4_2, b3_vv5_2, b3_vv6_2, b3_vv7_2, b3_vv8_2, b3_vv9_2, b3_vv10_2, b3_vv11_2, b3_vv12_2,b3_vv13_2,b3_vv14_2,b3_vv15_2 = sess.run([b3_varvar1,b3_varvar2,b3_varvar3,b3_varvar4,b3_varvar5,b3_varvar6,b3_varvar7,b3_varvar8,b3_varvar9,b3_varvar10,b3_varvar11,b3_varvar12,b3_varvar13,b3_varvar14,b3_varvar15])
    b4_vv1_2, b4_vv2_2, b4_vv3_2, b4_vv4_2, b4_vv5_2, b4_vv6_2, b4_vv7_2, b4_vv8_2, b4_vv9_2 = sess.run([b4_varvar1,b4_varvar2,b4_varvar3,b4_varvar4,b4_varvar5,b4_varvar6,b4_varvar7,b4_varvar8,b4_varvar9])
    pn_vv_2 = sess.run(pn_varvar)

    if not (b1_vv1_1 == b1_vv1_2).all():
        print 'b1_vv1'
    if not (b1_vv2_1 == b1_vv2_2).all():
        print 'b1_vv2'
    if not (b1_vv3_1 == b1_vv3_2).all():
        print 'b1_vv3'
    if not (b1_vv4_1 == b1_vv4_2).all():
        print 'b1_vv4'
    if not (b1_vv5_1 == b1_vv5_2).all():
        print 'b1_vv5'
    if not (b1_vv6_1 == b1_vv6_2).all():
        print 'b1_vv6'
    if not (b1_vv7_1 == b1_vv7_2).all():
        print 'b1_vv7'
    if not (b1_vv8_1 == b1_vv8_2).all():
        print 'b1_vv8'
    if not (b1_vv9_1 == b1_vv9_2).all():
        print 'b1_vv9'

    if not (b2_vv1_1 == b2_vv1_2).all():
        print 'b2_vv1'
    if not (b2_vv2_1 == b2_vv2_2).all():
        print 'b2_vv2'
    if not (b2_vv3_1 == b2_vv3_2).all():
        print 'b2_vv3'
    if not (b2_vv4_1 == b2_vv4_2).all():
        print 'b2_vv4'
    if not (b2_vv5_1 == b2_vv5_2).all():
        print 'b2_vv5'
    if not (b2_vv6_1 == b2_vv6_2).all():
        print 'b2_vv6'
    if not (b2_vv7_1 == b2_vv7_2).all():
        print 'b2_vv7'
    if not (b2_vv8_1 == b2_vv8_2).all():
        print 'b2_vv8'
    if not (b2_vv9_1 == b2_vv9_2).all():
        print 'b2_vv9'
    if not (b2_vv10_1 == b2_vv10_2).all():
        print 'b2_vv10'
    if not (b2_vv11_1 == b2_vv11_2).all():
        print 'b2_vv11'
    if not (b2_vv12_1 == b2_vv12_2).all():
        print 'b2_vv12'

    if not (b3_vv1_1 == b3_vv1_2).all():
        print 'b3_vv1'
    if not (b3_vv2_1 == b3_vv2_2).all():
        print 'b3_vv2'
    if not (b3_vv3_1 == b3_vv3_2).all():
        print 'b3_vv3'
    if not (b3_vv4_1 == b3_vv4_2).all():
        print 'b3_vv4'
    if not (b3_vv5_1 == b3_vv5_2).all():
        print 'b3_vv5'
    if not (b3_vv6_1 == b3_vv6_2).all():
        print 'b3_vv6'
    if not (b3_vv7_1 == b3_vv7_2).all():
        print 'b3_vv7'
    if not (b3_vv8_1 == b3_vv8_2).all():
        print 'b3_vv8'
    if not (b3_vv9_1 == b3_vv9_2).all():
        print 'b3_vv9'
    if not (b3_vv10_1 == b3_vv10_2).all():
        print 'b3_vv10'
    if not (b3_vv11_1 == b3_vv11_2).all():
        print 'b3_vv11'
    if not (b3_vv12_1 == b3_vv12_2).all():
        print 'b3_vv12'
    if not (b3_vv13_1 == b3_vv13_2).all():
        print 'b3_vv13'
    if not (b3_vv14_1 == b3_vv14_2).all():
        print 'b3_vv14'
    if not (b3_vv15_1 == b3_vv15_2).all():
        print 'b3_vv15'

    if not (b4_vv1_1 == b4_vv1_2).all():
        print 'b4_vv1'
    if not (b4_vv2_1 == b4_vv2_2).all():
        print 'b4_vv2'
    if not (b4_vv3_1 == b4_vv3_2).all():
        print 'b4_vv3'
    if not (b4_vv4_1 == b4_vv4_2).all():
        print 'b4_vv4'
    if not (b4_vv5_1 == b4_vv5_2).all():
        print 'b4_vv5'
    if not (b4_vv6_1 == b4_vv6_2).all():
        print 'b4_vv6'
    if not (b4_vv7_1 == b4_vv7_2).all():
        print 'b4_vv7'
    if not (b4_vv8_1 == b4_vv8_2).all():
        print 'b4_vv8'
    if not (b4_vv9_1 == b4_vv9_2).all():
        print 'b4_vv9'

    if not (pn_vv_1 == pn_vv_2).all():
       print 'pn_vv'

    testingFeats[step,:] = ff[0]

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
    # new_im = combine_horz([thisIm,topMatchIm1,topMatchIm2,topMatchIm3,topMatchIm4,topMatchIm5,topHitIm])

    if thisLabel in sortedLabels[:100]:
        testingAccuracy[idx,topHit:] = 1

    # if ctr%10 == 0:
    #     print np.mean(testingAccuracy[:idx,:],axis=0)[0], np.mean(testingAccuracy[:idx,:]), np.mean(testingAccuracy[:idx,:],axis=0)[-1]

    # save_path = os.path.join(out_dir,str(ctr)+'_'+str(topHit)+'.jpg')
    # new_im.save(save_path)
    ctr += 1

randomSuccess = np.zeros((len(queryImsAndLabels),100,1000))
for ctr in range(1000):
    for idx in range(len(queryImsAndLabels)):
        thisLabel = queryImsAndLabels[idx][1]
        randLabels = random.sample(testingLabels[np.arange(len(testingLabels))==idx],100)
        topHit = np.where(sortedLabels==thisLabel)[0][0]
        if thisLabel in sortedLabels[:100]:
            randomSuccess[idx,topHit:,ctr] = 1

print '---Triplepalooza--'
print 'Random:'
print np.mean(np.mean(randomSuccess,axis=2),axis=0)
print 'Network: ', pretrained_net
print 'NN Test Accuracy: ',np.mean(testingAccuracy,axis=0)
print 'Example directory: ', out_dir
