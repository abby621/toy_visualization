# -*- coding: utf-8 -*-
"""
# python training.py margin output_size learning_rate is_overfitting l1_weight
# python training.py .3 120 1000 .0001 False '2' .00001
"""

import tensorflow as tf
from classfile import CombinatorialTripletSet
import os.path
import time
from datetime import datetime
import numpy as np
from PIL import Image
from tensorflow.python.ops.image_ops_impl import *
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import tensorflow.contrib.slim as slim
from nets import resnet_v2
import socket
import signal
import sys

def main(margin,batch_size,output_size,learning_rate,is_overfitting,whichGPU,l1_weight):
    def handler(signum, frame):
        print 'Saving checkpoint before closing'
        pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
        saver.save(sess, pretrained_net, global_step=step)
        print 'Checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/ckpts/l1_loss/conv_and_feat'
    log_dir = './output/logs'
    train_filename = './train.txt'
    if 'abby' in socket.gethostname().lower():
        mean_file = '/Users/abby/Documents/repos/triplepalooza/models/traffickcam/tc_mean_im.npy'
    else:
        mean_file = '/project/focus/abby/triplepalooza/models/traffickcam/tc_mean_im.npy'
    # pretrained_net = os.path.join(ckpt_dir,'checkpoint-2018_02_12_1052_lr0pt0001_outputSz1000_margin0pt3-3582')
    pretrained_net = None
    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 200000
    summary_iters = 10
    save_iters = 1000

    # is_training = True
    if is_overfitting.lower()=='true':
        is_overfitting = True
    else:
        is_overfitting = False

    margin = float(margin)
    batch_size = int(batch_size)
    output_size = int(output_size)
    learning_rate = float(learning_rate)
    l1_weight = float(l1_weight)

    if batch_size%30 != 0:
        print 'Batch size must be divisible by 30!'
        sys.exit(0)

    num_pos_examples = batch_size/30

    # Create data "batcher"
    train_data = CombinatorialTripletSet(train_filename, mean_file, img_size, crop_size, batch_size, num_pos_examples, isTraining=True, isOverfitting=is_overfitting)
    numClasses = len(train_data.files)
    numIms = np.sum([len(train_data.files[idx]) for idx in range(0,numClasses)])
    datestr = datetime.now().strftime("%Y_%m_%d_%H%M")
    param_str = datestr+'_lr'+str(learning_rate).replace('.','pt')+'_outputSz'+str(output_size)+'_margin'+str(margin).replace('.','pt')+'_l1wgt'+str(l1_weight).replace('.','pt')
    logfile_path = os.path.join(log_dir,param_str+'_train.txt')
    train_log_file = open(logfile_path,'a')
    print '------------'
    print ''
    print 'Going to train with the following parameters:'
    print '# Classes: ',numClasses
    train_log_file.write('# Classes: '+str(numClasses)+'\n')
    print '# Ims: ',numIms
    train_log_file.write('# Ims: '+str(numIms)+'\n')
    print 'Margin: ',margin
    train_log_file.write('Margin: '+str(margin)+'\n')
    print 'Output size: ', output_size
    train_log_file.write('Output size: '+str(output_size)+'\n')
    print 'Learning rate: ',learning_rate
    train_log_file.write('Learning rate: '+str(learning_rate)+'\n')
    print 'Overfitting?: ',is_overfitting
    train_log_file.write('Is overfitting?'+str(is_overfitting)+'\n')
    print 'Logging to: ',logfile_path
    train_log_file.write('Param_str: '+param_str+'\n')
    train_log_file.write('----------------\n')
    print ''
    print '------------'

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
    if train_data.isOverfitting:
        final_batch = tf.subtract(image_batch,repMeanIm)
    else:
        noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 1], mean=0.0, stddev=0.0025, dtype=tf.float32)
        final_batch = tf.add(tf.subtract(image_batch,repMeanIm),noise)

    print("Preparing network...")
    with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=True, use_batch_norm=False, updates_collections=None, batch_norm_decay=.7, fused=True)):
        _, layers = resnet_v2.resnet_v2_50(final_batch, use_batch_norm=False,num_classes=output_size, is_training=True, scope='resnet')

    featLayer = 'resnet/logits'
    feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
    convOut = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet/block4/unit_3/bottleneck_v2/add:0"))
    # feat = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("pool5:0"),3))
    # varvar = tf.get_default_graph().get_tensor_by_name("resnet/block1/unit_1/bottleneck_v2/conv1/BatchNorm/moving_mean:0")

    expanded_a = tf.expand_dims(feat, 1)
    expanded_b = tf.expand_dims(feat, 0)
    D = tf.reduce_sum(tf.squared_difference(expanded_a, expanded_b), 2)

    posIdx = np.floor(np.arange(0,batch_size)/num_pos_examples).astype('int')
    posIdx10 = num_pos_examples*posIdx
    posImInds = np.tile(posIdx10,(num_pos_examples,1)).transpose()+np.tile(np.arange(0,num_pos_examples),(batch_size,1))
    anchorInds = np.tile(np.arange(0,batch_size),(num_pos_examples,1)).transpose()

    posImInds_flat = posImInds.ravel()
    anchorInds_flat = anchorInds.ravel()

    posPairInds = zip(posImInds_flat,anchorInds_flat)
    posDists = tf.reshape(tf.gather_nd(D,posPairInds),(batch_size,num_pos_examples))

    shiftPosDists = tf.reshape(posDists,(1,batch_size,num_pos_examples))
    posDistsRep = tf.tile(shiftPosDists,(batch_size,1,1))

    allDists = tf.tile(tf.expand_dims(D,2),(1,1,num_pos_examples))

    ra, rb, rc = np.meshgrid(np.arange(0,batch_size),np.arange(0,batch_size),np.arange(0,num_pos_examples))

    bad_negatives = np.floor((ra)/num_pos_examples) == np.floor((rb)/num_pos_examples)
    bad_positives = np.mod(rb,num_pos_examples) == np.mod(rc,num_pos_examples)

    mask = ((1-bad_negatives)*(1-bad_positives)).astype('float32')

    # loss = tf.reduce_sum(tf.maximum(0.,tf.multiply(mask,margin + posDistsRep - allDists)))/batch_size
    base_loss = tf.reduce_mean(tf.maximum(0.,tf.multiply(mask,margin + posDistsRep - allDists)))
    l1_loss = tf.multiply(l1_weight, tf.reduce_sum(tf.abs(feat)))
    l1_loss = l1_loss + tf.multiply(l1_weight/10000, tf.reduce_sum(tf.reduce_mean(tf.reduce_sum(tf.abs(tf.reshape(convOut,[convOut.shape[0],convOut.shape[1]*convOut.shape[2],convOut.shape[3]])),axis=1),axis=1)))
    loss = base_loss + l1_loss

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=500)

    # tf will consume any GPU it finds on the system. Following lines restrict it to specific gpus
    c = tf.ConfigProto()
    if not 'abby' in socket.gethostname().lower():
        c.gpu_options.visible_device_list=whichGPU

    print("Starting session...")
    sess = tf.Session(config=c)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    if pretrained_net:
        saver.restore(sess, pretrained_net)

    print("Start training...")
    ctr  = 0
    for step in range(num_iters):
        start_time = time.time()
        batch, labels, ims = train_data.getBatch()
        _, loss_val, bl, l1 = sess.run([train_op, loss, base_loss, l1_loss], feed_dict={image_batch: batch, label_batch: labels})
        end_time = time.time()
        duration = end_time-start_time
        out_str = 'Step %d: loss = %.6f (%.6f from loss, %.6f from l1) -- (%.3f sec)' % (step, loss_val, bl, l1, duration)
        # print(out_str)
        if step % summary_iters == 0 or is_overfitting:
            print(out_str)
            train_log_file.write(out_str+'\n')
        # Update the events file.
        # summary_str = sess.run(summary_op)
        # writer.add_summary(summary_str, step)
        # writer.flush()
        #
        # Save a checkpoint
        if (step + 1) % save_iters == 0:
            print('Saving checkpoint at iteration: %d' % (step))
            pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
            saver.save(sess, pretrained_net, global_step=step)
            print 'checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        if (step + 1) == num_iters:
            print('Saving final')
            pretrained_net = os.path.join(ckpt_dir, 'final-'+param_str)
            saver.save(sess, pretrained_net, global_step=step)
            print 'final-',pretrained_net+'-'+str(step), ' saved!'

    sess.close()
    train_log_file.close()

      #  coord.request_stop()
       # coord.join(threads)

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 5:
        print 'Expected four input parameters: margin, output_size, learning_rate, is_overfitting, whichGPU, l1_weight'
    margin = args[1]
    batch_size = args[2]
    output_size = args[3]
    learning_rate = args[4]
    is_overfitting = args[5]
    whichGPU = args[6]
    l1_weight = args[7]
    main(margin,batch_size,output_size,learning_rate,is_overfitting,whichGPU,l1_weight)
