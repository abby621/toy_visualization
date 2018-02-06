# -*- coding: utf-8 -*-
"""
# python mars_triplepalooza.py margin output_size learning_rate is_overfitting whichGPU
# python training.py .3 60 256 .00005 False '2'
"""

import tensorflow as tf
from classfile import CombinatorialTripletSet
import os.path
import time
from datetime import datetime
import numpy as np
import socket
import signal
import sys
import resnet_model

def main(margin,batch_size,output_size,learning_rate,is_overfitting,whichGPU):
    def handler(signum, frame):
        print 'Exiting at step: ', step
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/ckpts'
    log_dir = './output/logs'
    train_filename = './train.txt'
    if 'abby' in socket.gethostname().lower():
        mean_file = '/Users/abby/Documents/repos/triplepalooza/models/traffickcam/tc_mean_im.npy'
    else:
        mean_file = '/project/focus/abby/triplepalooza/models/traffickcam/tc_mean_im.npy'

    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 1000
    save_iters = 1000

    if is_overfitting.lower()=='true':
        is_overfitting = True
    else:
        is_overfitting = False

    margin = float(margin)
    batch_size = int(batch_size)
    output_size = int(output_size)
    learning_rate = float(learning_rate)

    if batch_size%20 != 0:
        print 'Batch size must be divisible by 20!'
        sys.exit(0)

    num_pos_examples = batch_size/20

    # Create data "batcher"
    train_data = CombinatorialTripletSet(train_filename, mean_file, img_size, crop_size, batch_size, num_pos_examples, isTraining=True, isOverfitting=is_overfitting)
    numClasses = len(train_data.files)
    numIms = np.sum([len(train_data.files[idx]) for idx in range(0,numClasses)])

    hyperparams = resnet_model.HParams(batch_size=batch_size,
                         num_classes=output_size,
                         num_residual_units=4,
                         use_bottleneck=True,
                         weight_decay_rate=0.0002,
                         relu_leakiness=0.1)

    datestr = datetime.now().strftime("%Y_%m_%d_%H%M")
    param_str = datestr+'_lr'+str(learning_rate).replace('.','pt')+'_outputSz'+str(output_size)+'_margin'+str(margin).replace('.','pt')

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
    if train_data.isOverfitting:
        final_batch = tf.subtract(image_batch,repMeanIm)
    else:
        noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 1], mean=0.0, stddev=0.0025, dtype=tf.float32)
        final_batch = tf.add(tf.subtract(image_batch,repMeanIm),noise)

    model = resnet_model.ResNet(hyperparams, final_batch, label_batch, 'train')
    model.build_graph()

    feat = model.logits
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
    loss = tf.reduce_mean(tf.maximum(0.,tf.multiply(mask,margin + posDistsRep - allDists)))
    regularizer = tf.nn.l2_loss(feat)
    loss = tf.reduce_mean(loss + 0.0001*regularizer)

    train_ops = tf.group(*[tf.train.AdamOptimizer(learning_rate).minimize(loss)] + model._extra_train_ops)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(max_to_keep=500)

    # tf will consume any GPU it finds on the system. Following lines restrict it to specific gpus
    c = tf.ConfigProto()
    if not 'abby' in socket.gethostname().lower():
        c.gpu_options.visible_device_list=whichGPU

    summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=ckpt_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': loss},
      every_n_iter=100)

     print("Starting session...")
     with tf.train.MonitoredTrainingSession(
            checkpoint_dir=out_dir,
            hooks=[logging_hook],
            chief_only_hooks=[summary_hook],
            save_summaries_steps=0,
            config=c) as sess:
         step = 0
         while step < num_iters:
             start_time = time.time()
             batch, labels, ims = train_data.getBatch()
             _, step, loss_val = sess.run([train_ops, model.global_step, loss], feed_dict={image_batch: batch, label_batch: labels})
            end_time = time.time()
            duration = end_time-start_time
            out_str = 'Step %d: loss = %.6f (%.3f sec)' % (step, loss_val, duration)
            if step%10 == 0:
                print(out_str)
