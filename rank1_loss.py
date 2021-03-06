# -*- coding: utf-8 -*-
"""
# python rank1_loss.py margin batch_size output_size learning_rate whichGPU bn_decay
# python rank1_loss.py .05 100 256 .001 '1' .9
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

def main(margin,batch_size,output_size,learning_rate,whichGPU, bn_decay):
    def handler(signum, frame):
        print 'Saving checkpoint before closing'
        pretrained_net = os.path.join(ckpt_dir, 'checkpoint-'+param_str)
        saver.save(sess, pretrained_net, global_step=step)
        print 'Checkpoint-',pretrained_net+'-'+str(step), ' saved!'
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    ckpt_dir = './output/ckpts/rank1_loss'
    log_dir = './output/logs'
    train_filename = './train.txt'
    test_filename = './val.txt'
    if 'abby' in socket.gethostname().lower():
        mean_file = '/Users/abby/Documents/repos/triplepalooza/models/traffickcam/tc_mean_im.npy'
    else:
        mean_file = '/project/focus/abby/triplepalooza/models/traffickcam/tc_mean_im.npy'
    pretrained_net = '/project/focus/abby/triplepalooza/models/ilsvrc-2012/resnet_v2_50.ckpt'
    # pretrained_net = None
    finetuning = True
    img_size = [256, 256]
    crop_size = [224, 224]
    num_iters = 200000
    summary_iters = 10
    save_iters = 100

    margin = float(margin)
    batch_size = int(batch_size)
    output_size = int(output_size)
    learning_rate = float(learning_rate)
    batch_norm_decay = float(bn_decay)

    if batch_size%10 != 0:
        print 'Batch size must be divisible by 10!'
        sys.exit(0)

    ims_per_class = batch_size/10

    # Create data "batcher"
    train_data = CombinatorialTripletSet(train_filename, mean_file, img_size, crop_size, batch_size, num_pos=ims_per_class,isTraining=True)

    numClasses = len(train_data.files)
    numIms = np.sum([len(train_data.files[idx]) for idx in range(0,numClasses)])
    datestr = datetime.now().strftime("%Y_%m_%d_%H%M")
    param_str = datestr+'_lr'+str(learning_rate).replace('.','pt')+'_outputSz'+str(output_size)+'_bndecay'+str(batch_norm_decay).replace('.','pt')
    logfile_path = os.path.join(log_dir,param_str+'_train.txt')
    train_log_file = open(logfile_path,'a')
    print '------------'
    print ''
    print 'Going to train with the following parameters:'
    print '# Classes: ',numClasses
    train_log_file.write('# Classes: '+str(numClasses)+'\n')
    print '# Ims: ',numIms
    train_log_file.write('# Ims: '+str(numIms)+'\n')
    print 'Output size: ', output_size
    train_log_file.write('Output size: '+str(output_size)+'\n')
    print 'Learning rate: ',learning_rate
    train_log_file.write('Learning rate: '+str(learning_rate)+'\n')
    print 'Logging to: ',logfile_path
    train_log_file.write('Param_str: '+param_str+'\n')
    train_log_file.write('----------------\n')
    print ''
    print '------------'

    # Queuing op loads data into input tensor
    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))
    im_id_batch = tf.placeholder(tf.int32, shape=(batch_size))

    repMeanIm = np.tile(np.expand_dims(train_data.meanImage,0),[batch_size,1,1,1])
    if train_data.isOverfitting:
        final_batch = tf.subtract(image_batch,repMeanIm)
    else:
        noise = tf.random_normal(shape=[batch_size, crop_size[0], crop_size[0], 1], mean=0.0, stddev=0.0025, dtype=tf.float32)
        final_batch = tf.add(tf.subtract(image_batch,repMeanIm),noise)

    print("Preparing network...")
    # with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=False, use_batch_norm=True, updates_collections=None, batch_norm_decay=.9, fused=True)):
    #     _, layers = resnet_v2.resnet_v2_50(final_batch, use_batch_norm=True,num_classes=output_size, is_training=False)
    with slim.arg_scope(resnet_v2.resnet_arg_scope(updates_collections=None, batch_norm_decay=batch_norm_decay)):
        _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=output_size, is_training=True)

    featLayer = 'resnet_v2_50/logits'
    feat = tf.squeeze(layers[featLayer])
    expanded_a = tf.expand_dims(feat, 1)
    expanded_b = tf.expand_dims(feat, 0)
    D = tf.abs(expanded_a-expanded_b)

    # To avoid trivial solutions based on the batch ordering, we shuffle our batches.
    # This means we need to figure out what our anchor-positive indices are in the distance
    # matrix, what our "gallery" (anchor-negative) indices are, and
    # also handle the case along the diagonal (same exact image).
    rep_im_id_batch = tf.reshape(tf.tile(im_id_batch,[batch_size]),(batch_size,batch_size))
    same_im = rep_im_id_batch - tf.transpose(rep_im_id_batch)
    zero = tf.constant(0,dtype=tf.int32)
    same_im_where = tf.equal(same_im,zero)
    same_im_inds = tf.where(same_im_where)

    rep_label_batch = tf.reshape(tf.tile(label_batch,[batch_size]),(batch_size,batch_size))
    same_label = rep_label_batch - tf.transpose(rep_label_batch)
    zero = tf.constant(0,dtype=tf.int32)
    gallery_where = tf.not_equal(same_label,zero)
    pos_where = tf.not_equal(tf.equal(same_label,zero),same_im_where) # we don't care about + pairs that are the exact same image
    gallery_inds = tf.where(gallery_where)
    pos_inds = tf.where(pos_where)

    # Get the indices of anchor-positive pairs, and grab those from the distance matrix
    num_pos = batch_size*(ims_per_class-1)
    _, sorted_pos_inds = tf.nn.top_k(-1*tf.squeeze(tf.slice(pos_inds,[0,1],[-1,1])), k=num_pos)
    posDists = tf.gather_nd(D,pos_inds)
    posDists2 = tf.gather(posDists,sorted_pos_inds)
    posDists3 = tf.reshape(posDists2,(batch_size,ims_per_class-1,output_size))

    # Get the incides of the "gallery" of anchor-negative pairs, and grab those from
    # the distance matrix and then for every anchor feature component,
    # find the closest feature components from the gallery
    num_gallery = batch_size*(ims_per_class-1)*ims_per_class
    _, sorted_gallery_inds = tf.nn.top_k(-1*tf.squeeze(tf.slice(gallery_inds,[0,1],[-1,1])), k=num_gallery)
    galleryDists = tf.gather_nd(D,gallery_inds)
    galleryDists2 = tf.gather(galleryDists,sorted_gallery_inds)
    galleryDists3 = tf.reshape(galleryDists2,(batch_size,ims_per_class*(ims_per_class-1),output_size))
    min_galleryDist = tf.tile(tf.expand_dims(tf.reduce_min(galleryDists3,axis=1),1),(1,ims_per_class-1,1))

    # Sum up the distances where the feature components are inverted:
    # posDists3 > min(anchor-gallery dists)
    inversions = tf.reshape(tf.reduce_sum(tf.maximum(posDists3 - min_galleryDist + margin, 0.),axis=2),[batch_size*(ims_per_class-1)])

    # Loss is the mean of the inversions over the whole batch
    loss = tf.reduce_mean(inversions)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.MomentumOptimizer(learning_rate,0.95)
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
        variables_to_restore = []
        # if we're fine-tuning, we need to make sure not to include the logits layer in the variables that we restore
        for var in slim.get_model_variables():
            excluded = False
            # exclude all momentum terms since we're using adam optimizer
            if 'momentum' in var.op.name.lower():
                excluded = True
            # if we're fine tuning, exclude the feature vector that we want to fine tune
            if finetuning and var.op.name.startswith(featLayer):
                excluded = True
            if not excluded:
                variables_to_restore.append(var)

        restore_fn = slim.assign_from_checkpoint_fn(pretrained_net,variables_to_restore)
        restore_fn(sess)

    print("Start training...")
    ctr  = 0
    for step in range(num_iters):
        start_time = time.time()
        batch, labels, ims = train_data.getBatch()
        _, loss_val = sess.run([train_op, loss], feed_dict={image_batch: batch, label_batch: labels, im_id_batch: range(len(ims))})
        end_time = time.time()
        duration = end_time-start_time
        out_str = 'Step %d: loss = %.15f -- (%.3f sec)' % (step, loss_val,duration)
        print(out_str)
        if step % summary_iters == 0:
            # print(out_str)
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
        print 'Expected input parameters:margin, batch_size, output_size, learning_rate, whichGPU, bn_decay'
    margin = args[1]
    batch_size = args[2]
    output_size = args[3]
    learning_rate = args[4]
    whichGPU = args[5]
    bn_decay = args[6]
    main(margin,batch_size,output_size,learning_rate,whichGPU,bn_decay)
