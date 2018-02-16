import tensorflow as tf
from classfile import CombinatorialTripletSet
import os
import time
from datetime import datetime
import numpy as np
from PIL import Image
import scipy.spatial.distance
import tensorflow.contrib.slim as slim
from nets import resnet_v2
import random
# from scipy.ndimage import zoom
from skimage.transform import rescale

test_file = './val.txt'
pretrained_net = './output/ckpts/l1_loss/conv_and_feat/checkpoint-2018_02_15_1059_lr0pt0001_outputSz1000_margin0pt3_l1wgt1e-07-39999'
img_size = [256, 256]
crop_size = [227, 227]
mean_file = '/project/focus/abby/triplepalooza/models/places365/places365CNN_mean.npy'
datestr = datetime.now().strftime("%Y%m%d%H%M")

c = tf.ConfigProto()
c.gpu_options.visible_device_list="1"

batch_size = 120
num_pos_examples = batch_size/30

output_size = 1000

test_data = CombinatorialTripletSet(test_file, mean_file, img_size, crop_size, batch_size, num_pos_examples,isTraining=False)

image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
repMeanIm = np.tile(np.expand_dims(test_data.meanImage,0),[batch_size,1,1,1])
final_batch = tf.subtract(image_batch,repMeanIm)
label_batch = tf.placeholder(tf.int32, shape=(batch_size))

print("Preparing network...")
with slim.arg_scope(resnet_v2.resnet_arg_scope(is_training=True, use_batch_norm=False, updates_collections=None, batch_norm_decay=.9, fused=True)):
    _, layers = resnet_v2.resnet_v2_50(image_batch, num_classes=output_size, is_training=True, use_batch_norm=False, scope='resnet')

# feat = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("pool5:0"),3))
featLayer = 'resnet/logits'
feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))

convOut = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet/block4/unit_3/bottleneck_v2/add:0"))
weights = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet/logits/weights:0"))
biases = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet/logits/biases:0"))
gap = tf.squeeze(tf.nn.l2_normalize(tf.get_default_graph().get_tensor_by_name("resnet/pool5:0"),3))

sess = tf.Session(config=c)
saver = tf.train.Saver(max_to_keep=100)
saver.restore(sess, pretrained_net)

testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(len(test_data.files[ix]))]
numTestingIms = batch_size*(len(testingImsAndLabels)/batch_size)
testingImsAndLabels = testingImsAndLabels[:numTestingIms]
random.shuffle(testingImsAndLabels)

testingLabels = np.array([label for (im,label) in testingImsAndLabels])
indsByLabel = {}
for label in np.unique(testingLabels):
    goodInds = np.where(testingLabels==label)[0]
    if len(goodInds) > 1:
        indsByLabel[label] = goodInds

reppedLabels = np.array(indsByLabel.keys())

import matplotlib.cm
cmap =matplotlib.cm.get_cmap('jet')

def combine_horz(images):
    new_im = np.zeros((images[0].shape[0],images[0].shape[1]*len(images),images[0].shape[2]))
    for idx in range(len(images)):
        new_im[:images[idx].shape[0],images[idx].shape[1]*idx:images[idx].shape[1]*idx+images[idx].shape[1],0] = images[idx][:,:,0]
        new_im[:images[idx].shape[0],images[idx].shape[1]*idx:images[idx].shape[1]*idx+images[idx].shape[1],1] = images[idx][:,:,1]
        new_im[:images[idx].shape[0],images[idx].shape[1]*idx:images[idx].shape[1]*idx+images[idx].shape[1],2] = images[idx][:,:,2]
        if images[0].shape[2] > 3:
            new_im[:images[idx].shape[0],images[idx].shape[1]*idx:images[idx].shape[1]*idx+images[idx].shape[1],3] = images[idx][:,:,3]
    return new_im

def combine_vert(images):
    new_im = np.zeros((images[0].shape[0]*len(images),images[0].shape[1],images[0].shape[2]))
    for idx in range(len(images)):
        new_im[images[idx].shape[0]*idx:images[idx].shape[0]*idx+images[idx].shape[0],:images[idx].shape[1],:] = images[idx]
    return new_im

def getHeatMap(cam):
    hm = cmap(cam)
    hm = hm[:,:,:3]*255.
    return hm

def scaleUp(mask):
    cam = mask + np.abs(np.min(mask))
    cam = cam / np.max(cam)
    cam = rescale(cam,scale=float(crop_size[0])/float(mask.shape[0]),order=1)
    return cam

def combineImWithHeatmap(im,hm):
    bg = Image.fromarray(im.astype('uint8')).convert('RGBA')
    fg = Image.fromarray(hm.astype('uint8'))
    im_with_heatmap = np.array(Image.alpha_composite(bg,fg).getdata()).reshape((crop_size[0],crop_size[1],4))
    return im_with_heatmap

def thresholdImByHeatmap(im,hm,thresh):
    alpha_im = np.dstack((im,np.zeros((im.shape[0],im.shape[1]))))
    alpha_im[:,:,3] = 255.
    y_inds,x_inds = np.where(hm<thresh)
    alpha_im[y_inds,x_inds,3] = 50.
    # alpha_im = np.dstack((im,np.zeros((im.shape[0],im.shape[1]))))
    # alpha_im[:,:,3] = hm*255.
    return alpha_im

def getDist(feat,otherFeats):
    dist = (otherFeats - feat)**2
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    # dist = np.array([np.dot(feat,otherFeat) for otherFeat in otherFeats])
    return dist

def getDotDist(feat,otherFeats):
    dist = np.array([np.dot(feat,otherFeat) for otherFeat in otherFeats])
    return dist

testingFeats = np.empty((numTestingIms,feat.shape[1]),dtype=np.float32)
testingCV = np.empty((numTestingIms,convOut.shape[1]*convOut.shape[2],convOut.shape[3]),dtype=np.float32)
testingGAP = np.empty((numTestingIms,gap.shape[1]),dtype=np.float32)
testingIms = np.empty((numTestingIms),dtype=object)
testingLabels = np.empty((numTestingIms),dtype=np.int32)
for idx in range(0,numTestingIms,batch_size):
    print idx, '/', numTestingIms
    il = testingImsAndLabels[idx:idx+batch_size]
    ims = [i[0] for i in il]
    labels = [i[1] for i in il]
    batch = test_data.getBatchFromImageList(ims)
    testingIms[idx:idx+batch_size] = ims
    testingLabels[idx:idx+batch_size] = labels
    ff, gg, cvOut = sess.run([feat, gap,convOut], feed_dict={image_batch: batch, label_batch:labels})
    testingFeats[idx:idx+batch_size,:] = np.squeeze(ff)
    testingGAP[idx:idx+batch_size,:] = np.squeeze(gg)
    testingCV[idx:idx+batch_size,:,:] = cvOut.reshape((cvOut.shape[0],cvOut.shape[1]*cvOut.shape[2],cvOut.shape[3]))

wgts = sess.run(weights, feed_dict={image_batch: batch, label_batch:labels})

## VISUALIZE FEAT LAYER
outfolder = os.path.join('/project/focus/abby/toy_visualization/output/visualizations/',datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(outfolder)
print outfolder
ctr = 0
for label in reppedLabels:
    possible_inds = np.where(testingLabels==label)[0]
    for query_ind in possible_inds:
        query_im_path = testingIms[query_ind]
        query_feat = testingFeats[query_ind,:]
        query_label = testingLabels[query_ind]
        dists = getDotDist(query_feat,testingFeats)
        sortedDists = np.sort(dists)[::-1][1:]
        sortedInds = np.argsort(-dists)[1:]
        sortedLabels = testingLabels[sortedInds]
        sortedIms = testingIms[sortedInds]
        topHit = np.where(sortedLabels==label)[0][0]
        topHitInd = sortedInds[topHit]

        query_im = test_data.getBatchFromImageList([query_im_path])
        squeezed_query_im = np.squeeze(query_im)

        # top result
        top_feat = testingFeats[sortedInds[0],:]
        top_im = test_data.getBatchFromImageList([testingIms[sortedInds[0]]])
        squeezed_top_im = np.squeeze(top_im)
        top_label = testingLabels[sortedInds[0]]
        top_dists = (query_feat*top_feat)
        top_sortedDists = np.sort(top_dists)[::-1]
        top_bestFeats = np.argsort(top_dists)[::-1]
        top_sumTo = [np.sum(top_sortedDists[:aa]) for aa in range(1,len(top_sortedDists)+1)]

        # get heat maps
        # top match - query
        query_cvout = np.rollaxis(testingCV[query_ind,:,:],-1)
        result_cvout = np.rollaxis(testingCV[sortedInds[0],:,:],-1)

        numFeats = 3
        feat_ims = []
        for ix in range(numFeats):
            wgt = wgts[:,top_bestFeats[ix]]
            qq = np.dot(wgt,query_cvout).reshape((cvOut.shape[1],cvOut.shape[2]))
            rr = np.dot(wgt,result_cvout).reshape((cvOut.shape[1],cvOut.shape[2]))
            qq  -= np.min(qq)
            qq  /= np.max(qq)
            rr  -= np.min(rr)
            rr  /= np.max(rr)
            q_map = scaleUp(qq)
            r_map = scaleUp(rr)

            query_hms = np.zeros((crop_size[0],crop_size[1],4))
            result_hms = np.zeros((crop_size[0],crop_size[1],4))

            if top_feat[top_bestFeats[ix]] > 0 and query_feat[top_bestFeats[ix]] > 0:
                query_hms[:,:,3] = (1-q_map)*255.
                result_hms[:,:,3] = (1-r_map)*255.
            else:
                query_hms[:,:,3] = (q_map)*255.
                result_hms[:,:,3] = (r_map)*255.

            q_im = combineImWithHeatmap(squeezed_query_im.copy(),query_hms)
            r_im = combineImWithHeatmap(squeezed_top_im.copy(),result_hms)
            out_im = combine_vert([q_im,r_im])
            feat_ims.append(out_im)

            out_im = combine_horz(feat_ims)
            pil_out_im = Image.fromarray(out_im.astype('uint8'))
            b, g, r, a = pil_out_im.split()
            save_im = Image.merge("RGBA",(r,g,b,a))
            save_im_path = os.path.join(outfolder,'%d_%.3f_%.3f_%d.png'%(topHit,top_sumTo[-1],top_sumTo[numFeats],numFeats))
            b, g, r, a = pil_out_im.split()
            saveim = Image.merge("RGBA", (r, g, b, a))
            saveim.save(save_im_path)

            print save_im_path

## VISUALIZE GAP LAYER

# outfolder = os.path.join('/project/focus/abby/toy_visualization/output/visualizations/',datetime.now().strftime("%Y%m%d_%H%M%S"))
# os.makedirs(outfolder)
# print outfolder
# ctr = 0
# for label in reppedLabels:
#     possible_inds = np.where(testingLabels==label)[0]
#     for query_ind in possible_inds:
#         query_im_path = testingIms[query_ind]
#         query_feat = testingFeats[query_ind,:]
#         query_label = testingLabels[query_ind]
#         dists = getDotDist(query_feat,testingFeats)
#         sortedDists = np.sort(dists)[::-1][1:]
#         sortedInds = np.argsort(-dists)[1:]
#         sortedLabels = testingLabels[sortedInds]
#         sortedIms = testingIms[sortedInds]
#         topHit = np.where(sortedLabels==label)[0][0]
#         topHitInd = sortedInds[topHit]
#
#         query_im = test_data.getBatchFromImageList([query_im_path])
#         squeezed_query_im = np.squeeze(query_im)
#
#         # top result
#         top_feat = testingFeats[sortedInds[0],:]
#         top_im = test_data.getBatchFromImageList([testingIms[topHitInd]])
#         squeezed_top_im = np.squeeze(top_im)
#         top_label = testingLabels[topHitInd]
#         top_dists = (query_feat*top_feat)
#         top_sortedDists = np.sort(top_dists)[::-1]
#         top_bestFeats = np.argsort(top_dists)[::-1]
#         top_sumTo = [np.sum(top_sortedDists[:aa]) for aa in range(1,len(top_sortedDists)+1)]
#
#         # get heat maps
#         # top match - query
#         query_cvout = np.rollaxis(testingCV[query_ind,:,:],-1).reshape((cvOut.shape[3],cvOut.shape[1],cvOut.shape[2]))
#         result_cvout = np.rollaxis(testingCV[topHitInd,:,:],-1).reshape((cvOut.shape[3],cvOut.shape[1],cvOut.shape[2]))
#
#         # good_inds = top_bestFeats[:3]
#         # how many features does it take for us to be closer than the average dot product (which is ~.35)
#         # numFeats = np.where(np.array(top_sumTo)>.65)[0][0]
#         numFeats = 1
#         good_inds = top_bestFeats[:numFeats]
#
#         query_hms = np.zeros((query_cvout.shape[1],query_cvout.shape[2]))
#         result_hms = np.zeros((result_cvout.shape[1],result_cvout.shape[2]))
#
#         for feat_ind in good_inds:
#             query_hms += query_cvout[feat_ind,:,:]
#             result_hms += result_cvout[feat_ind,:,:]
#
#         query_hms -= np.min(query_hms)
#         query_hms /= np.max(query_hms)
#         result_hms -= np.min(result_hms)
#         result_hms /= np.max(result_hms)
#
#         q_map = scaleUp(query_hms)
#         r_map = scaleUp(result_hms)
#         q_thresh = np.mean(q_map)
#         r_thresh = np.mean(r_map)
#         q_im = thresholdImByHeatmap(squeezed_query_im.copy(),q_map,q_thresh)
#         r_im = thresholdImByHeatmap(squeezed_top_im.copy(),r_map,r_thresh)
#         out_im = combine_vert([q_im,r_im])
#         pil_out_im = Image.fromarray(out_im.astype('uint8'))
#
#         save_im_path = os.path.join(outfolder,'%d_%.3f_%.3f_%d.png'%(topHit,top_sumTo[-1],top_sumTo[numFeats],numFeats))
#         b, g, r, a = pil_out_im.split()
#         saveim = Image.merge("RGBA", (r, g, b, a))
#         saveim.save(save_im_path)
#
#         print save_im_path
