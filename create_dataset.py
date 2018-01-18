import glob
import csv, os
import random

expedia_path = '/project/focus/datasets/traffickcam/resized_expedia/'
traffickcam_path = '/project/focus/datasets/traffickcam/resized_traffickcam/'
expedia_im_file = '/project/focus/datasets/traffickcam/current_expedia_ims.txt'
traffickcam_im_file = '/project/focus/datasets/traffickcam/current_traffickcam_ims.txt'

with open(traffickcam_im_file,'rU') as f:
    rd = csv.reader(f,delimiter='\t')
    traffickcam_ims = list(rd)

traffickcam_ims.pop(0)

im_list = []
flipped_im_list = []
for im_id,hotel_id,im in traffickcam_ims:
    new_path = im.replace('/mnt/EI_Code/ei_code/django_ei/submissions/',traffickcam_path)
    if os.path.exists(new_path):
        im_list.append((new_path,hotel_id))

with open(expedia_im_file,'rU') as f:
    rd = csv.reader(f,delimiter='\t')
    expedia_ims = list(rd)

expedia_ims.pop(0)

for im_id,hotel_id,im in expedia_ims:
    new_path = os.path.join(expedia_path,str(hotel_id),str(im_id)+'.jpg')
    if os.path.exists(new_path):
        im_list.append((new_path,hotel_id))

emoji_ims = glob.glob('/project/focus/datasets/emoticons/*.png')
num_classes = len(emoji_ims)
im_list = random.sample(im_list,num_classes*100)

new_dir = '/project/focus/datasets/hotels_with_emojis'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

for ix in range(len(emoji_ims)):
    im_list[2::100]
