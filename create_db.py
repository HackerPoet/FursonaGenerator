import os, random, sys
import numpy as np
import cv2
from scipy import ndimage

IMAGE_DIR = "good_pics"
DATA_DIR = "data"
IMAGE_SIZE = 128

color_imgs = []
num_imgs = 0

if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)

print "Loading Images..."
for file in os.listdir(IMAGE_DIR):
	path = IMAGE_DIR + "/" + file
	
	#Only attempt to load standard image formats
	path_split = path.split('.')
	if len(path_split) < 2: continue
	if path_split[-1] not in ['bmp', 'gif', 'png', 'jpg', 'jpeg']:
		continue
	
	#Make sure image is valid and not corrupt
	img = ndimage.imread(path)
	if img is None:
		assert(False)
	if len(img.shape) != 3 or img.shape[2] < 3:
		continue
	if img.shape[2] > 3:
		img = img[:,:,:3]
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	
	#Crop center square of image
	h = img.shape[0]
	w = img.shape[1]
	if w > h:
		offs = (w - h)/2
		img = img[:,offs:offs+h,:]
	elif h > w:
		offs = (h - w)/2
		img = img[offs:offs+w,:,:]
	
	#Scale all images to a uniform size
	img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_AREA)
	
	#Save some samples
	if num_imgs < 10:
		cv2.imwrite("color" + str(num_imgs) + ".png", img)
	
	#Add to running list
	color_imgs.append(np.transpose(img, (2,0,1)))

	#Show progress
	num_imgs += 1
	if num_imgs % 10 == 0:
		sys.stdout.write('\r')
		sys.stdout.write(str(num_imgs))
		sys.stdout.flush()
print "\nLoaded " + str(num_imgs) + " images."

print "Saving..."
color_imgs = np.stack(color_imgs, axis=0)
np.save(DATA_DIR + '/color' + str(IMAGE_SIZE) + '.npy', color_imgs)

print "Done"