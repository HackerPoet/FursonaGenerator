import pygame
import random, sys
import numpy as np
import cv2

#User constants
device = "cpu"
model_dir = 'test24/'
is_gan = True
background_color = (210, 210, 210)
edge_color = (60, 60, 60)
slider_color = (20, 20, 20)
num_params = 80
image_scale = 3
image_padding = 10
slider_w = 15
slider_h = 100
slider_px = 5
slider_py = 10
slider_cols = 20

#Keras
print "Loading Keras..."
import os
os.environ['THEANORC'] = "./" + device + ".theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.local import LocallyConnected2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as K
K.set_image_data_format('channels_first')

print "Loading model..."
if is_gan:
	gen_model = load_model(model_dir + 'generator.h5')
	num_params = gen_model.input_shape[1]
	img_c, img_h, img_w = gen_model.output_shape[1:]
	
	if len(sys.argv) >= 2:
		enc_model = load_model(model_dir + 'encoder.h5')
	
		fname_in = sys.argv[1]
		fname_out = fname_in.split('.')
		fname_out[-2] += "_out"
		fname_out = '.'.join(fname_out)

		img = cv2.imread(fname_in)
		h = img.shape[0]
		w = img.shape[1]
		if w > h:
			offs = (w - h)/2
			img = img[:,offs:offs+h,:]
		elif h > w:
			offs = (h - w)/2
			img = img[offs:offs+w,:,:]
		img = cv2.resize(img, (img_h, img_w), interpolation = cv2.INTER_AREA)
		
		img = np.transpose(img, (2, 0, 1))
		img = img.astype(np.float32) / 255.0
		img = np.expand_dims(img, axis=0)
		
		w = enc_model.predict(img)
		img = gen_model.predict(enc_model.predict(img))[0]
		
		img = (img * 255.0).astype(np.uint8)
		img = np.transpose(img, (1, 2, 0))
		cv2.imwrite(fname_out, img)
		exit(0)
else:
	model = load_model(model_dir + 'model.h5')
	gen_func = K.function([model.get_layer('encoder').input, K.learning_phase()], [model.layers[-1].output])
	num_params = model.get_layer('encoder').input_shape[1]
	img_c, img_h, img_w = model.output_shape[1:]
	
assert(img_c == 3)

#Derived constants
slider_w = slider_w + slider_px*2
slider_h = slider_h + slider_py*2
drawing_x = image_padding
drawing_y = image_padding
drawing_w = img_w * image_scale
drawing_h = img_h * image_scale
slider_rows = (num_params - 1) / slider_cols + 1
sliders_x = drawing_x + drawing_w + image_padding
sliders_y = image_padding
sliders_w = slider_w * slider_cols
sliders_h = slider_h * slider_rows
window_w = drawing_w + sliders_w + image_padding*3
window_h = drawing_h + image_padding*2

#Global variables
prev_mouse_pos = None
mouse_pressed = False
cur_slider_ix = 0
needs_update = True
cur_params = np.zeros((num_params,), dtype=np.float32)
cur_face = np.zeros((img_c, img_h, img_w), dtype=np.uint8)
rgb_array = np.zeros((img_h, img_w, img_c), dtype=np.uint8)

print "Loading Statistics..."
means = np.load(model_dir + 'means.npy')
stds  = np.load(model_dir + 'stds.npy')
evals = np.load(model_dir + 'evals.npy')
evecs = np.load(model_dir + 'evecs.npy')

#Open a window
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((window_w, window_h))
face_surface_mini = pygame.Surface((img_w, img_h))
face_surface = screen.subsurface((drawing_x, drawing_y, drawing_w, drawing_h))
pygame.display.set_caption('Fursona Editor - By CodeParade')
font = pygame.font.SysFont("monospace", 15)

def update_mouse_click(mouse_pos):
	global cur_slider_ix
	global mouse_pressed
	x = (mouse_pos[0] - sliders_x)
	y = (mouse_pos[1] - sliders_y)

	if x >= 0 and y >= 0 and x < sliders_w and y < sliders_h:
		slider_ix_w = x / slider_w
		slider_ix_h = y / slider_h

		cur_slider_ix = slider_ix_h * slider_cols + slider_ix_w
		mouse_pressed = True

def update_mouse_move(mouse_pos):
	global needs_update
	y = (mouse_pos[1] - sliders_y)

	if y >= 0 and y < sliders_h:
		slider_row_ix = cur_slider_ix / slider_cols
		slider_val = y - slider_row_ix * slider_h

		slider_val = min(max(slider_val, slider_py), slider_h - slider_py) - slider_py
		val = (float(slider_val) / (slider_h - slider_py*2) - 0.5) * 6.0
		cur_params[cur_slider_ix] = val
		
		needs_update = True

def draw_sliders():
	for i in xrange(num_params):
		row = i / slider_cols
		col = i % slider_cols
		x = sliders_x + col * slider_w
		y = sliders_y + row * slider_h

		cx = x + slider_w / 2
		cy_1 = y + slider_py
		cy_2 = y + slider_h - slider_py
		pygame.draw.line(screen, slider_color, (cx, cy_1), (cx, cy_2))
		
		py = y + int((cur_params[i] / 6.0 + 0.5) * (slider_h - slider_py*2)) + slider_py
		pygame.draw.circle(screen, slider_color, (cx, py), slider_w/2 - slider_px)
		
		cx_1 = x + slider_px
		cx_2 = x + slider_w - slider_px
		for j in xrange(7):
			ly = y + slider_h/2 + (j-3)*(slider_h/7)
			pygame.draw.line(screen, slider_color, (cx_1, ly), (cx_2, ly))

def draw_face():
	pygame.surfarray.blit_array(face_surface_mini, np.transpose(cur_face, (2, 1, 0)))
	pygame.transform.scale(face_surface_mini, (drawing_w, drawing_h), face_surface)
	pygame.draw.rect(screen, (0,0,0), (drawing_x, drawing_y, drawing_w, drawing_h), 1)
	
#Main loop
running = True
while running:
	#Process events
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
			break
		elif event.type == pygame.MOUSEBUTTONDOWN:
			if pygame.mouse.get_pressed()[0]:
				prev_mouse_pos = pygame.mouse.get_pos()
				update_mouse_click(prev_mouse_pos)
				update_mouse_move(prev_mouse_pos)
			elif pygame.mouse.get_pressed()[2]:
				cur_params = np.zeros((num_params,), dtype=np.float32)
				needs_update = True
		elif event.type == pygame.MOUSEBUTTONUP:
			mouse_pressed = False
			prev_mouse_pos = None
		elif event.type == pygame.MOUSEMOTION and mouse_pressed:
			update_mouse_move(pygame.mouse.get_pos())
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_r:
				cur_params = np.clip(np.random.normal(0.0, 1.0, (num_params,)), -3.0, 3.0)
				needs_update = True

	#Check if we need an update
	if needs_update:
		x = means + np.dot(cur_params * evals, evecs)
		#x = means + stds * cur_params
		x = np.expand_dims(x, axis=0)
		if is_gan:
			y = gen_model.predict(x)[0]
		else:
			y = gen_func([x, 0])[0][0]
		cur_face = (y * 255.0).astype(np.uint8)
		needs_update = False
	
	#Draw to the screen
	screen.fill(background_color)
	draw_face()
	draw_sliders()
	
	#Flip the screen buffer
	pygame.display.flip()
	pygame.time.wait(10)
