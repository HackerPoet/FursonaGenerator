import os, sys, random
import numpy as np
import cv2
from matplotlib import pyplot as plt

WRITE_DIR = "test26/"
DATA_DIR = 'data/'
CONTINUE_TRAIN = False
TRAIN_EDGES = False
USE_EMBEDDING = False
USE_MIRROR = False
USE_BG_SWAP = False
USE_ROLLS = False
PARAM_SIZE = 80
NUM_EPOCHS = 50
BATCH_SIZE = 32
RATIO_G = 1
LR_D = 0.0008
LR_G = 0.0008
BETA_1 = 0.8
EPSILON = 1e-4
ENC_WEIGHT = 400.0
BN_M = 0.9
DO_RATE = 0.5
DO_RATE_G = 0.3
NOISE_SIGMA = 0.15
NUM_RAND_FACES = 10
PREV_V = None

def save_config(fname):
	with open(fname, 'w') as fout:
		fout.write('PARAM_SIZE  ' + str(PARAM_SIZE ) + '\n') 
		fout.write('NUM_EPOCHS  ' + str(NUM_EPOCHS ) + '\n')
		fout.write('BATCH_SIZE  ' + str(BATCH_SIZE ) + '\n') 
		fout.write('RATIO_G     ' + str(RATIO_G    ) + '\n') 
		fout.write('LR_D        ' + str(LR_D       ) + '\n') 
		fout.write('LR_G        ' + str(LR_G       ) + '\n') 
		fout.write('BETA_1      ' + str(BETA_1     ) + '\n') 
		fout.write('EPSILON     ' + str(EPSILON    ) + '\n') 
		fout.write('ENC_WEIGHT  ' + str(ENC_WEIGHT ) + '\n') 
		fout.write('BN_M        ' + str(BN_M       ) + '\n') 
		fout.write('DO_RATE     ' + str(DO_RATE    ) + '\n') 
		fout.write('DO_RATE_G   ' + str(DO_RATE_G  ) + '\n') 
		fout.write('NOISE_SIGMA ' + str(NOISE_SIGMA) + '\n') 

if not os.path.exists(WRITE_DIR):
	os.makedirs(WRITE_DIR)
save_config(WRITE_DIR + 'config.txt')

def plotScores(scores, fname, on_top=True):
	plt.clf()
	ax = plt.gca()
	ax.yaxis.tick_right()
	ax.yaxis.set_ticks_position('both')
	ax.yaxis.grid(True)
	for s in scores:
		plt.plot(s)
	plt.xlabel('Epoch')
	loc = ('upper right' if on_top else 'lower right')
	plt.legend(['Dis', 'Gen', 'Enc'], loc=loc)
	plt.draw()
	plt.savefig(fname)

def shift_keep(imgs, sx, sy):
	assert(len(imgs.shape) == 4)

	#Shift X
	result_x = np.empty_like(imgs)
	if sx > 0:
		result_x[:,:,:,:sx] = imgs[:,:,:,:1]
		result_x[:,:,:,sx:] = imgs[:,:,:,:-sx]
	elif sx < 0:
		result_x[:,:,:,sx:] = imgs[:,:,:,-1:]
		result_x[:,:,:,:sx] = imgs[:,:,:,-sx:]
	else:
		result_x = imgs

	#Shift Y
	result_y = np.empty_like(result_x)
	if sy > 0:
		result_y[:,:,:sy] = result_x[:,:,:1]
		result_y[:,:,sy:] = result_x[:,:,:-sy]
	elif sy < 0:
		result_y[:,:,sy:] = result_x[:,:,-1:]
		result_y[:,:,:sy] = result_x[:,:,-sy:]
	else:
		result_y = result_x
	
	return result_y

#Load data set
print "Loading Image Data..."
if TRAIN_EDGES:
	y_train = np.load(DATA_DIR + 'gray128.npy').astype(np.float32) / 255.0
else:
	y_train = np.load(DATA_DIR + 'color128.npy').astype(np.float32) / 255.0
if USE_MIRROR:
	y_train = np.concatenate((y_train, np.flip(y_train, axis=3)), axis=0)
if USE_BG_SWAP:
	y_train = np.concatenate((y_train, y_train[:,[1,0,2]]), axis=0)
y_orig = y_train
print "Loaded " + str(y_train.shape[0]) + " Samples."

num_samples = y_train.shape[0]
i_train = np.arange(num_samples)
y_shape = y_train.shape
if USE_EMBEDDING:
	x_train = np.expand_dims(np.arange(num_samples), axis=1)
else:
	x_train = y_train
x_shape = x_train.shape

###################################
#  Create Model
###################################
print "Loading Keras..."
import os, math
os.environ['THEANORC'] = "./gpu.theanorc"
os.environ['KERAS_BACKEND'] = "theano"
import theano
print "Theano Version: " + theano.__version__
import keras
print "Keras Version: " + keras.__version__
from keras.initializers import RandomNormal
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, TimeDistributed, LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.utils import plot_model
from keras import backend as K
K.set_image_data_format('channels_first')

###################################
#  Create Model
###################################
if CONTINUE_TRAIN:
	print "Loading Discriminator..."
	discriminator = load_model(WRITE_DIR + 'discriminator.h5')
	print "Loading Generator..."
	generator = load_model(WRITE_DIR + 'generator.h5')
	print "Loading Encoder..."
	encoder = load_model(WRITE_DIR + 'encoder.h5')
	print "Loading Vectors..."
	PREV_V = np.load(WRITE_DIR + 'evecs.npy')
	z_test = np.load(WRITE_DIR + 'rand.npy')
else:
	print "Building Discriminator..."
	input_shape = y_shape[1:]
	print (None,) + input_shape
	discriminator = Sequential()
	discriminator.add(GaussianNoise(NOISE_SIGMA, input_shape=input_shape))

	discriminator.add(Conv2D(40, (5,5), padding='same'))
	discriminator.add(MaxPooling2D(2))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(BatchNormalization(momentum=BN_M, axis=1))
	if DO_RATE > 0:
		discriminator.add(Dropout(DO_RATE))
	print discriminator.output_shape
	
	discriminator.add(Conv2D(60, (5,5), padding='same'))
	discriminator.add(MaxPooling2D(2))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(BatchNormalization(momentum=BN_M, axis=1))
	if DO_RATE > 0:
		discriminator.add(Dropout(DO_RATE))
	print discriminator.output_shape
	
	discriminator.add(Conv2D(120, (5,5), padding='same'))
	discriminator.add(MaxPooling2D(8))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(BatchNormalization(momentum=BN_M, axis=1))
	if DO_RATE > 0:
		discriminator.add(Dropout(DO_RATE))
	print discriminator.output_shape

	discriminator.add(Flatten(data_format = 'channels_last'))
	print discriminator.output_shape
	
	discriminator.add(Dense(1, activation='sigmoid'))
	print discriminator.output_shape

	print "Building Generator..."
	generator = Sequential()
	input_shape = (PARAM_SIZE,)
	print (None,) + input_shape

	generator.add(Dense(360*4*4, input_shape=input_shape))
	generator.add(Reshape((360,4,4)))

	generator.add(LeakyReLU(0.2))
	print generator.output_shape
	if DO_RATE_G > 0: generator.add(Dropout(DO_RATE_G))
	generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(Conv2DTranspose(280, (5,5), strides=(2,2), padding='same'))
	generator.add(LeakyReLU(0.2))
	if DO_RATE_G > 0: generator.add(Dropout(DO_RATE_G))
	generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(Conv2DTranspose(200, (5,5), strides=(2,2), padding='same'))
	generator.add(LeakyReLU(0.2))
	if DO_RATE_G > 0: generator.add(Dropout(DO_RATE_G))
	generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(Conv2DTranspose(160, (5,5), strides=(2,2), padding='same'))
	generator.add(LeakyReLU(0.2))
	if DO_RATE_G > 0: generator.add(Dropout(DO_RATE_G))
	generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape

	generator.add(Conv2DTranspose(80, (5,5), strides=(2,2), padding='same'))
	generator.add(LeakyReLU(0.2))
	if DO_RATE_G > 0: generator.add(Dropout(DO_RATE_G))
	generator.add(BatchNormalization(momentum=BN_M, axis=1))
	print generator.output_shape
	
	generator.add(Conv2DTranspose(y_train.shape[1], (5,5), strides=(2,2), padding='same', activation='sigmoid'))
	print generator.output_shape

	print "Building Encoder..."
	encoder = Sequential()
	if USE_EMBEDDING:
		print (None, num_samples)
		encoder.add(Embedding(num_samples, PARAM_SIZE, input_length=1, embeddings_initializer=RandomNormal(stddev=1e-4)))
		encoder.add(Flatten(data_format = 'channels_last'))
		print encoder.output_shape
	else:
		input_shape = y_shape[1:]
		print (None,) + input_shape
		encoder = Sequential()

		encoder.add(Conv2D(120, (5,5), strides=(2,2), padding='same', input_shape=input_shape))
		encoder.add(LeakyReLU(0.2))
		encoder.add(BatchNormalization(momentum=BN_M, axis=1))
		print encoder.output_shape

		encoder.add(Conv2D(200, (5,5), strides=(2,2), padding='same'))
		encoder.add(LeakyReLU(0.2))
		encoder.add(BatchNormalization(momentum=BN_M, axis=1))
		print encoder.output_shape
		
		encoder.add(Conv2D(260, (5,5), strides=(2,2), padding='same'))
		encoder.add(LeakyReLU(0.2))
		encoder.add(BatchNormalization(momentum=BN_M, axis=1))
		print encoder.output_shape

		encoder.add(Conv2D(300, (5,5), strides=(2,2), padding='same'))
		encoder.add(LeakyReLU(0.2))
		encoder.add(BatchNormalization(momentum=BN_M, axis=1))
		print encoder.output_shape
		
		encoder.add(Conv2D(360, (5,5), strides=(2,2), padding='same'))
		encoder.add(LeakyReLU(0.2))
		encoder.add(BatchNormalization(momentum=BN_M, axis=1))
		print encoder.output_shape

		encoder.add(Flatten(data_format = 'channels_last'))
		print encoder.output_shape
		
		encoder.add(Dense(PARAM_SIZE))
		encoder.add(BatchNormalization(momentum=BN_M))
		print encoder.output_shape
	
print "Building GANN..."
d_optimizer = Adam(lr=LR_D, beta_1=BETA_1, epsilon=EPSILON)
g_optimizer = Adam(lr=LR_G, beta_1=BETA_1, epsilon=EPSILON)

discriminator.trainable = True
generator.trainable = False
encoder.trainable = False
d_in_real = Input(shape=y_shape[1:])
d_in_fake = Input(shape=x_shape[1:])
d_fake = generator(encoder(d_in_fake))
d_out_real = discriminator(d_in_real)
d_out_real = Activation('linear', name='d_out_real')(d_out_real)
d_out_fake = discriminator(d_fake)
d_out_fake = Activation('linear', name='d_out_fake')(d_out_fake)
dis_model = Model(inputs=[d_in_real, d_in_fake], outputs=[d_out_real, d_out_fake])
dis_model.compile(
	optimizer=d_optimizer,
	loss={'d_out_real':'binary_crossentropy', 'd_out_fake':'binary_crossentropy'},
	loss_weights={'d_out_real':1.0, 'd_out_fake':1.0})

discriminator.trainable = False
generator.trainable = True
encoder.trainable = True
g_in = Input(shape=x_shape[1:])
g_enc = encoder(g_in)
g_out_img = generator(g_enc)
g_out_img = Activation('linear', name='g_out_img')(g_out_img)
g_out_dis = discriminator(g_out_img)
g_out_dis = Activation('linear', name='g_out_dis')(g_out_dis)
gen_dis_model = Model(inputs=[g_in], outputs=[g_out_img, g_out_dis])
gen_dis_model.compile(
	optimizer=g_optimizer,
	loss={'g_out_img':'mse', 'g_out_dis':'binary_crossentropy'},
	loss_weights={'g_out_img':ENC_WEIGHT, 'g_out_dis':1.0})
	
plot_model(generator, to_file=WRITE_DIR + 'generator.png', show_shapes=True)
plot_model(discriminator, to_file=WRITE_DIR + 'discriminator.png', show_shapes=True)
plot_model(encoder, to_file=WRITE_DIR + 'encoder.png', show_shapes=True)

###################################
#  Encoder Decoder
###################################
def save_image(fname, x):
	img = (x * 255.0).astype(np.uint8)
	img = np.transpose(img, (1, 2, 0))
	cv2.imwrite(fname, img)

def make_rand_faces(write_dir, x_vecs):
	y_faces = generator.predict(x_vecs)
	for i in xrange(y_faces.shape[0]):
		save_image(write_dir + 'rand' + str(i) + '.png', y_faces[i])

def make_rand_faces_normalized(write_dir, rand_vecs):
	global PREV_V
	x_enc = encoder.predict(x_train, batch_size=BATCH_SIZE)
	
	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	u, s, v = np.linalg.svd(x_cov)
	e = np.sqrt(s)
	
	# This step is not necessary, but it makes random the generated test
	# samples consistent between epochs so you can see the evolution of
	# the training better.
	#
	# Like a square root, each prinicpal component has 2 solutions that
	# represent opposing vector directions.  For each component, just
	# choose the direction that was closest to the last epoch.
	if PREV_V is not None:
		d = np.sum(PREV_V * v, axis=1)
		d = np.where(d > 0.0, 1.0, -1.0)
		v = v * np.expand_dims(d, axis=1)
	PREV_V = v

	print "Evals: ", e[:6]
	
	np.save(write_dir + 'means.npy', x_mean)
	np.save(write_dir + 'stds.npy', x_stds)
	np.save(write_dir + 'evals.npy', e)
	np.save(write_dir + 'evecs.npy', v)

	x_vecs = x_mean + np.dot(rand_vecs * e, v)
	make_rand_faces(write_dir, x_vecs)
	
	plt.clf()
	e[::-1].sort()
	plt.title('evals')
	plt.bar(np.arange(e.shape[0]), e, align='center')
	plt.draw()
	plt.savefig(write_dir + '_evals.png')

	plt.clf()
	plt.title('means')
	plt.bar(np.arange(e.shape[0]), x_mean, align='center')
	plt.draw()
	plt.savefig(write_dir + '_means.png')
	
	plt.clf()
	plt.title('stds')
	plt.bar(np.arange(e.shape[0]), x_stds, align='center')
	plt.draw()
	plt.savefig(write_dir + '_stds.png')

def save_models():
	discriminator.save(WRITE_DIR + 'discriminator.h5')
	generator.save(WRITE_DIR + 'generator.h5')
	encoder.save(WRITE_DIR + 'encoder.h5')
	print "Saved"
	
###################################
#  Train
###################################
print "Training..."
generator_loss = []
discriminator_loss = []
encoder_loss = []

z_test = np.random.normal(0.0, 1.0, (NUM_RAND_FACES, PARAM_SIZE))
np.save(WRITE_DIR + 'rand.npy', z_test)

for iters in xrange(NUM_EPOCHS):
	if USE_ROLLS:
		y_rolls = []
		for i in xrange(10):
			sx = random.randint(-8,8)
			sy = random.randint(-8,8)
			y_rolls.append(shift_keep(y_orig, sx, sy))
		y_train = np.concatenate(y_rolls, axis=0)
		x_train = y_train
		num_samples = y_train.shape[0]
		i_train = np.arange(num_samples)

	loss_d = 0.0
	loss_g = 0.0
	loss_e = 0.0
	num_d = 0
	num_g = 0
	num_e = 0

	np.random.shuffle(i_train)
	for i in xrange(0, num_samples/BATCH_SIZE):
		if i % RATIO_G == 0:
			#Make samples
			j = i / RATIO_G
			i_batch1 = i_train[j*BATCH_SIZE:(j + 1)*BATCH_SIZE]
			x_batch1 = x_train[i_batch1]
			y_batch1 = y_train[i_batch1]
			
			ones = np.ones((BATCH_SIZE,), dtype=np.float32)
			zeros = np.zeros((BATCH_SIZE,), dtype=np.float32)

			losses = dis_model.train_on_batch([y_batch1, x_batch1], [ones, zeros])
			names = dis_model.metrics_names
			loss_d += losses[names.index('d_out_real_loss')]
			loss_d += losses[names.index('d_out_fake_loss')]
			num_d += 2

		i_batch2 = i_train[i*BATCH_SIZE:(i + 1)*BATCH_SIZE]
		x_batch2 = x_train[i_batch2]
		y_batch2 = y_train[i_batch2]
		
		losses = gen_dis_model.train_on_batch([x_batch2], [y_batch2, ones])
		names = gen_dis_model.metrics_names
		loss_e += losses[names.index('g_out_img_loss')]
		loss_g += losses[names.index('g_out_dis_loss')]
		num_e += 1
		num_g += 1

		progress = (i * 100)*BATCH_SIZE / num_samples
		sys.stdout.write(
			str(progress) + "%" +
			"  D:" + str(loss_d / num_d) +
			"  G:" + str(loss_g / num_g) +
			"  E:" + str(loss_e / num_e) + "        ")
		sys.stdout.write('\r')
		sys.stdout.flush()
	sys.stdout.write('\n')
	
	discriminator_loss.append(loss_d / num_d)
	generator_loss.append(loss_g / num_g)
	encoder_loss.append(loss_e * 10.0 / num_e)

	try:
		plotScores([discriminator_loss, generator_loss, encoder_loss], WRITE_DIR + 'Scores.png')
		save_models()

		make_rand_faces_normalized(WRITE_DIR, z_test)
		i_test = i_train[-NUM_RAND_FACES:]
		x_test = x_train[i_test]
		y_test = y_train[i_test]
		y_pred = generator.predict(encoder.predict(x_test))
		for i in xrange(y_pred.shape[0]):
			save_image(WRITE_DIR + "gt" + str(i) + ".png", y_test[i])
			save_image(WRITE_DIR + "pred" + str(i) + ".png", y_pred[i])
	except IOError:
		pass
		
print "Done"
