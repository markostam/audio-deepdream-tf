from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import tensorflow as tf
import librosa

layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139 # picking some feature channel to visualize
path_to_audio = './audio/helix_drum_track.wav'
iterations = 10
octaves = 8

def deepdream_func(layer,channel,path_to_audio,iterations,octaves):

	audio_path = os.path.dirname(path_to_audio)
	audio_filename = os.path.basename(path_to_audio)
	audio_filename_new = os.path.join(audio_path,'dreamed_on_'+audio_filename)

	model_fn = 'tensorflow_inception_graph.pb'

	# creating TensorFlow session and loading the model
	graph = tf.Graph()
	sess = tf.InteractiveSession(graph=graph)
	with tf.gfile.FastGFile(model_fn, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	t_input = tf.placeholder(np.float32, name='input') # define the input tensor
	imagenet_mean = 117.0
	t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
	tf.import_graph_def(graph_def, {'input':t_preprocessed})

	# start with a gray image with a little noise
	img_noise = np.random.uniform(size=(224,224,3)) + 100.0

	def showarray(a, fmt='jpeg'):
		a = np.uint8(np.clip(a, 0, 1)*255)
		f = BytesIO()
		PIL.Image.fromarray(a).save(f, fmt)
		display(Image(data=f.getvalue()))
		
	def visstd(a, s=0.1):
		'''Normalize the image range for visualization'''
		return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

	def T(layer):
		'''Helper for getting layer output tensor'''
		return graph.get_tensor_by_name("import/%s:0"%layer)

	def tffunc(*argtypes):
		'''Helper that transforms TF-graph generating function into a regular one.
		See "resize" function below.
		'''
		placeholders = list(map(tf.placeholder, argtypes)) # map all argtypes to a placeholder
		def wrap(f):
			out = f(*placeholders)
			def wrapper(*args, **kw):
				return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
			return wrapper
		return wrap

	# Helper function that uses TF to resize an image
	def resize(img, size):
		img = tf.expand_dims(img, 0)
		return tf.image.resize_bilinear(img, size)[0,:,:,:]
	resize = tffunc(np.float32, np.int32)(resize)


	def calc_grad_tiled(img, t_grad, tile_size=512):
		'''Compute the value of tensor t_grad over the image in a tiled way.
		Random shifts are applied to the image to blur tile boundaries over 
		multiple iterations.'''
		sz = tile_size
		h, w = img.shape[:2]
		sx, sy = np.random.randint(sz, size=2)
		img_shift = np.roll(np.roll(img, sx, axis=1), sy, axis=0)
		grad = np.zeros_like(img)
		for y in range(0, max(h-sz//2, sz),sz):
			for x in range(0, max(w-sz//2, sz),sz):
				sub = img_shift[y:y+sz,x:x+sz]
				g = sess.run(t_grad, {t_input:sub})
				grad[y:y+sz,x:x+sz] = g
		return np.roll(np.roll(grad, -sx, 1), -sy, 0)

	# start with a gray image with a little noise
	img_noise = np.random.uniform(size=(224,224,3)) + 100.0

	def render_deepdream(t_obj, img0=img_noise,
						 iter_n=10, step=1.5, octave_n=16, octave_scale=1.4):
		t_obj_scaled = tf.mul(t_obj, tf.to_float(tf.log(t_obj) < .8*tf.reduce_max(t_obj)))
		t_score = tf.reduce_mean(t_obj_scaled) # defining the optimization objective
		t_grad = tf.gradients(t_score, t_input)[0]

		# split the image into a number of octaves
		img = img0.copy()
		octaves = []
		for i in range(octave_n-1):
			hw = img.shape[:2]
			lo = resize(img, np.int32(np.float32(hw)/octave_scale))
			hi = img-resize(lo, hw)
			img = lo
			octaves.append(hi)
		
		# generate details octave by octave
		for octave in range(octave_n):
			if octave>0:
				hi = octaves[-octave]
				img = resize(img, hi.shape[:2])+hi
			for i in range(iter_n):
				g = calc_grad_tiled(img, t_grad)
				img += g*(step / (np.abs(g).mean()+1e-7))
				#print('.',end = ' ')
			#clear_output()
			#showarray(img/255.0)
		return img/255.0

	# load the audio
	y, sr = librosa.load(path_to_audio, sr=44100)
	# do the stft
	nfft = 2048
	hop = 256
	y_stft = librosa.core.stft(y, n_fft = nfft, hop_length = hop, center=True)

	# Separate the magnitude and phase
	y_stft_mag1, y_stft_ang = librosa.magphase(y_stft)

	# scale the spectrogram such that its values correspond to 0-255 (16-bit rgb amplitude)
	nonlin = 1.0/8.0
	y_stft_mag = np.power(y_stft_mag1, nonlin)
	y_stft_mag = np.flipud((1 - y_stft_mag/y_stft_mag.max()))
	y_stft_mag_rgb = np.zeros([y_stft_mag.shape[0], y_stft_mag.shape[1], 3])
	y_stft_mag_rgb[:, :, 0] = y_stft_mag
	y_stft_mag_rgb[:, :, 1] = y_stft_mag
	y_stft_mag_rgb[:, :, 2] = y_stft_mag
	img = 255*y_stft_mag_rgb

	og_spectrogram_img = img/255.0
	og_spectrogram = librosa.display.specshow(data=np.log(np.abs(y_stft_mag1)), sr=sr, x_axis='time', y_axis='log')
	dream_spec_img = render_deepdream(T(layer)[:,:,:,channel], img, iter_n=iterations, octave_n=octaves)


	# undo processing to bring the image back from 0-255 to original scale
	deepdream_out = np.flipud(dream_spec)
	deepdream_out = (1 - deepdream_out) * y_stft_mag.max()
	deepdream_out = np.power(deepdream_out, 1/nonlin)
	# flatten the three channels and normalize over number of channels
	deepdream_out = np.sum(deepdream_out, axis=2) / 3.0
	# show the new log-spectrogram
	dream_spectrogram = librosa.display.specshow(np.log(np.abs(deepdream_out)), sr=sr, x_axis='time', y_axis='log')

	# add back in the original phase
	deepdream_out_orig = deepdream_out.copy()
	deepdream_out = deepdream_out * y_stft_ang
	# add back in original phase
	_, orig_stft_ang = librosa.magphase(y_stft)
	deepdream_out = deepdream_out_orig * orig_stft_ang
	output = librosa.core.istft(deepdream_out, hop_length=hop, win_length=nfft, center=True)
	librosa.output.write_wav(os.path.join(audio_filename_new), output, sr)

	return og_spectrogram_img, og_spectrogram, dream_spec_img, dream_spectrogram

deepdream_func(layer,channel,path_to_audio,iterations,octaves)
