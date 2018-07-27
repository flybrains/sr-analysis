import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, ZeroPadding1D, Conv2DTranspose, UpSampling2D, Dense, BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras import optimizers

def FCN(num_GCaMP_channels=4, num_kinematic_channels=1, downsample_output=True, summary=False):
	"""
	Creates and compiles "1D" FCN for analysis of 10s time snippets
	
	:param num_GCaMP_channels: int between (1,4) specifying dimensionality of input volume
	:param num_kinematic_channels: int between (1,3) specifying dimensionality of output volume
	:param downsample_output: Reduce output space by half in the time dimension. Improves training
							speed and preserves generalized learned structures.
	:param summary: prints summary after compiling
	"""

	inputs = Input(shape=(1,112,num_GCaMP_channels), name="main_input")

	x1 = Conv2D(8, kernel_size=(1,3), padding="same")(inputs)
	x1 = MaxPooling2D(pool_size=(1,2), strides=(2))(x1)
	x1 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x1)

	x2 = Conv2D(16, kernel_size=(1,3),padding="same")(x1)
	x2 = MaxPooling2D(pool_size=(1,2), strides=(2))(x2)
	x2 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x2)

	x3 = Conv2D(32, kernel_size=(1,3),padding="same")(x2)
	x3 = MaxPooling2D(pool_size=(1,2), strides=(2))(x3)
	x3 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x3)

	x4 = Conv2D(64, kernel_size=(1,3),padding="same")(x3)
	x4 = MaxPooling2D(pool_size=(1,2), strides=(2))(x4)
	x4 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x4)

	x5 = Conv2D(128, kernel_size=(1,7))(x4)

	x6 = Dense(128)(x5)

	x7 = Conv2DTranspose(128, kernel_size=(1,7), padding='valid')(x6)
	x7 = UpSampling2D((1,2))(x7)
	x7 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x7)

	x8 = Conv2DTranspose(64, kernel_size=(1,3), padding='same')(x7)
	x8 = UpSampling2D((1,2))(x8)
	x8 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x8)

	x9 = Conv2DTranspose(32, kernel_size=(1,3), padding='same')(x8)
	x9 = UpSampling2D((1,2))(x9)
	x9 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x9)

	x10 = Conv2DTranspose(16, kernel_size=(1,3), padding='same')(x9)
	x10 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x10)

	if downsample_output==False:
		x10 = UpSampling2D((1,2))(x10)

	x11 = Conv2DTranspose(4, kernel_size=(1,3), padding='same')(x10)
	x11 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x11)

	x11 = Conv2DTranspose(2, kernel_size=(1,3), padding='same')(x11)
	x11 = Conv2DTranspose(1, kernel_size=(1,3), padding='same')(x11)
	x12 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x11)

	model = Model(inputs=inputs, outputs=x12)

	sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005)

	model.compile(optimizer=sgd, loss = 'mean_squared_error', metrics = ['mse'])

	return model










def deep_FCN(num_GCaMP_channels=4, num_kinematic_channels=1, downsample_output=True, summary=False):
	"""
	Creates and compiles "1D" FCN for analysis of 10s time snippets
	
	:param num_GCaMP_channels: int between (1,4) specifying dimensionality of input volume
	:param num_kinematic_channels: int between (1,3) specifying dimensionality of output volume
	:param downsample_output: Reduce output space by half in the time dimension. Improves training
							speed and preserves generalized learned structures.
	:param summary: prints summary after compiling
	"""
	inputs = Input(shape=(1,112,num_GCaMP_channels), name="main_input")

	x1 = Conv2D(64, kernel_size=(1,3), padding="same")(inputs)
	x1 = Conv2D(64, kernel_size=(1,3),padding="same")(x1)
	x1 = MaxPooling2D(pool_size=(1,2), strides=(2))(x1)
	x1 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x1)

	x2 = Conv2D(128, kernel_size=(1,3),padding="same")(x1)
	x2 = Conv2D(128, kernel_size=(1,3),padding="same")(x2)
	x2 = MaxPooling2D(pool_size=(1,2), strides=(2))(x2)
	x2 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x2)

	x3 = Conv2D(256, kernel_size=(1,3),padding="same")(x2)
	x3 = Conv2D(256, kernel_size=(1,3),padding="same")(x3)
	x3 = MaxPooling2D(pool_size=(1,2), strides=(2))(x3)
	x3 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x3)

	x4 = Conv2D(512, kernel_size=(1,3),padding="same")(x3)
	x4 = Conv2D(512, kernel_size=(1,3),padding="same")(x4)
	x4 = MaxPooling2D(pool_size=(1,2), strides=(2))(x4)
	x4 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x4)

	x5 = Conv2D(1024, kernel_size=(1,7))(x4)

	x6 = Dense(1024)(x5)

	#Presently doing unpooling in hacked way (No filtering for max value, naive upsampling)
	# https://arxiv.org/pdf/1311.2901v3.pdf
	# https://stackoverflow.com/questions/44991470/using-tensorflow-layers-in-keras

	#Need to create custom keras layer to take mask from MaxPooling2D (similar to TF's max_pool_with_argmax)
	#mask is used in unpooling layer

	x7 = Conv2DTranspose(512, kernel_size=(1,7), padding='valid')(x6)
	x7 = UpSampling2D((1,2))(x7)
	x7 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x7)

	x8 = Conv2DTranspose(512, kernel_size=(1,3), padding='same')(x7)
	x8 = Conv2DTranspose(512, kernel_size=(1,3), padding='same')(x8)
	#x8 = UpSampling2D((1,2))(x8)
	x8 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x8)

	x9 = Conv2DTranspose(512, kernel_size=(1,3), padding='same')(x8)
	x9 = Conv2DTranspose(256, kernel_size=(1,3), padding='same')(x9)
	x9 = UpSampling2D((1,2))(x9)
	x9 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x9)

	x10 = Conv2DTranspose(256, kernel_size=(1,3), padding='same')(x9)
	x10 = Conv2DTranspose(128, kernel_size=(1,3), padding='same')(x10)
	x10 = UpSampling2D((1,2))(x10)
	x10 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x10)

	x11 = Conv2DTranspose(128, kernel_size=(1,3), padding='same')(x10)
	x11 = Conv2DTranspose(64, kernel_size=(1,3), padding='same')(x11)
	x11 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x11)

	x11 = Conv2DTranspose(2, kernel_size=(1,3), padding='same')(x11)
	x11 = Conv2DTranspose(1, kernel_size=(1,3), padding='same')(x11)
	x12 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x11)

	model2 = Model(inputs=inputs, outputs=x12)

	sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005)

	model2.compile(optimizer=sgd, loss = 'mean_squared_error', metrics = ['mse'])

	return model2