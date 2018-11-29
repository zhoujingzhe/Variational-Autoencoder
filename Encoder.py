import keras.backend as K
from keras.layers import Input, Dense, Lambda
from Parameters import intermediate_dim, latent_dim
from keras.models import Model
from keras.utils import plot_model
from Util import sampling

def Encoder(input_data):
    # build encoder model
    Input_data = Input(input_data, name='encoder_input')
    x = Dense(intermediate_dim, input_dim=K.shape(Input_data), activation='relu')(Input_data)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(Input_data, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    return encoder