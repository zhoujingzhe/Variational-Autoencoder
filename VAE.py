from keras.models import Model
from keras.layers import Input
from keras.losses import mse, binary_crossentropy
from Parameters import original_dim
import keras.backend as K
def VAE(input_data, decoder, encoder):
    # VAE model = encoder + decoder
    # instantiate VAE model
    Input_data = Input(input_data)
    z_mean, z_log_var, z = encoder(Input_data)
    outputs = decoder(z)
    vae = Model(Input_data, outputs, name='vae_mlp')

    if False:
        reconstruction_loss = mse(Input_data, outputs)
    else:
        reconstruction_loss = binary_crossentropy(Input_data,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    return vae