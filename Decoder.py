from keras.layers import Input, Dense
from Parameters import intermediate_dim, original_dim
from keras.models import Model
from keras.utils import plot_model

def Decoder(input_data):
    # build decoder model
    latent_inputs = Input(input_data, name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    return decoder