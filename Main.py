'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from VAE import VAE
from Encoder import Encoder
from Decoder import Decoder
from Parameters import x_train, x_test, y_test, latent_dim, input_shape, epochs, batch_size
from keras.utils import plot_model
from Util import plot_results
if __name__ == '__main__':
    encoder = Encoder(input_data=input_shape)
    decoder = Decoder(input_data=(latent_dim,))
    models = (encoder, decoder)
    data = (x_test, y_test)
    vae = VAE(input_data=input_shape, encoder=encoder, decoder=decoder)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae, to_file='vae_mlp.png', show_shapes=True)
    vae.fit(x=x_train, y=None, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))
    vae.save_weights('vae_mlp_mnist.h5')
    plot_results(models, data, batch_size=batch_size, model_name="vae_mlp")