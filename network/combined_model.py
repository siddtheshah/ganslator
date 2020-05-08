from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt

from network.discriminator import *
from network.generator import *

import numpy as np
import os

class GANslator:
    def __init__(self,
                 sample_size=16384,
                 feature_size=32,
                 r_scale=16,
                 z_dim=100,
                 filter_dim=64):
        # Store parameters for building sub-models
        self.sample_size = sample_size
        self.feature_size = feature_size
        self.r_scale = r_scale
        self.z_dim = z_dim
        self.filter_dim = filter_dim

        # Input shape
        self.input_shape = tf.constant([self.sample_size])

        self.noise_shape = tf.constant([self.z_dim])

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        self.optimizer = Adam(0.0002, 0.5)

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        # Build the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()

        # Build the combined models
        self.d_combined = self.build_combined_discriminator_model()
        self.g_combined = self.build_combined_generator_model()

    def build_combined_generator_model(self):
        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Input images from both domains
        signal_A = tf.keras.layers.Input(shape=self.input_shape)
        signal_B = tf.keras.layers.Input(shape=self.input_shape)
        noise = tf.keras.layers.Input(shape=self.noise_shape)

        features_A = MelSpecFeatures(self.feature_size)(signal_A)
        features_B = MelSpecFeatures(self.feature_size)(signal_B)

        # Translate images to the other domain
        fake_B = self.g_AB({"Cond_in": features_A, "Z_in": noise})
        fake_A = self.g_BA({"Cond_in": features_B, "Z_in": noise})
        # Translate images back to original domain
        reconstr_A = self.g_BA({"Cond_in": fake_B, "Z_in": noise})[:, :, 0]
        reconstr_B = self.g_AB({"Cond_in": fake_A, "Z_in": noise})[:, :, 0]
        # print(reconstr_A.get_shape())
        # Identity mapping of images
        img_A_id = self.g_BA({"Cond_in": features_A, "Z_in": noise})[:, :, 0]
        img_B_id = self.g_AB({"Cond_in": features_B, "Z_in": noise})[:, :, 0]
        # print(reconstr_B.get_shape())

        # Only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False
        self.g_AB.trainable = True
        self.g_BA.trainable = True

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        combined = Model(inputs=[signal_A, signal_B, noise],
                              outputs=[valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id])
        combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                            optimizer=self.optimizer)

        return combined

    def build_combined_discriminator_model(self):
        #-------------------------
        # Construct Computational
        #   Graph of Discriminators
        #-------------------------

        # Input images from both domains
        signal_A = tf.keras.layers.Input(shape=self.input_shape)
        signal_B = tf.keras.layers.Input(shape=self.input_shape)
        noise = tf.keras.layers.Input(shape=self.noise_shape)

        features_A = MelSpecFeatures(self.feature_size)(signal_A)
        features_B = MelSpecFeatures(self.feature_size)(signal_B)

        print(features_A.get_shape())
        # print(features_B.compute_output_shape(signal_B.get_shape()))

        fake_B = self.g_AB({"Cond_in": features_A, "Z_in": noise})
        fake_A = self.g_BA({"Cond_in": features_B, "Z_in": noise})

        # Only train the discriminators
        self.d_A.trainable = True
        self.d_B.trainable = True
        self.g_AB.trainable = False
        self.g_BA.trainable = False

        # Get discriminator outputs
        d_valid_A = self.d_A(features_A)
        d_valid_B = self.d_B(features_B)
        d_fake_A = self.d_A(fake_A)
        d_fake_B = self.d_B(fake_B)

        combined = Model(inputs=[signal_A, signal_B, noise], outputs=[d_valid_A, d_valid_B, d_fake_A, d_fake_B])
        combined.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])

        return combined
        #
        # dA_loss_real = self.d_A.train_on_batch(features_A, features_valid)
        # dA_loss_fake = self.d_A.train_on_batch(fake_A, features_fake)
        # dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
        #
        # dB_loss_real = self.d_B.train_on_batch(features_B, features_valid)
        # dB_loss_fake = self.d_B.train_on_batch(fake_B, features_fake)
        # dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
        # # Total disciminator loss
        # d_loss = 0.5 * np.add(dA_loss, dB_loss)


    def build_generator(self):
        return GeneratorModel(self.sample_size, self.feature_size, self.z_dim, self.r_scale, self.filter_dim)

    def build_discriminator(self):
        return DiscriminatorModel(self.sample_size, self.feature_size, self.r_scale, self.z_dim)

    def save_to_path(self, model_path):
        generator_path = os.path.join(model_path, "generator.h5")
        self.g_combined.save_weights(generator_path, True)

        discriminator_path = os.path.join(model_path, "discriminator.h5")
        self.d_combined.save_weights(discriminator_path, True)

    def load_from_path(self, model_path):
        generator_path = os.path.join(model_path, "generator.h5")
        self.g_combined.load_weights(generator_path)
        discriminator_path = os.path.join(model_path, "discriminator.h5")
        self.d_combined.load_weights(discriminator_path)

    def train(self, dataset, epochs, batch_size=1, save_interval=50, save_path=''):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones(batch_size)
        fake = np.zeros(batch_size)

        for epoch in range(epochs):
            for batch_i, (signals_A, signals_B) in enumerate(dataset.batch(batch_size)):
                noise = tf.random.normal((batch_size, self.z_dim))

                # ----------------------
                #  Train Discriminators
                # ----------------------

                d_loss = self.d_combined.train_on_batch([signals_A, signals_B, noise],
                                                        [valid, valid, fake, fake])


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.g_combined.train_on_batch([signals_A, signals_B, noise],
                                                        [valid, valid,
                                                        signals_A, signals_B,
                                                        signals_A, signals_B],)

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print(
                    "[Epoch {:d}/{:d}] [Batch {:d}] [D loss: {:f}, acc: {:3f}%] [G loss: {:05f}, adv: {:05f}, recon: {:05f}, id: {:05f}] time: {} " \
                    .format(int(epoch), int(epochs),
                            int(batch_i),
                            d_loss[0], 100 * d_loss[1],
                            g_loss[0],
                            np.mean(g_loss[1:3]),
                            np.mean(g_loss[3:5]),
                            np.mean(g_loss[5:6]),
                            elapsed_time))

            # If at save interval => save generated image samples
            if save_path and epoch % save_interval == 0:
                self.sample_sounds(epoch, batch_i, signals_A, signals_B, noise)
                self.save_to_path(save_path)

    def sample_sounds(self, epoch, batch_i, signals_A, signals_B, noise):
        os.makedirs('audio/generated_samples', exist_ok=True)
        # Get fake and reconstructed outputs
        
        prefix = "results/epoch_{}_batch_{}".format(epoch, batch_i)
        wav_suffix = ".wav"
        img_suffix = ".jpg"

        plt.figure()
        plt.plot(signals_A[0].numpy())
        plt.savefig(prefix + "signal_A" + img_suffix)

        plt.clf()
        plt.plot(signals_B[0].numpy())
        plt.savefig(prefix + "signal_B" + img_suffix)

        plt.clf()
        plt.plot(fake_A[0].numpy())
        plt.savefig(prefix + "fake_A" + img_suffix)

        plt.clf()
        plt.plot(fake_B[0].numpy())
        plt.savefig(prefix + "fake_B" + img_suffix)

        plt.clf()
        plt.plot(reconstr_A[0].numpy())
        plt.savefig(prefix + "reconstr_A" + img_suffix)

        plt.clf()
        plt.plot(reconstr_B[0].numpy())
        plt.savefig(prefix + "reconstr_B" + img_suffix)

        # Save some sample sounds
        fake_A_encode = tf.audio.encode_wav(fake_A)
        tf.io.write_file(prefix + "fake_A" + wav_suffix, fake_A_encode)

        fake_B_encode = tf.audio.encode_wav(fake_B)
        tf.io.write_file(prefix + "fake_B" + wav_suffix, fake_B_encode)

if __name__ == '__main__':
    gan = GANslator()
