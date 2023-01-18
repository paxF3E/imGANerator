import tensorflow as tf
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from PIL import Image

BATCH_SIZE = 128
dataset = open('Data/trainimglist.txt').read().splitlines()

def generate_batch(BATCH_SIZE):
    batch=random.sample(dataset,BATCH_SIZE) # sample image names
    real_images=np.empty(shape=[BATCH_SIZE,64,64,3]) # prep ndarray for images
    encoded_sentence=np.empty(shape=[BATCH_SIZE,4800]) # prep ndarray for encoded sentences
    for i in range(len(batch)):
        real_images[i]=Image.open(r'Data/val2017/'+batch[i][:-4]+'.png').convert('RGB').resize((64,64))
        # resize to 560x560 and RGB
        sentence_list=np.load('Data/encoded_vector/val_annotations/'+batch[i][:-4]+'.npy')
        encoded_sentence[i] = sentence_list[random.randint(0,np.shape(sentence_list)[0]-1)]
    return real_images,encoded_sentence

def generator(BATCH_SIZE, noise_len, encoded_caption_tensor, training_flag, reuse=False):
    '''
        this section is for the conditioned latent vector
        condition here being the sentence encodings
        modeled as a normal distribution, with mean and standard deviation
        then appended with noise vector (x100)
        which is then fed to the generator
    '''
    # Xavier initializer in keras = Glorot uniform initializer in tensorflow
    initializer = tf.initializers.GlorotUniform()
    g_mu_w0 = tf.Variable(initializer(shape=[4800,128]), name='g_mu_w0', trainable=True)
    g_mu_b0 = tf.Variable(initializer(shape=[128]), name='g_mu_b0', trainable=True)
    g_mu_a0 = tf.matmul(encoded_caption_tensor, g_mu_w0) + g_mu_b0
    g_mu_a0 = tf.maximum(g_mu_a0, 0.3*g_mu_a0)  # leaky relu with slope 0.3

    g_sd_w0 = tf.Variable(initializer(shape=[4800,128]), name='g_sd_w0', trainable=True)
    g_sd_b0 = tf.Variable(initializer(shape=[128]), name='g_sd_b0', trainable=True)
    g_sd_a0 = tf.matmul(encoded_caption_tensor, g_sd_w0) + g_sd_b0
    g_sd_a0 = tf.maximum(g_sd_a0, 0.3*g_sd_a0)  # leaky relu with slope 0.3

    # sampling from conditioned latent vector
    epsilon = tf.random.normal(shape=[BATCH_SIZE,128], mean = 0.0, stddev = 1.0)
    g_cond = tf.multiply(g_sd_a0, epsilon) + g_mu_a0

    # sampling from noise vector
    z = tf.random.normal(shape=[BATCH_SIZE, noise_len], mean = 0.0, stddev = 1.0)
    inp_for_gen = tf.concat([z, g_cond], axis=1)

    '''
        this section defines the layers of the generator
    '''
    # first (input) layer with 4x4x1024 output
    g_w1 = tf.Variable(initializer(shape=[noise_len + 128, 4*4*1024]), name='g_w1', trainable=True)
    g_b1 = tf.Variable(initializer(shape=[4*4*1024]), name='g_b1', trainable=True)
    g_a1 = tf.matmul(inp_for_gen, g_w1) + g_b1
    g_a1 = tf.reshape(g_a1, shape=[-1, 4, 4, 1024])
    g_a1 = tf.layers.batch_normalization(g_a1, training=training_flag, reuse=reuse)
    g_a1 = tf.maximum(g_a1, 0.3*g_a1) # leaky relu with slope 0.3

    # second layer with 4x4x1024 input with 8x8x512 output
    g_w2 = tf.Variable(initializer(shape=[3, 3, 512, 1024]), name='g_w2', trainable=True)
    g_b2 = tf.Variable(initializer(shape=[512]), name='g_b2', trainable=True)
    g_a2 = tf.nn.conv2d_transpose(g_a1, g_w2, output_shape=[BATCH_SIZE, 8, 8, 512], strides=[1, 2, 2, 1], padding='SAME') + g_b2
    g_a2 = tf.layers.batch_normalization(g_a2, training=training_flag, reuse=reuse, scale=True)
    g_a2 = tf.maximum(g_a2, 0.3*g_a2) # leaky relu with slope 0.3

    # third layer with 8x8x512 input with 16x16x256 output
    g_w3 = tf.Variable(initializer(shape=[3, 3, 256, 512]), name='g_w3', trainable=True)
    g_b3 = tf.Variable(initializer(shape=[256]), name='g_b3', trainable=True)
    g_a3 = tf.nn.conv2d_transpose(g_a2, g_w3, output_shape=[BATCH_SIZE, 16, 16, 256], strides=[1, 2, 2, 1], padding='SAME') + g_b3
    g_a3 = tf.layers.batch_normalization(g_a3, training=training_flag, reuse=reuse, scale=True)
    g_a3 = tf.maximum(g_a3, 0.3*g_a3) # leaky relu with slope 0.3

    # fourth layer with 16x16x256 input with 32x32x128 output
    g_w4 = tf.Variable(initializer(shape=[3, 3, 3, 256]), name='g_w4', trainable=True)
    g_b4 = tf.Variable(initializer(shape=[128]), name='g_b4', trainable=True)
    g_a4 = tf.nn.conv2d_transpose(g_a3, g_w4, output_shape=[BATCH_SIZE, 32, 32, 128], strides=[1, 2, 2, 1], padding='SAME') + g_b4
    g_a4 = tf.layers.batch_normalization(g_a4, training=training_flag, reuse=reuse, scale=True)
    g_a4 = tf.maximum(g_a4, 0.3*g_a4) # leaky relu with slope 0.3

    # fifth layer with 32x32x128 input with 64x64x3 output
    g_w5 = tf.Variable(initializer(shape=[3, 3, 3, 128]), name='g_w5', trainable=True)
    g_b5 = tf.Variable(initializer(shape=[3]), name='g_b5', trainable=True)
    g_a5 = tf.nn.conv2d_transpose(g_a4, g_w5, output_shape=[BATCH_SIZE, 64, 64, 3], strides=[1, 2, 2, 1], padding='SAME') + g_b5
    
    return tf.nn.sigmoid(g_a5), g_mu_a0, g_sd_a0


def discriminator(image_tensor, encoded_caption_tensor, reuse=False):
    '''

    '''
    

    return
