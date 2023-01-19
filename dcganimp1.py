import tensorflow as tf
import numpy as np
import os
import random
from matplotlib import pyplot as plt
from PIL import Image


BATCH_SIZE = 128
NOISE_LEN = 100
EPOCHS = 10
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


def generator(batch_size, noise_len, encoded_caption_tensor, training_flag, reuse):
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
    epsilon = tf.random.normal(shape=[batch_size,128], mean = 0.0, stddev = 1.0)
    g_cond = tf.multiply(g_sd_a0, epsilon) + g_mu_a0

    # sampling from noise vector
    z = tf.random.normal(shape=[batch_size, noise_len], mean = 0.0, stddev = 1.0)
    inp_for_gen = tf.concat([z, g_cond], axis=1)

    '''
        this section defines the layers of the generator
    '''
    # first (input) layer with 4x4x1024 output
    g_w1 = tf.Variable(initializer(shape=[noise_len + 128, 4*4*1024]), name='g_w1', trainable=True)
    g_b1 = tf.Variable(initializer(shape=[4*4*1024]), name='g_b1', trainable=True)
    g_a1 = tf.matmul(inp_for_gen, g_w1) + g_b1
    g_a1 = tf.reshape(g_a1, shape=[-1, 4, 4, 1024])
    g_a1 = tf.compat.v1.layers.batch_normalization(g_a1, training=training_flag, reuse=reuse)
    g_a1 = tf.maximum(g_a1, 0.3*g_a1) # leaky relu with slope 0.3

    # second layer with 4x4x1024 input with 8x8x512 output
    g_w2 = tf.Variable(initializer(shape=[3, 3, 512, 1024]), name='g_w2', trainable=True)
    g_b2 = tf.Variable(initializer(shape=[512]), name='g_b2', trainable=True)
    g_a2 = tf.nn.conv2d_transpose(g_a1, g_w2, output_shape=[batch_size, 8, 8, 512], strides=[1, 2, 2, 1], padding='SAME') + g_b2
    g_a2 = tf.compat.v1.layers.batch_normalization(g_a2, training=training_flag, reuse=reuse, scale=True)
    g_a2 = tf.maximum(g_a2, 0.3*g_a2) # leaky relu with slope 0.3

    # third layer with 8x8x512 input with 16x16x256 output
    g_w3 = tf.Variable(initializer(shape=[3, 3, 256, 512]), name='g_w3', trainable=True)
    g_b3 = tf.Variable(initializer(shape=[256]), name='g_b3', trainable=True)
    g_a3 = tf.nn.conv2d_transpose(g_a2, g_w3, output_shape=[batch_size, 16, 16, 256], strides=[1, 2, 2, 1], padding='SAME') + g_b3
    g_a3 = tf.compat.v1.layers.batch_normalization(g_a3, training=training_flag, reuse=reuse, scale=True)
    g_a3 = tf.maximum(g_a3, 0.3*g_a3) # leaky relu with slope 0.3

    # fourth layer with 16x16x256 input with 32x32x128 output
    g_w4 = tf.Variable(initializer(shape=[3, 3, 3, 256]), name='g_w4', trainable=True)
    g_b4 = tf.Variable(initializer(shape=[128]), name='g_b4', trainable=True)
    g_a4 = tf.nn.conv2d_transpose(g_a3, g_w4, output_shape=[batch_size, 32, 32, 128], strides=[1, 2, 2, 1], padding='SAME') + g_b4
    g_a4 = tf.compat.v1.layers.batch_normalization(g_a4, training=training_flag, reuse=reuse, scale=True)
    g_a4 = tf.maximum(g_a4, 0.3*g_a4) # leaky relu with slope 0.3

    # fifth layer with 32x32x128 input with 64x64x3 output
    g_w5 = tf.Variable(initializer(shape=[3, 3, 3, 128]), name='g_w5', trainable=True)
    g_b5 = tf.Variable(initializer(shape=[3]), name='g_b5', trainable=True)
    g_a5 = tf.nn.conv2d_transpose(g_a4, g_w5, output_shape=[batch_size, 64, 64, 3], strides=[1, 2, 2, 1], padding='SAME') + g_b5
    
    return tf.nn.sigmoid(g_a5), g_mu_a0, g_sd_a0


def discriminator(image_tensor, encoded_caption_tensor, reuse):
    '''
        this section defines the layers of the discriminator
        caption encodings is mapped to a fully connected layer with 4x4x128 output
        then output is convolved and augmented along the channels
        after convolution, a 1x1x1024 tensor is obtained, which is furhter reshaped to 1-D vector and passed on to performance metrics
    '''
    # Xavier initializer in keras = Glorot uniform initializer in tensorflow
    initializer = tf.initializers.GlorotUniform()
    d_w0 = tf.Variable(initializer(shape=[4800, 4*4*128]), name='d_w0', trainable=True)
    d_b0 = tf.Variable(initializer(shape=[4*4*128]), name='d_b0', trainable=True)
    d_a0 = tf.matmul(encoded_caption_tensor, d_w0) + d_b0
    d_a0 = tf.maximum(d_a0, 0.3*d_a0) # leaky relu with slope 0.3
    d_a0 = tf.reshape(d_a0, shape=[BATCH_SIZE, 4, 4, 128])

    # first layer with 4x4x128 input with 32x32x64 output
    d_w1 = tf.Variable(initializer(shape=[3, 3, 3, 64]), name='d_w1', trainable=True)
    d_b1 = tf.Variable(initializer(shape=[64]), name='d_b1', trainable=True)
    d_a1 = tf.nn.conv2d(image_tensor, d_w1, strides=[1, 2, 2, 1], padding='SAME') + d_b1
    d_a1 = tf.maximum(d_a1, 0.3*d_a1) # leaky relu with slope 0.3
    d_a1 = tf.nn.dropout(d_a1, 0.7) # dropout with keep probability 0.7

    # second layer with 32x32x64 input with 16x16x128 output
    d_w2 = tf.Variable(initializer(shape=[3, 3, 64, 128]), name='d_w2', trainable=True)
    d_b2 = tf.Variable(initializer(shape=[128]), name='d_b2', trainable=True)
    d_a2 = tf.nn.conv2d(d_a1, d_w2, strides=[1, 2, 2, 1], padding='SAME') + d_b2
    d_a2 = tf.maximum(d_a2, 0.3*d_a2) # leaky relu with slope 0.3
    d_a2 = tf.nn.dropout(d_a2, 0.7) # dropout with keep probability 0.7

    # third layer with 16x16x128 input with 8x8x256 output
    d_w3 = tf.Variable(initializer(shape=[3, 3, 128, 256]), name='d_w3', trainable=True)
    d_b3 = tf.Variable(initializer(shape=[256]), name='d_b3', trainable=True)
    d_a3 = tf.nn.conv2d(d_a2, d_w3, strides=[1, 2, 2, 1], padding='SAME') + d_b3
    d_a3 = tf.maximum(d_a3, 0.3*d_a3) # leaky relu with slope 0.3
    d_a3 = tf.nn.dropout(d_a3, 0.7) # dropout with keep probability 0.7

    # fourth layer with 8x8x256 input with 4x4x512 output
    d_w4 = tf.Variable(initializer(shape=[3, 3, 256, 512]), name='d_w4', trainable=True)
    d_b4 = tf.Variable(initializer(shape=[512]), name='d_b4', trainable=True)
    d_a4 = tf.nn.conv2d(d_a3, d_w4, strides=[1, 2, 2, 1], padding='SAME') + d_b4
    d_a4 = tf.maximum(d_a4, 0.3*d_a4) # leaky relu with slope 0.3
    d_a4 = tf.nn.dropout(d_a4, 0.7) # dropout with keep probability 0.7

    # augmenting the layer outputs; convolved a4 and caption weights a0
    d_a4 = tf.concat([d_a4, d_a0], axis=3)

    # fifth layer with 4x4x1024 input with 1x1x1024 output
    d_w5 = tf.Variable(initializer(shape=[4, 4, 512+128, 1024]), name='d_w5', trainable=True)
    d_b5 = tf.Variable(initializer(shape=[1024]), name='d_b5', trainable=True)
    d_a5 = tf.nn.conv2d(d_a4, d_w5, strides=[1, 1, 1, 1], padding='VALID') + d_b5
    d_a5 = tf.maximum(d_a5, 0.3*d_a5) # leaky relu with slope 0.3
    d_a5 = tf.nn.dropout(d_a5, 0.7) # dropout with keep probability 0.7
    d_a5 = tf.reshape(d_a5, shape=[BATCH_SIZE, 1024])

    # sixth layer with 1024 input with 1 output
    d_w6 = tf.Variable(initializer(shape=[1024, 1]), name='d_w6', trainable=True)
    d_b6 = tf.Variable(initializer(shape=[1]), name='d_b6', trainable=True)
    d_a6 = tf.matmul(d_a5, d_w6) + d_b6 # no activation function for the last layer
    '''
        discretion made by the discriminator is passed on to compute the loss
    '''
    return d_a6


# placeholders for the input images and captions
tf.compat.v1.disable_eager_execution()
encoded_caption_tensor = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, 4800], name='encoded_caption_tensor')
real_image_tensor = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, 3], name='real_image_tensor')
arbitary_caption_tensor = tf.compat.v1.placeholder(tf.float32, shape=[BATCH_SIZE, 4800], name='arbitary_caption_tensor')
training_flag = tf.compat.v1.placeholder(tf.bool, name='training_flag')


## generating images from caption encoding and random noise from generator
generated_img_tensor, mu1, sd1 = generator(batch_size=BATCH_SIZE, noise_len=NOISE_LEN, encoded_caption_tensor=encoded_caption_tensor, training_flag=training_flag, reuse=False)

## passing the generated images, real image with real descriptions and real image with arbitary descriptions to the discriminator
Dg = discriminator(generated_img_tensor, encoded_caption_tensor, False)
Dx = discriminator(real_image_tensor, encoded_caption_tensor, True)
Db = discriminator(real_image_tensor, arbitary_caption_tensor, True)


## cross entropy loss for the generator
W_g_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)-0.1))

# KLDivergence loss for the conditioned latent variable
N1 = tf.compat.v1.distributions.Normal(mu1, sd1)
N1_n = tf.compat.v1.distributions.Normal(tf.zeros([1,128]), tf.ones([1,128]))
W_g_kl = tf.reduce_mean(input_tensor=tf.compat.v1.distributions.kl_divergence(N1, N1_n))
W_g = W_g_ce + 2.0*W_g_kl # 2.0 divergence loss for lagrangian multiplier


## cross entropy loss for the discriminator
W_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)-0.1))
W_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)+0.1))
W_d_arbitary = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Db, labels=tf.zeros_like(Db)+0.1))
W_d = W_d_real + W_d_fake + W_d_arbitary


## preparing optimizers
trainable_vars = tf.compat.v1.trainable_variables()
generator_variables = [var for var in trainable_vars if 'g_' in var.name]
discriminator_variables = [var for var in trainable_vars if 'd_' in var.name]
variable_names = [v.name for v in tf.compat.v1.trainable_variables()]

update_optimizer = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_optimizer):
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=25e-6, beta_1=0.5).minimize(W_d, var_list=discriminator_variables, tape=tf.GradientTape())
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=25e-6, beta_1=0.5).minimize(W_g, var_list=generator_variables, tape=tf.GradientTape())
'''
    use GradientTape for optimization, as when loss function is provided, it is required.
    last edit : fixing this optimization failure in above code block
'''

## initializing generator and discriminator
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
session = tf.compat.v1.Session(config=config)
session.run(tf.compat.v1.global_variables_initializer())
saver = tf.compat.v1.train.Saver()
saver.restore(session, './model/model.ckpt')


## training the model
curr_epoch = 1
while curr_epoch <= EPOCHS:
    real_images, encoded_captions = generate_batch(batch_size=BATCH_SIZE)
    _, arbitary_caption = generate_batch(batch_size=BATCH_SIZE)

    # updating discriminator
    _, d_loss = session.run(
                        [d_optimizer, W_d], 
                        feed_dict={
                                real_image_tensor: real_images, encoded_caption_tensor: encoded_captions, arbitary_caption_tensor: arbitary_caption, training_flag: True})

    # sampling mini batch
    real_images, encoded_captions = generate_batch(batch_size=BATCH_SIZE)

    # updating generator
    _, g_loss = session.run(
                        [g_optimizer, W_g],
                        feed_dict={
                                real_image_tensor: real_images, encoded_caption_tensor: encoded_captions, training_flag: True})

    print(f"#{curr_epoch}>>\tGenerator Loss: {g_loss}\tDiscriminator Loss: {d_loss}")

    # saving the model and images
    if curr_epoch % 5 == 0:
        saver.save(session, 'model_saves/model.ckpt', global_step=curr_epoch)
        gen_img = session.run(
                        generated_img_tensor, 
                        feed_dict={
                                real_image_tensor: real_images, 
                                encoded_caption_tensor: encoded_captions, 
                                training_flag: False})

        plt.imsave(arr = gen_img[0], fname = f'generated_images/{curr_epoch}_gen.png')
        plt.imsave(arr = real_images[0], fname = f'generated_images/{curr_epoch}_real.png')
        curr_epoch = curr_epoch + 1
