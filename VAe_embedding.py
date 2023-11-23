# Created on Wed Aug 9 13:50:00 2023 - by AHmed Yasser Eita


# Import libraries
import time
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import backend as K
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle as pkl



# DATA ------------------------------------------------------------------------
# Loading the dataset and preprocessing
path = "modelnet10_12v"   #dataset directory
class_names = os.listdir(path)    #list all the class names and append them in a variable

# Append the Test dataset paths and labels
train_image_paths, train_labels = [], []
for i1 in class_names:  #bathtub
    dir1 = os.path.join(path,i1+"//train")  #dataset/bathtub/train
    for i2 in os.listdir(dir1):
        # if ".png" in i2:      #use when implementing ModelNet40 - 12 views as there are non-image files, add a tab space to the next line
        train_image_paths.append(os.path.join(dir1, i2))    #append dataset/bathtub/train/bathtub_000000001_001.jpg
        for n,j in enumerate(class_names):
            if j == i1:
                train_labels.append(n)      #append 'bathtub'

# Append the Test dataset paths and labels
test_image_paths, test_labels = [], []
for i1 in class_names:  #bathtub
    dir1 = os.path.join(path,i1+"//test")   #dataset/bathtub/test
    for i2 in os.listdir(dir1):
        test_image_paths.append(os.path.join(dir1, i2))     #append dataset/bathtub/test/bathtub_000000140_001.jpg
        for n,j in enumerate(class_names):
            if j == i1:
                test_labels.append(n)      #append 'bathtub'


# Read the train images in gray scale and resize them to (64x64x1)
train_images_list = []
for image in train_image_paths:
    im = cv2.imread(image,cv2.IMREAD_GRAYSCALE) 
    im = cv2.resize(im, (64,64))
    train_images_list.append(im)
    
# Read the test images in gray scale and resize them to (64x64x1)
test_images_list = []
for image in test_image_paths:
    im = cv2.imread(image,cv2.IMREAD_GRAYSCALE) 
    im = cv2.resize(im, (64,64))
    test_images_list.append(im)
    

# Convert the train lists to numpy array type (to do the following preprocessing)
train_images_list = np.asarray(train_images_list)
train_labels = np.asarray(train_labels)
train_images_list = train_images_list.astype('float32')
train_images_list = train_images_list / 255

# Convert the test lists to numpy array type (to do the following preprocessing)
test_images_list = np.asarray(test_images_list)
test_labels = np.asarray(test_labels)
test_images_list = test_images_list.astype('float32')
test_images_list = test_images_list / 255


# Reshape the train and test arrays to be in the shape (N x img_width x img_height x num_channels)
img_width = train_images_list.shape[1]
img_height = train_images_list.shape[2]
num_channels = 1

train_images_list = train_images_list.reshape(train_images_list.shape[0], img_height, img_width, num_channels)
test_images_list = test_images_list.reshape(test_images_list.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)


# shuffle the test data
test_images_list, test_labels= shuffle(test_images_list, test_labels)


# Batch the train data (without shuffle because we will need to combine all views for each object together in one array in the next stage of the model "ViT_perception.py")
train_dataset = tf.data.Dataset.from_tensor_slices(train_images_list)
batch_size = 128
train_dataset = train_dataset.batch(batch_size)
print('train_dataset length:',len(train_dataset))
# =============================================================================


# MODEL -----------------------------------------------------------------------
# Latent dimenssion (Bottleneck size)
latent_dim = 512


# Encoder
## takes an input of size [None, 64, 64, 1]
## In each block, the image is downsampled by a factor of two
def encoder_func(l_d):
   
    encoder_input = keras.Input(shape=input_shape, name='encoder_input')
 
    # Block-1
    x = layers.Conv2D(32, kernel_size=5, strides = 1, padding = 'same', name='conv_1')(encoder_input)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    
    # Block-2
    x = layers.Conv2D(64, kernel_size=5, strides = 2, padding = 'same', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)
    
    # Block-3
    x = layers.Conv2D(128, kernel_size=5, strides = 2, padding = 'same', name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    
    # Block-4
    x = layers.Conv2D(256, kernel_size=5, strides = 2, padding = 'same', name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)
    
    # Block-5
    x = layers.Conv2D(l_d, kernel_size=5, strides = 2, padding = 'same', name='conv_5')(x)
    x = layers.BatchNormalization(name='bn_5')(x)
    x = layers.LeakyReLU(name='lrelu_5')(x)
    
        
    # Final Block
    global shape_before_flatten
    shape_before_flatten = K.int_shape(x)[1:]
    flatten = layers.Flatten(name='flatten')(x) ## convert the matrix shape into a vector form
    
    ## the two outputs of the encoder network (latent-variables)
    mean = layers.Dense(l_d, name='mean')(flatten)
    log_var = layers.Dense(l_d, name='log_var')(flatten)
    
    model = tf.keras.Model(encoder_input, outputs = [mean, log_var], name="Encoder")
    return model

encoder = encoder_func(latent_dim) # later on the this defined object will be always used and not the func
print(encoder.summary())


# Sampling function
## takes mean and variance tensors as input and return the latent vector in tensor format
def sampling(mean,log_var,l_d):
    out = layers.Lambda(sampling_reparameterization, output_shape=(l_d, ), name='encoder_output')([mean, log_var])
    return out

## called by the above function
## to be fed the output of the encoder i.e. mean and variance
def sampling_reparameterization(distribution_params):
    mean, log_var = distribution_params
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.) # Returns a tensor with normal distribution of values
    z = mean + K.exp(log_var / 2) * epsilon
    return z
## outputs log-variance instead of the variance to maintain numerical stability


# Decoder
## In each block, the image is upsampled by a factor of two
def decoder_func(l_d):
     
    ## takes an input of size [None, latent_dim]
    decoder_input = keras.Input(shape=(l_d,), name='decoder_input')
    ## Define a Dense layer with the same length as the output of the encoder then reshape it to a matrix format
    y = layers.Dense(np.prod(shape_before_flatten), name='dense_1')(decoder_input)
    y = layers.Reshape(shape_before_flatten, name='reshape_Layer')(y)
    
    # Block-1
    y = layers.Conv2DTranspose(256, 5, strides= 2, padding='same',name='conv_transpose_1')(y)
    y = layers.BatchNormalization(name='bn_1')(y)
    y = layers.LeakyReLU(name='lrelu_1')(y)
    
    # Block-2
    y = layers.Conv2DTranspose(128, 5, strides= 2, padding='same',name='conv_transpose_2')(y)
    y = layers.BatchNormalization(name='bn_2')(y)
    y = layers.LeakyReLU(name='lrelu_2')(y)
    
    # Block-3
    y = layers.Conv2DTranspose(64, 5, strides= 2, padding='same',name='conv_transpose_3')(y)
    y = layers.BatchNormalization(name='bn_3')(y)
    y = layers.LeakyReLU(name='lrelu_3')(y)
    
    # Block-4
    y = layers.Conv2DTranspose(32, 5, strides= 2, padding='same',name='conv_transpose_4')(y)
    y = layers.BatchNormalization(name='bn_4')(y)
    y = layers.LeakyReLU(name='lrelu_4')(y)
    
    # Block-5
    output = layers.Conv2DTranspose(1, 5, strides= 1, activation = "sigmoid", padding='same',name='conv_transpose_5')(y)
    
    model = tf.keras.Model(decoder_input, output, name="Decoder")
    return model

decoder = decoder_func(latent_dim) # later on the this defined object will be always used and not the func
print(decoder.summary())


# DEFINE THE LOSS FUNCTION
## vae loss = reconstruction loss + KL divergence
def reconstruction_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

def kl_loss(mu, log_var):
    loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return loss

def vae_loss(y_true, y_pred, mu, log_var):
    return reconstruction_loss(y_true, y_pred) + (1 / (64*64)) * kl_loss(mu, log_var)

# =============================================================================


# PROGRESS VISUALIZATION FUNC -------------------------------------------------

# Function to save a set of images from the test set during the training to show progress
def save_images(epoch, step, l_d):
    
    m, v = encoder.predict(test_images_list[:25])
    z = sampling(m,v,l_d)
    reconst = decoder.predict(z)
    fig = plt.figure(figsize=(15, 10))
     
    for i in range(25):
        ax = fig.add_subplot(5, 5, i+1)
        ax.axis('off')
        ax.text(0.5, -0.15, class_names[test_labels[i]], fontsize=10, ha='center', transform=ax.transAxes)
         
        ax.imshow(reconst[i, :,:,:], cmap='gray')
    
    output_path = "output/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(output_path + "LatentDim_{:04d}_epoch_{:04d}_step_{:04d}.jpg".format(l_d, epoch, step))
    plt.close()
# =============================================================================    


# TRAIN -----------------------------------------------------------------------
optimizer = keras.optimizers.Adam(0.0001, 0.5)
epochs = 100

for epoch in range(1, epochs + 1):
    print("\nEpoch: ", epoch)
    start = time.time()
    for step, train_batch in enumerate(train_dataset):
        #provides hooks that give the user control over what is or is not watched.
        with tf.GradientTape() as enc, tf.GradientTape() as dec:
            #we first pass the batches to the encoder
            mean, log_var = encoder(train_batch)
            z = sampling(mean, log_var, latent_dim)
            if epoch==epochs:
                if step==0:
                    z_train=z
                else:
                    z_train = np.append(z_train, z.numpy(), axis=0)
                    
            #the output latent is finally fed to the decoder
            generated_images = decoder(z)
            ## The loss is computed over the images generated by the decoder
            loss = vae_loss(train_batch, generated_images, mean, log_var)

            #compute the gradient while training all sorts of neural networks.
            #the computed gradients are essential in order to do backpropagation
            #to correct the errors of the neural network to make it gradualy improve.
            gradients_of_enc = enc.gradient(loss, encoder.trainable_variables)
            gradients_of_dec = dec.gradient(loss, decoder.trainable_variables)
                 
            #to update the new model's attribute from the calculated gradients
            optimizer.apply_gradients(zip(gradients_of_enc, encoder.trainable_variables))
            optimizer.apply_gradients(zip(gradients_of_dec, decoder.trainable_variables))
            
            print("Epoch: %i - Step: %i - Total loss: %f" % (epoch, step, 0.01*np.mean(loss)))
            
            if step % 10 == 0:
                save_images(epoch, step, latent_dim)
 
    print ('Time for epoch %i is %.2f sec\n' %(epoch, time.time()-start))
# =============================================================================


# VAe classification accuracy of latent vectors -------------------------------
# Use SVM classifier
svm = SVC(kernel='linear')
x_trn, x_tst, y_trn, y_tst = train_test_split(z_train, train_labels, test_size=0.2, random_state=42)
svm.fit(x_trn, y_trn)
y_pred = svm.predict(x_tst)
acc = accuracy_score(y_tst, y_pred)
print('Classification accuracy of VAE = ', acc)

# Confusion matrix
model_conf=confusion_matrix(y_tst, y_pred)
# print('\nConfusion matrix of %i latent dimenssion and %i epochs:' %(latent_dim,epochs))
# print(model_conf)
conf_table = pd.DataFrame(model_conf,columns=class_names,index=class_names)
conf_plot = sns.heatmap(conf_table,annot=True,cmap='BuGn',fmt='g')
# plt.title('64p gray%i l_d, %i e, classification acc: %.0f%%' %(latent_dim,epochs,(acc*100)))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('output/conf_matrix_pixels_{:03d}_LatentDim_{:03d}_epoch_{:03d}.pdf'.format(img_height, latent_dim, epoch), bbox_inches='tight')
plt.show()
# =============================================================================


# Save the trained latent vectors and their labels ----------------------------
with open('models/z_train_MN10_12v_512ld_100e.pkl', 'wb') as f:
    pkl.dump(z_train, f, protocol=pkl.HIGHEST_PROTOCOL)
    
with open('models/y_train_MN10_12v_512ld_100e.pkl', 'wb') as f:
    pkl.dump(train_labels, f, protocol=pkl.HIGHEST_PROTOCOL)
# =============================================================================
