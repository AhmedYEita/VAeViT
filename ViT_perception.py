# Created on Thu Aug 10 12:00:00 2023 - *******************


# Import libraries
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import pickle as pkl


# Load data -------------------------------------------------------------------
#latent vectors of the trained model
with open('models/z_train_MN10_12v_512ld_100e.pkl', 'rb') as f:
    x_data = pkl.load(f)
#labels of the trained model
with open('models/y_train_MN10_12v_512ld_100e.pkl', 'rb') as f:
    y_data = pkl.load(f)
# =============================================================================


# Define parameters -----------------------------------------------------------
#optimizer params
learning_rate = 0.001
weight_decay = 0.0001   #to be used in AdamW optimizer

#training params
batch_size = 128
num_epochs = 100

#dataset preprocessing params
num_views = 12
num_classes = 10

# ViT model params
projection_dim = 512    #size of the hidden dimenssion feature vectors (used in preprocessing to convert patches into vectors),
num_heads = 4           #attention head (4 query,key,value matrecies)
#params for skip connection:
transformer_units = [
    projection_dim * 2,
    projection_dim,
    ]   #size of the transformer layers
transformer_layers = 6      #stacked transformer blocks
mlp_head_units = [2048, 1024]   #size of the dense layers of the final classifier
# =============================================================================


# Preprocessing input data ----------------------------------------------------
def prep_data(data, labels, split=0.2):
    #group latent vectors of all views of the same object together
    data_grouped = [data[n:n+num_views] for n in range(0, len(data), num_views)]
    data_grouped = np.array(data_grouped)
    
    #do the same for labels
    new_labels = []
    for i,l in enumerate(labels):
        if i==0 or i%num_views==0:
            new_labels.append(l)
    new_labels = np.reshape(new_labels, (len(new_labels), 1))
    
    #shuffle data
    data_grouped, new_labels = shuffle(data_grouped, new_labels, random_state=42)
    
    #dataset split
    x_train, x_test, y_train, y_test = train_test_split(data_grouped, new_labels, test_size=split, random_state=42)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = prep_data(x_data, y_data, split=0.2)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
# =============================================================================


# Define necessary funcs ------------------------------------------------------
#overwriting the Dense layer to add a skip connection, so rather than using model.add.Dense, we will be overwirting this with a custom MLP layer
def mlp (x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


#define the positional embeddings class
class PosEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PosEncoder, self).__init__()
        self.num_views = num_views          #define the number of batches
        self.position_embedding = layers.Embedding(             #define the embedding tables that takes from an index code into a positional embedding table and returns as vector of the same dimenssion as the embedded patches
            input_dim=num_views, output_dim=projection_dim)     #output_dim: dimension of the dense embedding
        
    def call(self, onedimage):
        positions = tf.range(start=0, limit=self.num_views, delta=1)
        encoded = onedimage + self.position_embedding(positions)
        return encoded
# =============================================================================


# Build the ViT model ---------------------------------------------------------
# The Transformer blocks produce a [batch_size, num_patches, projection_dim] tensor, which is processed via an classifier head with softmax to produce the final class probabilities output.
def create_vit_classifier():
    inputs = layers.Input(shape=(num_views,projection_dim))
    
    encoded_images = PosEncoder(num_views, projection_dim)(inputs)    # take all of those patches, create the vector embeddings for them, add the positional embeddings
    
    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers): #6 layers
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_images)
        
        # Create a multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1,x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_images])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        
        # Skip connection 2
        encoded_images = layers.Add()([x3,x2])
    
    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_images)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Add MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    
    # Classify outputs
    logits = layers.Dense(num_classes)(features)
    
    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=logits)

    print(model.summary())

    return model
# =============================================================================
    

# Define the objects and run the training -------------------------------------
#training run func
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)
    
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy")
            ],
        )

    # checkpoint_filepath = "...modelnet10_12v_ViT_logs_directory..."   
    # checkpoint_callback = keras.callbacks.ModelCheckpoint(
    #     checkpoint_filepath,
    #     monitor="val_accuracy",
    #     save_best_only=True,
    #     save_weights_only=True,
    #     )
    
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        # callbacks=[checkpoint_callback],
        )
    
    # model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history


#run the training
vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)
# =============================================================================
