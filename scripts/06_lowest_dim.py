import os
# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from termcolor import colored, cprint
from abyss.helpers import metrics_viewer
from abyss.models import create_autoencoder
from abyss.glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping



# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1],
    args[3].split(':')[0]: args[3].split(':')[1],
    args[4].split(':')[0]: args[4].split(':')[1]

}

bin = arguments['bin']
# Set default values if arguments are not provided
try:
    epoch = int(arguments['epoch'])
except ValueError:
    epoch = 150

try:
    bottleneck = int(arguments['bottle'])
    bottles = [bottleneck]
except ValueError:
    bottles = np.geomspace(1, 25, 3, dtype=int)
    bottles = np.unique(bottles)
    # Numbers to add
    numbers_to_add = np.array([1, 2, 3, 4, 5])

    # Loop through each number in numbers_to_add
    for number in numbers_to_add:
        # Check if the number is not already in the existing_array
        if number not in bottles:
            # Append the number to the existing_array
            bottles = np.append(bottles, number)
try:
    bottle_latent = int(arguments['bottle_latent'])
except ValueError:
    bottle_latent = 15

# Get PATH_GENO from abyss.glob module
PATH_GENO = glob.PATH_GENO
path_input = f"{PATH_GENO}/06_Concated_latent/{bottle_latent}_bottleneck_dims/concatted_bin_{bin}.pkl"
path_output = f"{PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{bottle_latent}_dim_latent"
os.makedirs(path_output, exist_ok=True)
genos = pd.read_pickle(path_input)

for bottleneck in bottles:
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,     # Minimum change to qualify as an improvement
        patience=20,         # Stop after 10 epochs with no improvement
        verbose=1,           # Print a message when stopping
        mode='min'           # Stop when the quantity monitored has stopped decreasing
    )
    
    # Define EarlyStopping callback

    # Activations
    activation = 'elu'
    depth = 1
        # divide in train - test sets
    X = tf.convert_to_tensor(genos, dtype=tf.float32)
    # Splitting the data into train and test sets
    X_train, X_test = train_test_split(genos, test_size=0.3)  # 70% training and 30% testing
    
    # Convert to TensorFlow tensors
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    input_len = int(tf.shape(X_train)[1])

    # Fitting the autoencoder
    t = time.time()
    print(colored(f"Doing dimensionality reduction on {genos.shape[1]} features of {genos.shape[0]} samples using an autoencoder with {epoch} epochs", 'yellow'), colored(' [can take time, be patient]', 'red'))
    AE = create_autoencoder(input_len, bottleneck, activation, depth, optimizer = 'Adam', loss = 'mse', output_activation = 'linear')
    history = AE.fit(X_train,
                X_train,
                validation_data=(X_test, X_test),
                epochs = epoch,
                verbose = 0,
                callbacks=[early_stopping])

    metric_plt = metrics_viewer(history)
    metric_plt.savefig(f"{path_output}/metrics_reduction_concatted_bin_{bin}_bottle_{bottleneck}.png")
    metric_plt.close()

    elapsed = round(time.time() - t,2)
    time_string = str(datetime.timedelta(seconds=elapsed))
    time_array = time_string.split(':')
    hours = time_array[0]
    minutes = time_array[1]
    seconds = round(float(time_array[2]),2)

    cprint(f"This took {hours} hours, {minutes} minutes and {seconds} seconds",'red')

    # Getting the lower dimensionality hidden representation
    hidden_repr = AE.layers[1](X).numpy()

    # depending on how many dims in the bottleneck
    cols = []
    for i2 in range(np.shape(hidden_repr)[1]):
        cols.append(f'dim{i2+1}')
    low_dim = pd.DataFrame(hidden_repr, columns = cols)
    low_dim['FID'] = list(genos.index)
    low_dim.set_index('FID', inplace=True)
    elapsed = round(elapsed)
    low_dim.to_pickle(
        f"{path_output}/reduction_of_ae_{bottleneck}_dims_{elapsed}_seconds_{epoch}_epoch.pkl")
