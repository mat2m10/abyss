import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored, cprint


import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd  # Import Pandas and Numpy to create databases
import tensorflow as tf  # Import tensorflow to do tensor manipulations

from abyss.helpers import metrics_viewer
from abyss.models import create_autoencoder
from abyss.glob import glob
from keras.callbacks import EarlyStopping


# Parse command line arguments
args = sys.argv

arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1],
    args[3].split(':')[0]: args[3].split(':')[1],
    args[4].split(':')[0]: args[4].split(':')[1],
}

bin = arguments['bin']
#chrom = 21
# Set default values if arguments are not provided
try:
    epoch = int(arguments['epoch'])
except ValueError:
    epoch = 15
try:
    PCs = int(arguments['PCs'])
except ValueError:
    PCs = 15
try:
    bottleneck = int(arguments['bottle'])
except ValueError:
    bottleneck = 15

# Define paths and directories
PATH_GENO = glob.PATH_GENO
path_input = f"{PATH_GENO}/04_LD_blocks/{PCs}_PCs/bin_{bin}/"
os.makedirs(f"{PATH_GENO}/05_Latent_populations", exist_ok=True)
os.makedirs(f"{PATH_GENO}/05_Latent_populations/{bottleneck}_bottleneck_dims/", exist_ok=True)
os.makedirs(f"{PATH_GENO}/05_Latent_populations/{bottleneck}_bottleneck_dims/bin_{bin}", exist_ok=True)
os.makedirs(f"{PATH_GENO}/05_Latent_populations/{bottleneck}_bottleneck_dims/bin_{bin}_metric_plots", exist_ok=True)

path_output = f"{PATH_GENO}/05_Latent_populations/{bottleneck}_bottleneck_dims/bin_{bin}/"
path_metrics = f"{PATH_GENO}/05_Latent_populations/{bottleneck}_bottleneck_dims/bin_{bin}_metric_plots/"

# List of input files
onlyfiles_input = [f for f in listdir(path_input) if f.split(".")[0] != "mapping"]

# List of output files
onlyfiles_output = [f for f in listdir(path_output) if isfile(join(path_output, f))]

# Get files that need processing
onlyfiles = [x for x in onlyfiles_input if x not in onlyfiles_output]
i = len(onlyfiles_output)

i = len(onlyfiles_output)




for file in onlyfiles:
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,     # Minimum change to qualify as an improvement
        patience=20,         # Stop after 10 epochs with no improvement
        verbose=1,           # Print a message when stopping
        mode='min'           # Stop when the quantity monitored has stopped decreasing
        )
    
    num = file.split("_")[0]
    print(f"Progress for bin {bin} = {round((i/len(onlyfiles_input))*100,2)}% now processing file nr {num}")

    # Define hyperparameters
    activation = 'elu'
    depth = 1

    # Load genotypic data
    genos = pd.read_pickle(join(path_input, file))
    X = tf.convert_to_tensor(genos, dtype=tf.float32)
    frac = genos.sample(frac=1)
    len_test = round(frac.shape[0] / 10) * 3
    len_train = frac.shape[0] - len_test
    X_train = frac.head(len_train)
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = frac.tail(len_test)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    input_len = int(tf.shape(X_train)[1])

    # Create and fit the autoencoder
    t = time.time()

    
    print(colored(f"Performing dimensionality reduction on {frac.shape[1]} features of {frac.shape[0]} samples using an autoencoder with {epoch} epochs", 'yellow'), colored(' [This may take some time, please be patient]', 'red'))
    AE = create_autoencoder(input_len, bottleneck, activation, depth, optimizer='Adam', loss='mse', output_activation='linear')
    history = AE.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        epochs=epoch,
        verbose=0,
        callbacks=[early_stopping]
    )

    # Generate and save the metric plots
    metric_plt = metrics_viewer(history)
    metric_plt.savefig(f"{path_metrics}/file_{file}_metrics_ae_epoch_{epoch}_bottleneck_{bottleneck}.png")
    metric_plt.close()

    elapsed = round(time.time() - t, 2)
    time_string = str(datetime.timedelta(seconds=elapsed))
    time_array = time_string.split(':')
    hours = time_array[0]
    minutes = time_array[1]
    seconds = round(float(time_array[2]), 2)
    cprint(f"This process took {hours} hours, {minutes} minutes, and {seconds} seconds", 'red')

    # Get the lower dimensionality hidden representation
    hidden_repr = AE.layers[1](X).numpy()

    # Depending on how many dimensions in the bottleneck, create column names
    cols = [f'dim{i+1}' for i in range(np.shape(hidden_repr)[1])]
    low_dim = pd.DataFrame(hidden_repr, columns=cols)
    low_dim['FID'] = list(genos.index)
    low_dim.set_index('FID', inplace=True)
    # Save the lower-dimensional representation
    low_dim.to_pickle(join(path_output, file))
    i += 1
