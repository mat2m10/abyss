import os
# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from termcolor import colored, cprint
from os import listdir
from abyss.glob import glob
from abyss.helpers import metrics_viewer
from abyss.models import create_decoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1],
    args[3].split(':')[0]: args[3].split(':')[1],
    args[4].split(':')[0]: args[4].split(':')[1],
    args[5].split(':')[0]: args[5].split(':')[1],
    args[6].split(':')[0]: args[6].split(':')[1],
}

# Set default values if arguments are not provided

try:
    epoch = int(arguments['epoch'])
except ValueError:
    epoch = 150
try:
    PCs = int(arguments['PCs'])
except ValueError:
    PCs = 15

bin = arguments['bin']

try:
    latent_dim = int(arguments['bottle_latent'])
except ValueError:
    latent_dim = 15


path_dims = f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{latent_dim}_dim_latent/"
existing_dims = [f for f in os.listdir(path_dims) if f.startswith("reduction")]
try:
    pheno_name = arguments['pheno']
    accu = pd.read_pickle(f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{pheno_name}_best_bottles.pkl")
    dims = int(accu.sort_values(by='accuracy', ascending=False)[['bottleneck']].iloc[0, 0])
except Exception as e:
    print(e)
    pheno_name = "no_pheno"
    try:
        dims = int(arguments['bottle'])
        print(dims)
    except:
        path_dims = f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{latent_dim}_dim_latent/"
        dims = int(existing_dims[-1].split("_")[3])

# Define output directory for MAF probability landscape
path_output = f"{glob.PATH_GENO}/08_MAF_probability_landscape/{pheno_name}/bin_{bin}/{dims}_bottleneck_dims/"
os.makedirs(path_output, exist_ok=True)

# Define input paths
path_input = f"{glob.PATH_GENO}/04_LD_blocks/{PCs}_PCs/bin_{bin}"
path_dims = f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{latent_dim}_dim_latent/"
tmp = [f for f in existing_dims if int(f.split("_")[3]) == dims][0]
path_dims = path_dims + tmp

path_metrics = f"{glob.PATH_GENO}/08_MAF_probability_landscape/{pheno_name}/bin_{bin}/{dims}_bottleneck_dims/bin_{bin}_metric_plots/"
os.makedirs(path_metrics, exist_ok=True)

dims = pd.read_pickle(path_dims)
# Find input files for reconstruction
onlyfiles_input = [f for f in listdir(path_input) if not f.startswith("mapping")]
onlyfiles_output = [f for f in listdir(path_output)]
onlyfiles = [x for x in onlyfiles_input if x not in onlyfiles_output]

i = len(onlyfiles_output)

for file in onlyfiles:
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,     # Minimum change to qualify as an improvement
        patience=20,         # Stop after 10 epochs with no improvement
        verbose=1,           # Print a message when stopping
        mode='min'           # Stop when the quantity monitored has stopped decreasing
    )
    num_one = file.split("_")[0]
    num_two = file.split("_")[1]

    print(f"Progress for bin {bin} = {round((i/len(onlyfiles_input))*100,2)}% now processing file nr {num_one}_{num_two}")

    # Define hyperparameters
    activation = 'elu'
    depth = 1

    # Load SNP data
    path_genos = os.path.join(path_input, file)
    genos = pd.read_pickle(path_genos)

    # Convert data to tensors
    y = tf.convert_to_tensor(genos, dtype=tf.float32)
    X = tf.convert_to_tensor(dims, dtype=tf.float32)

    # Merge the dimensions data with SNP data
    merge = dims.join(genos)
    merge = merge.sample(frac=1).reset_index(drop=True)  # Randomize and reset index

    # Split data into training and testing sets
    X_data = merge[list(dims.columns)].values
    y_data = merge[list(genos.columns)].values

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    n_inputs = X_train.shape[1]
    n_outputs = y_train.shape[1]

    # Fitting the autoencoder for reconstruction
    t = time.time()
    print(colored(f"Performing dimensionality reconstruction on {genos.shape[1]} SNPs of {genos.shape[0]} samples using an autoencoder with {epoch} epochs", 'yellow'), colored(' [This may take some time, please be patient]', 'red'))

    # Create decoder model
    decoder = create_decoder(n_inputs, n_outputs, activation, depth, optimizer='Adam', loss='mse', output_activation='linear')

    # Train the decoder
    history = decoder.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epoch,
        verbose=0,
        callbacks=[early_stopping]
    )

    # Generate and save the metric plot
    metric_plt = metrics_viewer(history)
    metric_plt.savefig(f"{path_metrics}/file_{file.split('.')[0]}_metrics_reconstruction_epoch_{epoch}_bottleneck_{n_inputs}.png")
    metric_plt.close()

    elapsed = round(time.time() - t, 2)
    time_string = str(datetime.timedelta(seconds=elapsed))
    time_array = time_string.split(':')
    hours = time_array[0]
    minutes = time_array[1]
    seconds = round(float(time_array[2]), 2)

    cprint(f"This took {hours} hours, {minutes} minutes and {seconds} seconds", 'red')

    # Get the lower-dimensional hidden representation
    AF = decoder(X).numpy()

    # Create a DataFrame with the hidden representation
    AF = pd.DataFrame(AF, columns=list(genos.columns))
    AF['FID'] = genos.index
    AF.set_index('FID', inplace=True)

    AF = (AF+1)/2

    # Save the hidden representation
    AF.to_pickle(f"{path_output}{file}")
    i += 1

print(f"========\nDONE for {pheno_name}\n========")
