import os
import sys
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import datetime
from termcolor import colored, cprint
import numpy as np
import pandas as pd  # Import Pandas and Numpy to create databases
import tensorflow as tf  # Import tensorflow to do tensor manipulations

from sklearn.model_selection import train_test_split
from abyss.models import create_decoder
from abyss.glob import glob
from abyss.stats import ols_many

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import warnings
warnings.filterwarnings('ignore')
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

try:
    PCs = int(arguments['PCs'])
except ValueError:
    PCs = 15

try:
    bottle_latent = int(arguments['bottle_latent'])
except ValueError:
    bottle_latent = 15

bin = int(arguments['bin'])
pheno_name = arguments['pheno']
pheno = pd.read_pickle(f"{glob.PATH_PHENO}/{pheno_name}.pkl")

# If you want to find the best bottleneck with covariates
try:
    covname = arguments['covs']
    path_cov = f"{glob.PATH_COVS}/{covname}.pkl"
    covs = pd.read_pickle(path_cov)
    # Select the first 2 columns
    n_first_columns = 5
    covs = covs.iloc[:, :n_first_columns]
except Exception as e:
    print(e)
    pass

# Default training epoch is 1000
try:
    epoch = int(arguments['epoch'])
except ValueError as v:
    epoch = 150

pheno = pd.read_pickle(f"{glob.PATH_PHENO}/{pheno_name}.pkl")
pheno_holdout = pd.read_pickle(f"{glob.PATH_PHENO}/{pheno_name}_holdout.pkl")
# Count the occurrences of each phenotype value
counts = pheno['phenotype'].value_counts()

# Identify the smaller group
min_count = counts.min()

# Sample the rows where phenotype is 0
pheno_0_sample = pheno[pheno['phenotype'] == 0].sample(min_count)

# Get all rows where phenotype is 1
pheno_1_sample = pheno[pheno['phenotype'] == 1].sample(min_count)

# Concatenate the two balanced groups
balanced_pheno = pd.concat([pheno_0_sample, pheno_1_sample])


geno_ids = pd.read_pickle(glob.PATH_usefull + "/id_samples.pkl")
path_input = glob.PATH_GENO + f"/04_LD_blocks/{PCs}_PCs/bin_{bin}/"
LDs = [f for f in os.listdir(path_input) if f.split('.')[0] != 'mapping']
split_files = [f.split('_') for f in LDs]
# Find the file whose first number is most in the middle
middle_values = [int(s[0]) for s in split_files]
middle_value = min(middle_values, key=lambda x: abs(x - sum(middle_values)/len(middle_values)))

# Selecting the files based on the criteria
selected_middle = [f for f in LDs if f.startswith(str(middle_value) + '_')][0]

genos = pd.read_pickle(f"{path_input}/{selected_middle}")

PATH_bin_bottle = f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}"
PATH_bottles = f"{PATH_bin_bottle}/{bottle_latent}_dim_latent"

bottle_array = []
accuracies = []
inputs = [f for f in os.listdir(PATH_bottles) if f.split("_")[0] != "metrics"]


bottle_array = []
accuracies = []
inputs = [f for f in os.listdir(PATH_bottles) if f.split("_")[0] != "metrics"]

# The scaled genotypes and their merges with the phenotypes do not depend on
# the bottleneck file, so compute them once instead of once per bottleneck.
scaled_data = scaler.fit_transform(genos)
geno_to_test = pd.DataFrame(scaled_data, columns=genos.columns)
geno_to_test['FID'] = genos.index
geno_to_test.set_index('FID', inplace=True)
geno_holdout = pd.merge(geno_to_test, pheno_holdout[['phenotype','FID']], on='FID')

temp = pd.merge(geno_to_test, balanced_pheno[['phenotype','FID']], on='FID')

for bottlefile in inputs:
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        min_delta=0.001,     # Minimum change to qualify as an improvement
        patience=20,         # Stop after 10 epochs with no improvement
        verbose=1,           # Print a message when stopping
        mode='min'           # Stop when the quantity monitored has stopped decreasing
    )
    bottle_nr = int(bottlefile.split("_")[3])
    bottle = pd.read_pickle(PATH_bottles + "/" + bottlefile)

    # First finding the hyperparams corresponding to the file
    activation = 'elu'
    depth = 1
     # divide in train - test sets
    y = tf.convert_to_tensor(genos, dtype=tf.float32)
    X = tf.convert_to_tensor(bottle, dtype=tf.float32)
    
    # Use train_test_split to split the data
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.3)
    
    # Convert them back to TensorFlow tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

    n_inputs = int(tf.shape(X_train)[1])
    n_outputs = int(tf.shape(y_train)[1])

    # Fitting the autoencoder
    t = time.time()
    print(colored(f"for {pheno_name} Doing dimensionality reconstruction on {genos.shape[1]} snps of {genos.shape[0]} samples using an autoencoder with {epoch} epochs", 'yellow'), colored(' [can take time, be patient]', 'red'))

    decoder = create_decoder(n_inputs, n_outputs, activation, depth,
                         optimizer='Adam', loss='mse', output_activation='linear')

    history = decoder.fit(X_train,
                y_train,
                validation_data=(X_test, y_test),
                epochs = epoch,
                verbose = 0,

                callbacks=[early_stopping])
    
    elapsed = round(time.time() - t,2)
    time_string = str(datetime.timedelta(seconds=elapsed))
    time_array = time_string.split(':')
    hours = time_array[0]
    minutes = time_array[1]
    seconds = round(float(time_array[2]),2)

    cprint(f"This took {hours} hours, {minutes} minutes and {seconds} seconds",'red')

    # Getting the lower dimensionality hidden representation
    AF = decoder(X).numpy()

    # depending on how many dims in the bottleneck
    AF = (pd.DataFrame(AF, columns = list(genos.columns))+1)/2
    AF.columns = [col + '_corr' for col in AF.columns]
    AF['FID'] = bottle.index
    AF.set_index('FID', inplace=True)

    total = pd.merge(temp, AF, on='FID')
    y = total[['phenotype']]
    PRS = 0
    filtered_columns = [col for col in total.columns 
                    if not col.startswith('PC') 
                    and not col.endswith('_corr') 
                    and col not in ['phenotype', 'FID']]

    # Fit all per-SNP OLS models at once (same results as sm.OLS per SNP);
    # designs statsmodels treats specially still go through statsmodels.
    designs = [[snp_id, f"{snp_id}_corr"] for snp_id in filtered_columns]
    fast = {}
    unique_cols = total.columns[~total.columns.duplicated(keep=False)]
    col_pos = {c: i for i, c in enumerate(unique_cols)}
    eligible = [i for i, cols in enumerate(designs) if all(c in col_pos for c in cols)]
    if eligible and pd.api.types.is_numeric_dtype(total['phenotype']):
        used = sorted({c for i in eligible for c in designs[i]}, key=col_pos.get)
        used_pos = {c: j for j, c in enumerate(used)}
        M = total[used].to_numpy(dtype=np.float64)
        cols = np.array([[used_pos[c] for c in designs[i]] for i in eligible])
        res = ols_many(total['phenotype'].to_numpy(dtype=np.float64), M, cols)
        for row, i in enumerate(eligible):
            if not res['fallback'][row]:
                fast[i] = (res['params'][row][1], res['pvalues'][row][1])

    for i, (snp_id, snp_columns) in enumerate(zip(filtered_columns, designs)):
        if i in fast:
            effect_size_snp, P = fast[i]
        else:
            X = total[snp_columns]
            X2 = sm.add_constant(X)
            est = sm.OLS(y, X2)
            est2 = est.fit()
            # Extract coefficients (betas)
            betas = est2.params
            effect_size_snp = betas[snp_id]
            P = est2.pvalues[snp_id]
        logp = - np.log(P)
        if logp > 0:
            arr =  np.array(effect_size_snp*(geno_holdout[snp_id]))
            # Check for NaN values
            has_nan = np.isnan(arr).any()

            # Check for Infinite values (positive and negative)
            has_inf = np.isinf(arr).any()
            if not has_nan and not has_inf:
                PRS = PRS +arr
            else:
                pass

    geno_holdout['PRS'] = PRS
    correlation = geno_holdout['phenotype'].corr(geno_holdout['PRS'])
    bottle_array.append(bottle_nr)
    accuracies.append(correlation)

accu = pd.DataFrame(data={"bottleneck":bottle_array, "accuracy":accuracies})
accu = accu.sort_values(by='accuracy')

os.system(f"rm {PATH_bin_bottle}/{pheno_name}_best_bottles.pkl")
accu.to_pickle(PATH_bin_bottle + f"/{pheno_name}_best_bottles.pkl")
