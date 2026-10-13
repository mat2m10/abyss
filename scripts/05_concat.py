import os
import sys
from os import listdir
import pandas as pd
from abyss.glob import glob

# Get the PATH_GENO from the abyss.glob module
PATH_GENO = glob.PATH_GENO

# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1]
}
# Set default values if arguments are not provided
try:
    bottle = int(arguments['bottle'])
except ValueError:
    bottle = 15

bin = arguments['bin']

# Define input and output paths
PATH_input = f"{PATH_GENO}/05_Latent_populations/{bottle}_bottleneck_dims/bin_{bin}"
path_output = f"{PATH_GENO}/06_Concated_latent/{bottle}_bottleneck_dims/"
os.makedirs(path_output, exist_ok=True)

latent_dfs = []
for latent in os.listdir(PATH_input):
    new_path = f"{PATH_input}/{latent}"
    latent_df = pd.read_pickle(new_path)
    latent_dfs.append(latent_df)

concated = pd.concat(latent_dfs, axis = 1)
number_of_cols = len(concated.columns)

# Creating new column names
new_column_names = [f'dim{i+1}' for i in range(number_of_cols)]

# Renaming the columns
concated.columns = new_column_names
concated.to_pickle(path_output + f"/concatted_bin_{bin}.pkl")