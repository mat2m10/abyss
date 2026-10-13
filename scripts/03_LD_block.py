import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from k_means_constrained import KMeansConstrained
from abyss.glob import glob
import random
from math import floor
from collections import Counter
# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1],
    args[3].split(':')[0]: args[3].split(':')[1],
}

bin = arguments['bin']

# Set default values if arguments are not provided
try:
    n_components = int(arguments['PCs'])
except ValueError:
    n_components = 15

try:
    sample_size = int(arguments['size'])
except ValueError:
    sample_size = 300

# Define paths and directories
path_geno = glob.PATH_GENO
size_min = sample_size - round(sample_size / 5)
size_max = sample_size + round(sample_size / 5)
PATH_input = f"{path_geno}/03_macro_similar_AF/bin_{bin}/"
PATH_output_0 = f"{path_geno}/04_LD_blocks/"
PATH_output_1 = f"{path_geno}/04_LD_blocks/{n_components}_PCs"
PATH_output = f"{path_geno}/04_LD_blocks/{n_components}_PCs/bin_{bin}/"

# Create output directories if they don't exist
os.makedirs(PATH_output_0, exist_ok=True)
os.makedirs(PATH_output_1, exist_ok=True)
os.makedirs(PATH_output, exist_ok=True)

path_af = PATH_input + '/'+[f for f in os.listdir(PATH_input) if f.startswith("AF")][0]
AFs = pd.read_pickle(path_af)
AFs["BIN"] = bin
AFs['ID'] = AFs['ID'].astype(str)


ids = []
paths = []
# Get a list of output numbers to avoid processing duplicates
output_numbers = [f.split('_')[0] for f in os.listdir(PATH_output)]

# Process LD files
for LDs in [f for f in os.listdir(PATH_input) if f.split('_')[0] not in output_numbers and not f.startswith("AF")]:
    print(LDs)
    try:
        clus_nr = LDs.split('.pkl')[0].split('_')[0]
        # Load genotypic data
        genos = pd.read_pickle(f"{PATH_input}/{LDs}")
        genos = genos.T

        # Calculate the number of clusters based on sample size
        num_clus = round(genos.shape[0] / sample_size)
        features = genos.columns

        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(genos)

        # Perform PCA with n components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_scaled)

        # Create a new DataFrame to store the principal components
        pc_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(data=principal_components, columns=pc_columns)

        # Apply constrained K-Means clustering
        clf = KMeansConstrained(
            n_clusters=num_clus,
            size_min=size_min,
            size_max=size_max,
            random_state=0
        )
        clf.fit_predict(np.array(df_pca))

        # Assign cluster labels to genotypic data
        count = Counter(clf.labels_)
        empty = []
        sample_size_temp = floor(len(clf.labels_)/sample_size)
        for val in count.values():
            multi = floor(val/sample_size_temp)+1
            list_to_sample = multi*list(range(sample_size_temp))
            random.shuffle(list_to_sample)
            empty = empty + list_to_sample[0:val]
        genos['clusters_k_means'] = empty
        df_pca['clusters_k_means'] = empty
        genos['clusters_k_means'] = clf.labels_

        # Process and save clusters separately
        for num in genos.clusters_k_means.unique():
            to_save = genos.loc[genos['clusters_k_means'] == num]
            to_save = to_save.drop(columns=['clusters_k_means']).T

            # Calculate minimum and maximum MAF for the cluster
            tot_mafs = []
            for snp_id in to_save:
                ids.append(snp_id)
                paths.append(f"{clus_nr}_{num}")
                try:
                    try:
                        num_maj = to_save[[snp_id]].value_counts()[1]
                    except Exception as e:
                        num_maj = 0
                    try:
                        num_het = to_save[[snp_id]].value_counts()[0]
                    except Exception as e:
                        num_het = 0
                    try:
                        num_min = to_save[[snp_id]].value_counts()[-1]
                    except Exception as e:
                        num_min = 0
                    total_humans = num_maj + num_het + num_min
                    maf = (num_min*2 + num_het)/(total_humans*2)

                except Exception as e:
                    print(e)
                    print(f"snp {snp_id} has a problem")

                tot_mafs.append(maf)

            max_maf = np.round(max(tot_mafs), 5)
            min_maf = np.round(min(tot_mafs), 5)
            size = to_save.shape[1]

            # Save the processed cluster data
            to_save.to_pickle(f"{PATH_output}{clus_nr}_{num}_{size}_maf_{min_maf}_{max_maf}.pkl")
            print(f"Done subgroup {num} for bin {bin}")
    except:
        print(f"Problem with {LDs}")

path_df = pd.DataFrame(data={"ID": ids, "path": paths})
print(path_df)
# Convert the 'ID' column to string type before splitting
path_df['ID'] = path_df['ID'].astype(str)
path_df[['Chromosome', 'Position', 'Alls2']] = path_df['ID'].str.split(':', expand=True)
path_df = path_df.drop(['ID'], axis=1)

# Ensure the 'ID' column in AFs DataFrame is also string type before splitting
AFs['ID'] = AFs['ID'].astype(str)
AFs[['Chromosome', 'Position', 'Alls']] = AFs['ID'].str.split(':', expand=True)

# Merge the DataFrames
merged_df = pd.merge(path_df, AFs, on=['Chromosome', 'Position'], how="inner")
merged_df.to_pickle(PATH_output+"/mapping.pkl")
