import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

import numpy as np
import pandas as pd  # Import Pandas and Numpy to create databases

from abyss.glob import glob
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1],
    args[3].split(':')[0]: args[3].split(':')[1],
    args[4].split(':')[0]: args[4].split(':')[1],
}



disease = arguments['pheno']
scenario = arguments['scenario']

try:
    PCs = int(arguments['PCs'])
except:
    PCs = 5

def filter_bins(bin_list, bin_param):
    if bin_param in bin_list:
        return [bin_param]
    else:
        return bin_list

path_input = glob.PATH_GENO + f"/09_summary_statistics/{disease}/{scenario}"
list_of_bins = os.listdir(path_input)
try:
    bin = arguments['bin']
    if bin == 0:
        bin = "total"
    else:
        bin_param = f"bin_{bin}_sumstat.pkl"
        list_of_bins = filter_bins(list_of_bins, bin_param)
except:
    bin = "total"

phenotypes_labels = {
    "bc": "Breast Cancer",
    "cad": "Coronary Artery Disease",
    "ms": "Multiple Sclerosis",
    "t2d": "Type 2 Diabetes",
    "pc": "Prostate Cancer",
    "ibd": "Inflammatory Bowel Disease"
}

scenario_labels = {
    "no_cov": "no covariates",
    "abyss": "Abyss correction",
    "cov": "PCA correction",
    "abyss_cov": "Abyss + PCA as covariates"
}

# Set default values if arguments are not provided
path_output = glob.PATH_PHENO
path_output = path_output + f"/SNP_from_PRS/{disease}/"
os.makedirs(path_output, exist_ok=True)

p_max = 500
p_min = 7.5
list_of_dfs = []
for biny in list_of_bins:
    try:
        temp = pd.read_pickle(path_input+f"/{biny}")
        temp['bin'] = biny.split('_')[1]
        
        # Sort by -logp in descending order and take the top SNPs
        #top_snps = temp.sort_values(by='-logp', ascending=False).head(nr_snps)
        #top_snps = temp[temp['-logp'] > p_value_to_cut]
        top_snps = temp[temp['-logp'] < p_max]
        top_snps = temp[temp['-logp'] > p_min]
        
        list_of_dfs.append(top_snps)
    except Exception as e:
        print(e)
        print(f"Problem with chrom {biny}")

# Concatenating the top SNPs from each bin
sumstat = pd.concat(list_of_dfs)

# Sorting by chromosome if needed
sumstat = sumstat.sort_values(by='#CHROM', ascending=True)
sumstat.reset_index(drop=True, inplace=True)


running_pos = 0
cumulative_pos = []
for chrom, group_df in sumstat.groupby('#CHROM'):
    cumulative_pos.append(group_df['POS'] + running_pos)
    running_pos += group_df['POS'].max()


path_LDblocks = glob.PATH_GENO + f"/04_LD_blocks/{PCs}_PCs/"
path_mafs = glob.PATH_GENO + f"/08_MAF_probability_landscape/{disease}/"

# Initialize PRS (Polygenic Risk Score) to 0
prs = 0
unique_bins = list(sumstat['bin'].unique())
# Loop through unique bins



for unique_bin in unique_bins:
    genos = []
    mafs = []
    # Select data from sumstat that matches the current unique bin
    temp_df = sumstat[sumstat['bin'] == unique_bin]

    # Construct the path to load mapping data
    path_mapping = path_LDblocks + f"bin_{unique_bin}/mapping.pkl"
    mapp = pd.read_pickle(path_mapping)
    mapp.rename(columns={'Position': 'POS'}, inplace=True)
    mapp['POS'] = mapp['POS'].astype(int)
    mapp['ALT_FREQS'] = mapp['ALT_FREQS'].astype(float)

    # Merge data from temp_df and mapping data based on 'POS'
    merged_df = pd.merge(temp_df, mapp[['POS','ALT_FREQS','REF','ALT','path']], on='POS', how='inner')

    # Extract unique paths from merged data
    unique_paths = list(merged_df['path'].unique())

    # Loop through unique paths
    for unique_path in unique_paths:
        # Find the file with SNP information for the current unique path
        file_with_snp = [f for f in os.listdir(path_LDblocks + f"bin_{unique_bin}/") if f.startswith(unique_path)][0]
        path_file = path_LDblocks + f"bin_{unique_bin}/" + file_with_snp
        path_maf_file = path_mafs + f"bin_{unique_bin}/"
        bottle_of_maf = os.listdir(path_maf_file)[0]
        path_maf_file = path_maf_file + f"{bottle_of_maf}/" + file_with_snp

        # Load genetic data from the path file
        geno = pd.read_pickle(path_file)
        maf = pd.read_pickle(path_maf_file)
        # Select data from merged_df that matches the current unique path
        temp2_df = merged_df[merged_df['path'] == unique_path]

        # Iterate over rows in temp2_df
        for index, row in temp2_df.iterrows():
            try:
                # Select the genetic data for the current SNP
                genos.append(geno[[row['snp']]])
                mafs.append(maf[[row['snp']]])
            except KeyError:
                print(f"SNP {row['snp']} not found in {path_file}")
            except FileNotFoundError:
                print(f"File not found: {path_file}")

    # Concatenating the top SNPs from each bin
    geno_for_PRS = pd.concat(genos, axis =1)
    geno_for_PRS['FID'] = list(geno.index)
    geno_for_PRS.set_index('FID', inplace=True)
    maf_for_PRS = pd.concat(mafs, axis =1)
    maf_for_PRS['FID'] = list(geno.index)
    maf_for_PRS.set_index('FID', inplace=True)
    path_final = path_output + f"/bin_{unique_bin}/"
    os.makedirs(path_final, exist_ok=True)

    geno_for_PRS.to_pickle(path_final+f"/snp_prs_{scenario}_pvals_{p_min}_{p_max}.pkl")
    maf_for_PRS.to_pickle(path_final+f"/mafs_prs_{scenario}_pvals_{p_min}_{p_max}.pkl")
