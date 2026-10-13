import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from os import listdir

import numpy as np
import pandas as pd  # Import Pandas and Numpy to create databases
import matplotlib.pyplot as plt

from abyss.glob import glob
import seaborn as sns
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
path_output = path_output + f"/PRS/{disease}/bin_{bin}"
os.makedirs(path_output, exist_ok=True)

nr_snps = 10000
p_value_to_cut = 0.5
p_max = 10000
p_min = 3
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

sumstat['cumulative_pos'] = pd.concat(cumulative_pos)
sumstat['SNP number'] = sumstat.index

# Define two contrasting colors from the rocket palette
palette = sns.color_palette("rocket", as_cmap=False)
color1 = palette[3]  # First color of the palette
color2 = palette[0] # Last color of the palette, for maximum contrast

# Function to assign color based on chromosome
def assign_color(chrom_number):
    return color1 if chrom_number % 2 == 0 else color2

# Apply the function to create a color column
sumstat['color'] = sumstat['#CHROM'].apply(assign_color)

chrom_midpoints = []
for chrom, group_df in sumstat.groupby('#CHROM'):
    midpoint = group_df['cumulative_pos'].min() + (group_df['cumulative_pos'].max() - group_df['cumulative_pos'].min()) / 2
    chrom_midpoints.append(midpoint)

# Plotting
g = sns.relplot(
    data=sumstat,
    x='cumulative_pos',
    y='-logp',
    aspect=2,
    hue='color',
    palette=[color1, color2],
    legend=False,
    size=3
)


# Set x-axis ticks and labels with reversed chromosome order
chrom_labels = np.sort(sumstat['#CHROM'].unique()) # Reversing the order of chromosomes

# Set the title and x-axis, y-axis labels
g.set(title=f"GWAS Manhattan Plot for {phenotypes_labels[disease]} with {scenario_labels[scenario]}")
g.set(xticks=chrom_midpoints, xticklabels=chrom_labels)
g.set_axis_labels("Chromosome", r"$-\log_{10}(P)$")  # Set y-axis label with subscript

# Set the tick labels at a 45-degree angle
for ax in g.axes.flat:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Optional: Improve the layout
plt.tight_layout()
#plt.savefig(f"{path_output}/sumstat_plot_{scenario}_bin_{bin}_{nr_snps}snps.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{path_output}/sumstat_plot_{scenario}_bin_{bin}_pvals_{p_min}_{p_max}.png", dpi=300, bbox_inches='tight')




path_LDblocks = glob.PATH_GENO + f"/04_LD_blocks/{PCs}_PCs/"

# Initialize PRS (Polygenic Risk Score) to 0
prs = 0
unique_bins = list(sumstat['bin'].unique())
# Loop through unique bins
for unique_bin in unique_bins:
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

        # Load genetic data from the path file
        geno = pd.read_pickle(path_file)

        # Select data from merged_df that matches the current unique path
        temp2_df = merged_df[merged_df['path'] == unique_path]

        # Iterate over rows in temp2_df
        for index, row in temp2_df.iterrows():
            try:
                # Select the genetic data for the current SNP
                genos = geno[[row['snp']]]

                # Scale the genetic data
                scaled_data = scaler.fit_transform(genos)
                genos = pd.DataFrame(scaled_data, columns=genos.columns)

                # Calculate PRS for the current SNP and add it to the overall PRS
                tmp_prs = genos * row.beta_snp
                if not tmp_prs.isna().any().any():
                    prs = prs + tmp_prs.iloc[:, 0]
            except KeyError:
                print(f"SNP {row['snp']} not found in {path_file}")
            except FileNotFoundError:
                print(f"File not found: {path_file}")

path_pheno = glob.PATH_PHENO + f"/{disease}_holdout.pkl"
pheno = pd.read_pickle(path_pheno)

PRS = pd.DataFrame(data={'PRS': prs, 'FID': geno.index})
PRS = pd.merge(PRS, pheno[['FID','phenotype']], on='FID')
correlation = PRS['phenotype'].corr(PRS['PRS'])

#PRS.to_pickle(path_output+f"/prs_{scenario}_corr_{correlation}_{nr_snps}snps.pkl")
PRS.to_pickle(path_output+f"/prs_{scenario}_corr_{correlation}_pvals_{p_min}_{p_max}.pkl")
