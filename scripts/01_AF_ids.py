# Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from abyss.glob import glob
import pandas as pd

# Define paths
PATH_input = glob.PATH_GENO+ "/02_per_chrom/"
PATH_output = glob.PATH_GENO + "/03_macro_similar_AF/"
PATH_usefull = glob.PATH_usefull
name = "total_pop"
os.makedirs(PATH_output,exist_ok=True) # make directory for the output files
os.makedirs(PATH_usefull,exist_ok=True) # make directory for the output files

# Iterate over the existing chromosome files (here total_pop_chrom_#chrom)
elements = [f.split("_chr_")[1].split(".")[0] for f in os.listdir(PATH_input)]
# Convert the list to a set to remove duplicates
unique_elements = set(elements)

# Convert the set back to a list if needed
unique_elements_list = list(unique_elements)

# Finding all the allele frequencies over all the chromosomes
AF_array = []
for chrom in unique_elements_list:
    print(f"Finding allele frequencies of chromosome {chrom}")
    output_file = "output.txt"
    os.system(f"./plink2 --bfile {PATH_input}{name}_chr_{chrom} --make-bed --out ./temp_{chrom}"+ " > " + output_file + " 2>&1")
    os.system(f"./plink2 --bfile ./temp_{chrom} --freq --out ./AF_{chrom}"+ " > " + output_file + " 2>&1")
    AFs_meta = pd.read_csv(f"./AF_{chrom}.afreq", sep="\t") # convert to pandas dataframe
    AF_array.append(AFs_meta)
    os.system(f"rm AF_{chrom}*")
    os.system(f"rm temp_{chrom}*")
    os.system(f"rm output.txt")

# Concatenate all dataframes in AF_array into one
AF_df = pd.concat(AF_array, ignore_index=True)
AF_df = AF_df.sort_values(by='ALT_FREQS')
AF_df.to_pickle(f"{PATH_usefull}/AF.pkl")

# Creating 32 bins where all the snps corresponding to the different allele frequency bins will go

# Define the number of segments
num_segments = 32

# Calculate the segment length
segment_length = 0.5 / num_segments

# Initialize an empty list to store the segments
segments = []

# Create the segments
for i in range(num_segments):
    start = i * segment_length
    end = (i + 1) * segment_length
    segment = [start, end, i+1]
    segments.append(segment)

# Print the list of segments
for segment in segments:
    AFs = AF_df[AF_df['ALT_FREQS'] > segment[0]]
    AFs = AFs[AFs['ALT_FREQS'] < segment[1]]
    path_bin = PATH_output + f"bin_{segment[2]}"
    os.makedirs(path_bin, exist_ok=True)
    AFs.to_pickle(path_bin+ f"/AF_{segment[0]}_{segment[1]}.pkl")

# Putting the IDs of our samples in a usefull folder
PATH_input_raw = glob.PATH_GENO+ "/02_per_chrom/"
name_raw = "total_pop"
chrom = 22
output_file = "output.txt"
os.system(f"./plink2 --bfile {PATH_input}{name}_chr_{chrom} --maf 0 --max-maf 0.01 --mac 50 --geno 0.05 --recode A --out ./plink"+ " > " + output_file + " 2>&1")
id_samples = pd.read_csv(f"./plink.raw", sep="\t")[['FID','IID','PAT','MAT','SEX','PHENOTYPE']]
os.system(f"rm plink.raw")
os.system(f"rm plink.log")
os.system(f"rm output.txt")

id_samples.to_pickle(f"{PATH_usefull}/id_samples.pkl")