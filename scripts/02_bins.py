# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import pandas as pd
import numpy as np

# Import relevant functions from abyss.glob
from abyss.glob import glob

# Define the name for output files
name = "total_pop"

# Parse command line arguments
args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1]
}

# We supdivided the genome in 32 bins of similar allele frequencies
bin_nr = arguments['bin']

try:
    plink_snps_pool = int(arguments['segment'])
except ValueError:
    plink_snps_pool = 1000  # Default segment size if not provided

PATH_input_raw = glob.PATH_GENO+ "/02_per_chrom/"
name_raw = "total_pop"
PATH_output = glob.PATH_GENO + "/03_macro_similar_AF/"
os.makedirs(PATH_output,exist_ok=True) # make directory for the output files


print(f"Now filling bin nr {bin_nr}")
path_bin = PATH_output + f"/bin_{bin_nr}/"
file_name = [af for af in os.listdir(path_bin) if af.split("_")[0] == "AF"][0]
AFs = pd.read_pickle(path_bin+file_name)
"""
Little script to know how much SNPs we need per group to have more or less #snsps_pool number of snps
"""
num_subframes = len(AFs) // plink_snps_pool
remaining_rows = len(AFs) % plink_snps_pool

try:
    to_divide_in = remaining_rows//num_subframes
    rest = len(AFs)- ((plink_snps_pool + to_divide_in)*num_subframes)
    snps_per_segments = np.ones(num_subframes) * (plink_snps_pool+ to_divide_in)
    to_add = np.concatenate((np.ones(rest), np.zeros(num_subframes - rest)),axis = 0)
    snps_per_segments = snps_per_segments + to_add
except Exception as e:
    snps_per_segments = [len(AFs)]

i = 0
start = 0
end = 0
to_drop = ['IID','PAT','MAT','SEX', 'PHENOTYPE']
for nr_snps in snps_per_segments:
    end = int(end + nr_snps)
    temp = AFs[start:end]
    unique_values = temp['#CHROM'].unique()
    temp_genos_array = []
    for chromys in unique_values:
        print(f"For bin nr {bin_nr} and segment nr {i} processing chrom {chromys}")
        temp_per_chrom = temp[temp["#CHROM"] == chromys]
        snp_ids_to_keep = " ,".join(list(temp_per_chrom['ID']))
        output_file = f"output_{chromys}_bin_{bin_nr}_{i}.txt"
        os.system(f"./plink2 --bfile {PATH_input_raw}{name_raw}_chr_{chromys} --make-bed --out ./temp_{chromys}_bin_{bin_nr}_{i}"+ " > " + output_file + " 2>&1")
        os.system(f"./plink2 --bfile ./temp_{chromys}_bin_{bin_nr}_{i} --snps {snp_ids_to_keep} --mac 50 --geno 0.05  --recode A --out ./temp2_{chromys}_bin_{bin_nr}_{i}"+ " > " + output_file + " 2>&1")
        
        try:
            genos = pd.read_csv(f"./temp2_{chromys}_bin_{bin_nr}_{i}.raw", sep="\t")
            genos = genos.fillna(2)
            genos.set_index('FID', inplace=True)
            genos = genos.drop(to_drop, axis=1).astype(int) - 1 # Encoding it [-1, 0, 1]
            genos = genos.applymap(lambda x :x if x in [-1, 0, 1] else 1)
            genos = genos.T.drop_duplicates(keep='first').T
            temp_genos_array.append(genos)

        except:
            print(f"Problem with {chromys}_bin_{bin_nr}_{i}")
        
        os.system(f"rm temp_{chromys}_bin_{bin_nr}_{i}*")
        os.system(f"rm temp2_{chromys}_bin_{bin_nr}_{i}*")
        os.system(f"rm output_{chromys}_bin_{bin_nr}_{i}.txt")
    try:
        concatenated_df = pd.concat(temp_genos_array, axis=1)
        min_af = np.round(list(temp.head(1).ALT_FREQS)[0],3)
        max_af = np.round(list(temp.tail(1).ALT_FREQS)[0],3)
        concatenated_df.to_pickle(path_bin + f"{i}_maf_{min_af}_{max_af}.pkl")
    except:
        print(f"nothing to concatenate")
    start = int(start + nr_snps)
    i += 1
