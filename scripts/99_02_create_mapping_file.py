import os
import pandas as pd
import sys  # Import Pandas and Numpy to create databases

from abyss.glob import glob

args = sys.argv
arguments = {
    args[1].split(':')[0]: args[1].split(':')[1],
    args[2].split(':')[0]: args[2].split(':')[1],
}

bin = arguments['bin']
PCs = int(arguments['PCs'])


files = []
snps = []
bins = []
path_geno = f"{glob.PATH_GENO}/04_LD_blocks/{PCs}_PCs/bin_{bin}"
for file in os.listdir(path_geno):
    temp_geno = pd.read_pickle(path_geno + "/" + file)
    for snp_name in list(temp_geno.columns):
        snps.append(snp_name)
        files.append(file)
        bins.append(bin)
mapping = pd.DataFrame(data={"snps":snps, "file_name":files, "bin": bin})
mapping = mapping.set_index('snps')
mapping.to_pickle(f"{glob.PATH_usefull}/mapping_bin_{bin}.pkl")
