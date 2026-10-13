import os
import pandas as pd

import pandas as pd  # Import Pandas and Numpy to create databases

from abyss.glob import glob

PATH_PHENO = glob.PATH_PHENO

for famfile in os.listdir(PATH_PHENO):
    path_pheno = PATH_PHENO + "/" +famfile
    pheno_name = path_pheno.split('/')[-1].split('_')[0]
    column_names = ['FID', 'IID', '0', '1', '2', '3']
    pheno = pd.read_csv(path_pheno, delim_whitespace=True, header=None, names=column_names)
    # Select only the specified columns
    pheno_subset = pheno[['FID', 'IID', '3']]

    # Rename column '3' to 'phenotype'
    pheno_subset.rename(columns={'3': 'phenotype'}, inplace=True)
    pheno_subset.to_pickle(PATH_PHENO + "/" + pheno_name +".pkl")