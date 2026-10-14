import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd  # Import Pandas and Numpy to create databases

from abyss.glob import glob
from abyss.stats import ols_many
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        print(dup)
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df


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
bin = arguments['bin']
pheno_name = arguments['pheno']
id_samples = pd.read_pickle(f"{glob.PATH_usefull}/id_samples.pkl")

try:
    PCs = int(arguments['PCs'])
except ValueError:
    PCs = 15

try:
    latent_dim = int(arguments['bottle_latent'])
except ValueError:
    latent_dim = 15

try:
    covname = arguments['covs']
    path_cov = f"{glob.PATH_COVS}/{covname}.pkl"
    covs = pd.read_pickle(path_cov)
    covs['FID'] = id_samples['FID']
except:
    pass

# Possibility to choose between 4 scenarios: "cov" "abyss" "abyss_cov" "no_cov"
try:
    scenario = arguments['scenario']
except:
    scenario = "abyss"

pheno = pd.read_pickle(f"{glob.PATH_PHENO}/{pheno_name}.pkl")


path_dims = f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{latent_dim}_dim_latent/"
existing_dims = [f for f in os.listdir(path_dims) if f.startswith("reduction")]
try:
    pheno_name = arguments['pheno']
    accu = pd.read_pickle(f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{pheno_name}_best_bottles.pkl")
    dims = int(accu.sort_values(by='accuracy', ascending=False)[['bottleneck']].iloc[0, 0])
except Exception as e:
    print(e)
    path_dims = f"{glob.PATH_GENO}/07_lowest_dimensionality/bin_{bin}/{latent_dim}_dim_latent/"
    dims = int(existing_dims[-1].split("_")[3])

print(dims)

path_output_GWAS =  glob.PATH_GENO + f"/09_summary_statistics/{pheno_name}/"
os.makedirs(path_output_GWAS, exist_ok=True)

path_output = path_output_GWAS + scenario
os.makedirs(path_output, exist_ok=True)
path_input_maf_prob = glob.PATH_GENO + f"/08_MAF_probability_landscape/{pheno_name}/bin_{bin}/{dims}_bottleneck_dims/"
path_input_geno = glob.PATH_GENO + f"/04_LD_blocks/{PCs}_PCs/bin_{bin}/"

if scenario == 'abyss' or scenario == 'abyss_cov':
    input_files = [f for f in os.listdir(path_input_maf_prob) if not f.endswith("metric_plots")]
else:
    input_files= [f for f in os.listdir(path_input_geno) if not f.startswith("mapping")]

# Initialize lists to store results
snps = []
consts = []
effect_size_snps = []
effect_size_mafs = []
std_errs = []
ts = []
Ps = []
all_betas = []
# Loop through each file in the input MAF probability dataset
for abyss_file in input_files:
    # Construct file paths for the MAF and genotype data
    path_maf_file = path_input_maf_prob + abyss_file
    path_geno_file = path_input_geno + abyss_file

    # Load genotype data and scale it
    genos = pd.read_pickle(path_geno_file)
    ids = genos.index
    scaled_data = scaler.fit_transform(genos)
    genos = pd.DataFrame(scaled_data, columns=genos.columns)
    genos = rename_duplicates(genos)  # Handling duplicate column names
    genos['FID'] = ids
    genos.set_index('FID', inplace=True)
    # Merge genotype data with phenotype data
    total = pd.merge(genos, pheno[['FID','phenotype']], on='FID')

    if scenario == 'cov' or scenario == 'abyss_cov':
        covs['FID'] = ids
        if scenario == 'abyss_cov':
            n_first_columns = 5
            covs = covs.iloc[:, :n_first_columns]
            covs['FID'] = ids

        total = pd.merge(total, covs, on='FID')


    if scenario == 'abyss' or scenario == 'abyss_cov':
        # Load MAF probability data and rename columns
        mafprob = pd.read_pickle(path_maf_file)
        mafprob.columns = [col + '_corr' for col in mafprob.columns]
        # Merge MAF probability data with the FID column
        total = pd.merge(total, mafprob, on='FID')

    filtered_columns = [col for col in total.columns 
                    if not col.startswith('PC') 
                    and not col.endswith('_corr') 
                    and col not in ['phenotype', 'FID']]
    PCs = [col for col in total.columns
                if col.startswith('PC')]

    y = total[['phenotype']]

    designs = []
    for snp_id in filtered_columns:
        if scenario == 'abyss':
            snp_columns = [snp_id, f"{snp_id}_corr"]
        if scenario == 'abyss_cov':
            snp_columns = [snp_id, f"{snp_id}_corr"] + PCs
        if scenario == 'cov':
            snp_columns = [snp_id] + PCs
        if scenario == 'no_cov':
            snp_columns = [snp_id]
        designs.append(snp_columns)

    # Fit all per-SNP OLS models at once (same results as sm.OLS per SNP).
    # Designs with missing/duplicated columns, or that statsmodels treats
    # specially, are fitted one by one with statsmodels as before.
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
                fast[i] = (res['params'][row], res['bse'][row], res['tvalues'][row], res['pvalues'][row])

    for i, (snp_id, snp_columns) in enumerate(zip(filtered_columns, designs)):
        if i in fast:
            params, bse, tvalues, pvalues = fast[i]
            names = ['const'] + snp_columns
            betas = pd.Series(params, index=names)
            consts.append(params[0])
            effect_size_snps.append(params[1])
            effect_size_mafs.append(params[2] if f"{snp_id}_corr" in snp_columns else 0)
            std_errs.append(bse[1])
            ts.append(tvalues[1])
            Ps.append(pvalues[1])
            snps.append(snp_id)
            all_betas.append(betas)
            continue

        X = total[snp_columns]
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()
        # Extract coefficients (betas)
        betas = est2.params
        try:
            consts.append(betas['const'])
        except:
            consts.append(0)
        effect_size_snps.append(betas[snp_id])
        try:
            effect_size_mafs.append(betas[f"{snp_id}_corr"])
        except:
            effect_size_mafs.append(0)

        std_errs.append(est2.bse[snp_id])
        ts.append(est2.tvalues[snp_id])

        P = est2.pvalues[snp_id]
        Ps.append(P)
        snps.append(snp_id)
        all_betas.append(est2.params)

    sumstat = pd.DataFrame(data={
        'snp': snps,
        'consts': consts,
        'beta_snp': effect_size_snps,
        'beta_maf': effect_size_mafs,
        'std_err': std_errs,
        'Z': ts,
        'P': Ps,
        'all_betas':all_betas
        })

    try:
        split_data = sumstat['snp'].str.split(':', expand=True)
        chromosome = split_data[0]
        position = split_data[1]
        sumstat['#CHROM'] = chromosome.astype(int)
        sumstat['POS'] = position.astype(int)
        sumstat['POS'] = sumstat['POS'].astype(int)
        sumstat['-logp'] = - np.log(sumstat['P'])
    except:
        print("there is something weird")
    
    sumstat.to_pickle(path_output+f"/bin_{bin}_sumstat.pkl")
