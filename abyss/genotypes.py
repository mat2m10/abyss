"""
Helpers for genotype tables coded -1/0/1 (PLINK --recode A minus 1).
"""
import numpy as np


def clean_coded_genotypes(genos):
    """
    Map every value outside {-1, 0, 1} to 1 and drop duplicated SNP columns
    (identical genotype vectors), keeping the first occurrence.

    Same result as the former
        genos.applymap(lambda x: x if x in [-1, 0, 1] else 1)
        genos.T.drop_duplicates(keep='first').T
    but vectorised and without transposing the table twice.
    """
    genos = genos.where(genos.isin([-1, 0, 1]), 1)
    values = np.ascontiguousarray(genos.to_numpy().T)
    if values.shape[0] == 0:
        return genos
    _, first = np.unique(values, axis=0, return_index=True)
    return genos.iloc[:, np.sort(first)]

