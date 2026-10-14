"""
Vectorised per-SNP ordinary least squares.

The GWAS / PRS scripts fit one small OLS model per SNP
(phenotype ~ const + snp [+ snp_corr] [+ PCs]). Doing that with
statsmodels costs ~1 ms of Python overhead per SNP. `ols_many` fits all
of them at once with stacked numpy linear algebra and reproduces exactly
what `sm.OLS(y, sm.add_constant(X)).fit()` returns for params, bse,
tvalues and pvalues (same pinv/SVD solution, same rank rule, same
t-distribution p-values), up to floating point rounding.

Designs that statsmodels would treat differently (a column that is
already a non-zero constant, so add_constant skips the intercept, or
non-finite values) are flagged in `fallback` so the caller can run them
through statsmodels unchanged.
"""
import numpy as np
from scipy import stats


def ols_many(y, M, cols, max_elems=2 ** 24):
    """
    Fit OLS of y on [1, M[:, cols[i]]] for every row i of `cols`.

    y    : (n,) array, shared response
    M    : (n, q) array holding every regressor column
    cols : (m, p) int array, the p columns of M used by model i

    Returns a dict with arrays params/bse/tvalues/pvalues of shape (m, p+1)
    (constant first, like sm.add_constant) and a boolean `fallback` (m,).
    Models flagged in `fallback` are left as NaN and should be fitted with
    statsmodels by the caller.
    """
    y = np.asarray(y, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.intp).reshape(len(cols), -1)
    n = M.shape[0]
    m, p = cols.shape
    k = p + 1
    out = {key: np.full((m, k), np.nan) for key in ("params", "bse", "tvalues", "pvalues")}

    # Cases statsmodels handles differently -> let the caller use statsmodels
    col_finite = np.isfinite(M).all(axis=0)
    col_nonzero_const = (np.ptp(M, axis=0) == 0) & np.any(M != 0.0, axis=0)
    fallback = (~col_finite[cols]).any(axis=1) | col_nonzero_const[cols].any(axis=1)
    if not np.isfinite(y).all():
        fallback[:] = True
    out["fallback"] = fallback

    eps = np.finfo(np.float64).eps
    chunk = max(1, int(max_elems // (n * k)))
    todo = np.flatnonzero(~fallback)
    for start in range(0, len(todo), chunk):
        idx = todo[start:start + chunk]
        A = np.empty((len(idx), n, k))
        A[:, :, 0] = 1.0
        A[:, :, 1:] = np.moveaxis(M[:, cols[idx]], 0, 1)

        # pinv exactly like statsmodels.tools.tools.pinv_extended (rcond=1e-15)
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        cutoff = 1e-15 * s.max(axis=1, keepdims=True)
        s_inv = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)
        pinv = np.matmul(np.swapaxes(vt, 1, 2), s_inv[:, :, None] * np.swapaxes(u, 1, 2))

        # rank like np.linalg.matrix_rank(np.diag(singular_values))
        rank = (s > s.max(axis=1, keepdims=True) * k * eps).sum(axis=1)
        df_resid = (n - rank).astype(np.float64)

        params = pinv @ y
        resid = y[None, :] - np.einsum("mnk,mk->mn", A, params)
        ssr = np.einsum("mn,mn->m", resid, resid)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = ssr / df_resid
            norm_cov_diag = np.einsum("mkn,mkn->mk", pinv, pinv)
            bse = np.sqrt(norm_cov_diag * scale[:, None])
            tvalues = params / bse
        pvalues = stats.t.sf(np.abs(tvalues), df_resid[:, None]) * 2

        out["params"][idx] = params
        out["bse"][idx] = bse
        out["tvalues"][idx] = tvalues
        out["pvalues"][idx] = pvalues
    return out
