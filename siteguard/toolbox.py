"""SiteGuard-CP conformal toolbox (same code as notebook cell 2b).

Split / Mondrian / weighted conformal thresholds, label-free density ratios,
ESS-constrained tempering, EM prior estimation and set evaluation.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

# ---------- 2b. Conformal toolbox (no data needed; run this before the replay cells in a fresh session) ----------
def lac(P):
    """LAC nonconformity: s(x, y) = 1 - p_y(x). Returns an (n, K) score matrix."""
    return 1.0 - P

def threshold(cal_s, alpha, n_test, cal_w=None, test_w=None):
    """Split-conformal threshold per test point. Unweighted: ceil((n+1)(1-a))-th score.
    Weighted (Tibshirani et al. 2019): smallest s with sum_{s_i<=s} w_i >= (1-a)(sum w + w_test)."""
    if len(cal_s) == 0:
        return np.full(n_test, np.inf)
    order = np.argsort(cal_s); s = cal_s[order]
    if cal_w is None:
        k = int(np.ceil((len(s) + 1) * (1 - alpha)))
        return np.full(n_test, np.inf if k > len(s) else s[k - 1])
    cw = np.cumsum(cal_w[order]); target = (1 - alpha) * (cw[-1] + test_w)
    idx = np.searchsorted(cw, target, side='left')
    return np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)

def ess_frac(w):
    return (w.sum() ** 2) / (np.square(w).sum() * len(w))

def temper(r_cal, r_tgt, ess_min):
    """Largest gamma in [0,1] with ESS(r_cal**gamma) >= ess_min. gamma=0 -> plain conformal."""
    for g in np.linspace(1, 0, 21):
        if ess_frac(r_cal ** g) >= ess_min:
            return g, r_cal ** g, r_tgt ** g
    return 0.0, np.ones_like(r_cal), np.ones_like(r_tgt)

def density_ratio(Z_cal, Z_tgt, seed, clip):
    """Cross-fitted domain classifier calibration-vs-target. Returns r(x)=p_T(x)/p_S(x) up to a constant, and AUC."""
    Xd = np.vstack([Z_cal, Z_tgt]); d = np.r_[np.zeros(len(Z_cal)), np.ones(len(Z_tgt))]
    n_splits = int(min(5, len(Z_tgt), len(Z_cal)))
    clf = LogisticRegression(C=0.1, max_iter=3000)
    p = cross_val_predict(clf, StandardScaler().fit_transform(Xd), d,
                          cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed), method='predict_proba')[:, 1]
    auc = roc_auc_score(d, p); p = np.clip(p, 1e-4, 1 - 1e-4)
    r = np.clip(p / (1 - p), 1 / clip, clip); r = r / r[:len(Z_cal)].mean()
    return r[:len(Z_cal)], r[len(Z_cal):], auc

def em_prior(P_tgt, pi_src, iters=100):
    """Saerens-Latinne-Decaestecker EM: target class prior from unlabeled target posteriors."""
    pi = pi_src.copy()
    for _ in range(iters):
        q = P_tgt * (pi / pi_src); q /= q.sum(1, keepdims=True)
        new = np.clip(q.mean(0), 0.05, 0.95); new /= new.sum()
        if np.abs(new - pi).max() < 1e-6: break
        pi = new
    return pi

def hybrid_factor(P, pi_src, pi_tgt):
    """1/Z(x), Z(x)=sum_j p_S(j|x) pi_T(j)/pi_S(j). Turns covariate-shift weights into class-conditional
    weights p_T(x|k)/p_S(x|k) up to a per-class constant, exact under pure label shift."""
    return 1.0 / (P * (pi_tgt / pi_src)).sum(1)

def sets_from(S_test, thr):          # thr: (n_test, K)
    return S_test <= thr

def evaluate(sets, y):
    n = len(y); size = sets.sum(1); cover = sets[np.arange(n), y]
    single = size == 1; pred = sets.argmax(1)
    out = dict(coverage=cover.mean(), set_size=size.mean(), singleton=single.mean(), empty_set=(size == 0).mean(),
               refer=(size >= 2).mean(),
               single_acc=(pred[single] == y[single]).mean() if single.any() else np.nan)
    for k in (0, 1):
        out[f'cov_class{k}'] = cover[y == k].mean() if (y == k).any() else np.nan
    return out
