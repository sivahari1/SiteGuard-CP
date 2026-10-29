"""Reproduce the paper's equal-set-size analysis, leave-site-out margin and figures.

Reads the cache written by the notebook's LOSO loop (siteguard_cache.pkl) and
per_site_results.csv, then writes:
  fine_grid_per_site.csv        coverage / set size per site on an alpha grid
  paper_table2_extra.csv        Table 2 rows: matched Mondrian, WM-CP variants, LSO margin
  paper_equal_size_tests.csv    equal-set-size comparison vs Mondrian (Sect. 5.2)
  paper_label_shift_baseline.csv label-shift CP baseline (Podkopaev & Ramdas 2021), Table 2
  hybrid_prior_em_error.csv     per-site EM prevalence error vs hybrid coverage loss
  figures/per_site_coverage.pdf, frontier.pdf, prior_error.pdf

Usage:
  python scripts/paper_analysis.py --results path/to/siteguard_results
Runs in about a minute on a CPU. The cache is a pickle written by your own
notebook run; only load caches you produced yourself.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from siteguard.toolbox import density_ratio, em_prior, lac, temper, threshold  # noqa: E402

CFG = dict(ALPHA=0.10, ESS_MIN=0.20, W_CLIP=20.0)
ALPHAS = np.round(np.arange(0.02, 0.2501, 0.0025), 4)


def weights_for(e, domain):
    """Tempered density-ratio weights from unlabeled target scans ('pca' or 'score' space)."""
    if domain == 'pca':
        zc, zt = e['Zc'].astype(float), e['Zt'].astype(float)
    else:
        lg = lambda p: np.log(p[:, 1:2] + 1e-9) - np.log(p[:, 0:1] + 1e-9)
        zc, zt = lg(e['Pcal']), lg(e['Pte'])
    r_cal, r_te, _ = density_ratio(zc, zt, e['seed'], CFG['W_CLIP'])
    _, wc, wt = temper(r_cal, r_te, CFG['ESS_MIN'])
    return wc, wt


def fine_grid(cache):
    rows = []
    for e in cache:
        yc, yt = e['ycal'], e['yte']
        st = lac(e['Pcal'])[np.arange(len(yc)), yc]
        ste = lac(e['Pte'])
        n = len(yt)
        w = {'mondrian': None, 'pca': weights_for(e, 'pca'), 'score': weights_for(e, 'score')}
        for a in ALPHAS:
            for v, wv in w.items():
                t = np.zeros((n, 2))
                for k in (0, 1):
                    m = yc == k
                    t[:, k] = threshold(st[m], a, n) if wv is None else threshold(st[m], a, n, wv[0][m], wv[1])
                s = ste <= t
                sz = s.sum(1)
                cov = s[np.arange(n), yt]
                sacc = (s.argmax(1) == yt)[sz == 1].mean() if (sz == 1).any() else np.nan
                rows.append((e['site'], e['rep'], a, v, cov.mean(), sz.mean(), (sz == 1).mean(), sacc))
    g = pd.DataFrame(rows, columns=['site', 'rep', 'alpha', 'v', 'cov', 'size', 'single', 'sacc'])
    return g.groupby(['v', 'alpha', 'site']).mean(numeric_only=True).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True, help='folder with siteguard_cache.pkl and per_site_results.csv')
    ap.add_argument('--out', default=None, help='output folder (default: --results)')
    args = ap.parse_args()
    out = args.out or args.results
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    with open(os.path.join(args.results, 'siteguard_cache.pkl'), 'rb') as f:
        cache = pickle.load(f)['cache']
    print(len(cache), 'cached site x split entries')

    G = fine_grid(cache)
    G.to_csv(os.path.join(out, 'fine_grid_per_site.csv'), index=False)
    mg = G[G.v == 'mondrian']
    piv = {c: mg.pivot(index='alpha', columns='site', values=c) for c in ['cov', 'size', 'single', 'sacc']}
    A = piv['cov'].index.values
    msz = piv['size'].mean(axis=1).values

    def mondrian_at(alpha):
        return pd.DataFrame({c: pd.Series({s: np.interp(alpha, A, piv[c][s].values) for s in piv[c].columns})
                             for c in piv})

    def stats(name, d, ref=None):
        c = d['cov']
        sf = np.clip(0.90 - c, 0, None)
        o = dict(method=name, mean_cov=c.mean(), worst=c.min(), below_085=int((c < 0.85).sum()), shortfall=sf.mean(),
                 set_size=d['size'].mean(), singleton=d['single'].mean(), single_acc=d['sacc'].mean())
        if ref is not None:
            o['p_vs_mondrian'] = wilcoxon(sf, np.clip(0.90 - ref['cov'], 0, None),
                                          alternative='less', zero_method='zsplit').pvalue
        return o

    m10 = mondrian_at(0.10)
    tab = [stats('Mondrian a=0.10', m10)]
    rng = np.random.default_rng(0)
    eq = []
    for v in ('score', 'pca'):
        for a in (0.05, 0.075, 0.10, 0.125, 0.15, 0.20):
            g = G[(G.v == v) & np.isclose(G.alpha, a)].set_index('site')
            am = np.interp(g['size'].mean(), msz[::-1], A[::-1])
            mm = mondrian_at(am).loc[g.index]
            if np.isclose(a, 0.10):
                tab.append(stats(f'WM-CP {v} a=0.10', g, m10))
                tab.append(stats(f'Mondrian a={am:.3f} (matched to {v}; uses test labels)', mm))
            d = g['cov'] - mm['cov']
            try:
                p = wilcoxon(g['cov'], mm['cov'], alternative='greater').pvalue
            except ValueError:
                p = np.nan
            boot = [d[rng.choice(g.index.values, len(g), replace=True)].mean() for _ in range(4000)]
            eq.append(dict(variant=v, alpha=a, set_size=g['size'].mean(), matched_alpha=am, mean_cov_gain=d.mean(),
                           sites_better=int((d > 0).sum()), p_wilcoxon=p,
                           ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975)))

    rules = {'shortfall<=0.005 (main)': lambda o: np.clip(0.90 - o, 0, None).mean(axis=1) <= 0.005,
             'no other site <0.85': lambda o: o.min(axis=1) >= 0.85,
             '10th pct site >=0.875': lambda o: o.quantile(0.10, axis=1) >= 0.875}
    for rn, rule in rules.items():
        pick = {}
        for s in piv['cov'].columns:
            ok = rule(piv['cov'].drop(columns=s)).values
            pick[s] = A[ok].max() if ok.any() else A.min()
        d = pd.DataFrame({c: pd.Series({s: piv[c].loc[pick[s], s] for s in pick}) for c in piv})
        r = stats(f'LSO margin, {rn}', d, m10)
        r['alpha_range'] = f'{min(pick.values()):.4f}-{max(pick.values()):.4f}'
        tab.append(r)
    tab = pd.DataFrame(tab).set_index('method').round(4)
    eq = pd.DataFrame(eq).round(4)
    tab.to_csv(os.path.join(out, 'paper_table2_extra.csv'))
    eq.to_csv(os.path.join(out, 'paper_equal_size_tests.csv'), index=False)
    pd.set_option('display.width', 200)
    print(tab.to_string())
    print(eq.to_string(index=False))
    g10 = G[(G.v == 'score') & np.isclose(G.alpha, 0.10)].set_index('site')
    rho = spearmanr(g10['size'] - m10.loc[g10.index, 'size'], 0.90 - m10.loc[g10.index, 'cov'])
    print('Adaptivity: Spearman(extra set size of score WM-CP, Mondrian coverage gap) =', rho)

    # ---------------- label-shift CP baseline (Podkopaev & Ramdas 2021) ----------------
    def bbse_prior(pcal, ycal, pte, ps):
        cm = np.stack([pcal[ycal == j].mean(0) * ps[j] for j in (0, 1)], axis=1)
        w = np.linalg.solve(cm, pte.mean(0))
        pi = np.clip(w * ps, 0.05, 0.95)
        return pi / pi.sum()
    ls_rows = []
    for e in cache:
        yc, yt, pc, pt_ = e['ycal'], e['yte'], e['Pcal'], e['Pte']
        st = lac(pc)[np.arange(len(yc)), yc]
        ste = lac(pt_)
        n = len(yt)
        ps = np.bincount(yc, minlength=2) / len(yc)
        for name, pi_t in (('EM', em_prior(pt_, ps)), ('BBSE', bbse_prior(pc, yc, pt_, ps))):
            wcal = (pi_t / ps)[yc]
            t = np.zeros((n, 2))
            for k in (0, 1):
                t[:, k] = threshold(st, 0.10, n, wcal, np.full(n, pi_t[k] / ps[k]))
            s = ste <= t
            sz = s.sum(1)
            ls_rows.append(dict(site=e['site'], prior=name, cov=s[np.arange(n), yt].mean(), size=sz.mean(),
                                single=(sz == 1).mean(),
                                sacc=(s.argmax(1) == yt)[sz == 1].mean() if (sz == 1).any() else np.nan))
    LS = pd.DataFrame(ls_rows).groupby(['prior', 'site']).mean(numeric_only=True).reset_index()
    ls_tab = [stats(f'Label-shift CP ({pr})', g.set_index('site'), m10) for pr, g in LS.groupby('prior')]
    ls_tab = pd.DataFrame(ls_tab).set_index('method').round(4)
    ls_tab.to_csv(os.path.join(out, 'paper_label_shift_baseline.csv'))
    print(ls_tab.to_string())

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fdir = os.path.join(out, 'figures')

    fr = G.groupby(['v', 'alpha']).agg(size=('size', 'mean'), worst=('cov', 'min'),
                                       p10=('cov', lambda c: np.quantile(c, 0.10))).reset_index()
    sty = {'mondrian': ('#d95f02', 's', 'Mondrian CP ($\\alpha$ varied)'), 'pca': ('0.55', '^', 'WM-CP, PCA weights'),
           'score': ('#1b9e77', 'o', 'WM-CP, score weights (ours)')}
    fig, ax = plt.subplots(1, 2, figsize=(4.8, 2.25))
    for v, (c, mk, lab) in sty.items():
        d = fr[fr.v == v].sort_values('size')
        d = d[(d['size'] > 1.4) & (d['size'] < 1.95)]
        p = fr[(fr.v == v) & np.isclose(fr.alpha, 0.10)]
        for a_, col in zip(ax, ['worst', 'p10']):
            a_.plot(d['size'], d[col], c=c, lw=1.1, label=lab)
            a_.plot(p['size'], p[col], marker=mk, c=c, ms=5, ls='none', mec='white', mew=0.6)
    for a_, t in zip(ax, ['Worst-site coverage', '10th-percentile site coverage']):
        a_.set_xlabel('Mean set size (labels)')
        a_.set_title(t, fontsize=8)
    ax[0].axhline(0.85, ls=':', c='#b2182b', lw=0.8)
    ax[1].axhline(0.90, ls='--', c='k', lw=0.6)
    ax[0].set_ylabel('Coverage')
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(fdir, 'frontier.pdf'))
    fig.savefig(os.path.join(fdir, 'frontier.png'), dpi=200)

    R = pd.read_csv(os.path.join(args.results, 'per_site_results.csv'))
    sm = R.groupby(['method', 'site']).coverage.mean().unstack(0)
    sm['score'] = g10['cov']
    order = sm.sort_values('mondrian').index
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(4.8, 2.3))
    st = {'marginal': dict(marker='o', c='0.65', ms=3.5, label='Marginal CP'),
          'mondrian': dict(marker='s', c='#d95f02', ms=3.5, label='Mondrian CP'),
          'score': dict(marker='v', c='#1b9e77', ms=4.5, label='WM-CP, score weights (ours)')}
    for i, s in enumerate(order):
        ax.plot([i, i], [sm.loc[s, 'mondrian'], sm.loc[s, 'score']], c='0.8', lw=0.7, zorder=0)
    for m, k in st.items():
        ax.plot(x, sm.loc[order, m], ls='none', mec='none', **k)
    ax.axhline(0.90, ls='--', c='k', lw=0.7)
    ax.axhline(0.85, ls=':', c='#b2182b', lw=0.8)
    ax.text(len(order) - 0.4, 0.902, 'target 0.90', ha='right', va='bottom', fontsize=6.5)
    ax.text(len(order) - 0.4, 0.847, 'failure line 0.85', ha='right', va='top', fontsize=6.5, color='#b2182b')
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=60, ha='right', fontsize=6.3)
    ax.set_ylabel('Coverage at held-out site')
    ax.set_ylim(0.76, 1.0)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'per_site_coverage.pdf'))
    fig.savefig(os.path.join(fdir, 'per_site_coverage.png'), dpi=200)

    err = {}
    for e in cache:
        ps = np.bincount(e['ycal'], minlength=2) / len(e['ycal'])
        pt = em_prior(e['Pte'], ps)
        err.setdefault(e['site'], []).append(abs(pt[1] - e['yte'].mean()))
    err = pd.Series({k: np.mean(v) for k, v in err.items()})
    loss = (sm.wm_hyb - sm.wm_cov).loc[err.index]
    rho, p = spearmanr(err, loss)
    pd.DataFrame({'em_prior_abs_error': err, 'hybrid_minus_wmcov_coverage': loss}).to_csv(
        os.path.join(out, 'hybrid_prior_em_error.csv'))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.scatter(err, loss, s=16, c='#7570b3', zorder=3)
    for s in loss.sort_values().index[:5]:
        ax.annotate(s, (err[s], loss[s]), fontsize=6.3, xytext=(4, 1), textcoords='offset points')
    ax.axhline(0, c='0.6', lw=0.6)
    ax.set_xlabel(r'Mean $|\hat\pi_T-\pi_T|$ per site (EM)')
    ax.set_ylabel('Coverage: hybrid $-$ WM-CP')
    ax.set_title(rf'Spearman $\rho={rho:.2f}$, $p={p:.1e}$', fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'prior_error.pdf'))
    fig.savefig(os.path.join(fdir, 'prior_error.png'), dpi=200)
    print(f'EM prior error vs hybrid loss: Spearman rho={rho:.3f}, p={p:.2e}')
    print('Wrote outputs to', os.path.abspath(out))


if __name__ == '__main__':
    main()
