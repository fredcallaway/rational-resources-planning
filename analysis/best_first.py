bfo = pd.DataFrame(get_result(VERSION, 'bestfirst_optimal.json'))
bfo.sort_values('cost')
bfr = pd.DataFrame(get_result(VERSION, 'bestfirst_random.json'))
bfo = bfo.sort_values('n_click')
bfr = bfr.sort_values('n_click')
pdf['best_first'] = load_cf('Human').query('~is_term').groupby('wid').is_best.mean()

# %% ==================== plot ====================

@figure()
def plot_best_first():    
    plt.figure(figsize=(4,4))
    pdf.best_first.plot.hist(bins=np.linspace(0,1,11), color=palette['Human'])
    plt.xlabel('Proportion of Clicks on Best Path')

# %% --------

@figure()
def best_first_by_clicks(ax=None):
    if ax is None:
        plt.figure(figsize=(4,4))
    else:
        plt.sca(ax)

    plt.axhline([1], label='BestFirst', color=palette['Best'], lw=3)
    x = bfo.query('0 < cost < 50')
    plt.plot(x.n_click, x.best_first, label="Optimal", color=palette["Optimal"], lw=3, alpha=0.8)
    
    plt.plot(bfr.n_click, bfr.best_first, label="Random", color=palette["Random"], lw=3, alpha=0.8)
    
    plt.scatter('n_click', 'best_first', data=pdf, color=palette["Human"], label='Human', s=10)


    plt.ylabel('Proportion of Clicks on Best Path')
    plt.xlabel('Number of Clicks per Trial')
    plt.xlim(-0.5,15.5)
    plt.ylim(0.48, 1.02)


# %% ==================== stats ====================

x = load_cf('Human').query('~is_term')
bf_rate = x.groupby('wid').is_best.mean()
bf_rand_rate = x.groupby('wid').p_best_rand.mean()
write_tex("best_first", mean_std(bf_rate*100, fmt='pct'))

from statsmodels.stats.proportion import proportions_ztest
r = load_cf('Random').query('~is_term').is_best
h = x.is_best
z, p = proportions_ztest([h.sum(), r.sum()], [len(h), len(r)], alternative='larger')
write_tex("best_first_random", rf"{bf_rand_rate.mean() * 100:.1f}\%" )
write_tex("best_first_test", rf"$z={z:.1f},\ {pval(p)}$")

from scipy.stats import spearmanr
r, p = spearmanr(pdf.best_first, pdf.n_click)
write_tex('best_first_click', fr'$\rho={r:.3f},\ {pval(p)}$')

# %% --------
# from statsmodels.formula.api import ols
# model = ols('best_first ~ n_click', data=pdf).fit()
# write_tex('best_first_click', fr'$B={model.params.n_click:.4f},\ {pval(model.pvalues.n_click)}$')


# %% ==================== correct for random ====================

def linear_interpolate(x1, x2, y1, y2, x):
    assert x1 < x < x2
    d = (x - x1) / (x2 - x1)
    return y1 + d * (y2 - y1)

def get_closest(row, comp, xvar, yvar):
    comp = comp.sort_values(xvar)
    x_diff = (comp[xvar] - row[xvar]).values
    cross_point = (x_diff > 0).argmax()
    
    if cross_point == 0:
        o = comp.iloc[0]
        assert row[xvar] > o[xvar]
        return o[yvar]
    else:
        assert x_diff[cross_point - 1] < 0
        o = comp.iloc[cross_point-1:cross_point+1]
        return linear_interpolate(*o[xvar], *o[yvar], row.n_click)

rand_best = pdf.apply(get_closest, axis=1, comp=bfr, xvar='n_click', yvar='best_first')
r, p = spearmanr(pdf.best_first - rand_best, pdf.n_click)
write_tex('best_first_click_corrected', fr'$\rho={r:.3f},\ {pval(p)}$')


# %% --------
sns.regplot(pdf.n_click, pdf.best_first - rand_best)
show()