bfo = pd.DataFrame(get_result(VERSION, 'bestfirst.json'))
bfo = bfo.sort_values('n_click')

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
    # bfo.plot(label="Optimal", color=palette["Optimal"], lw=3)
    x = bfo.query('0 < cost < 50')
    plt.plot(x.n_click, x.best_first, label="Optimal", color=palette["Optimal"], lw=3)
    rand = load_cf('Random').query('~is_term').is_best.mean()
    plt.axhline([rand], label='Random', color=palette['Random'], lw=3)
    # x = bfo.query('cost == 0')
    # plt.scatter(x.n_click, x.best_first, color=palette["Optimal"], s=20, )
    # plt.annotate('cost = 0', )
    # sns.regplot('cost', 'best_first', lowess=True, data=pdf, color=palette["Human"])
    plt.scatter('n_click', 'best_first', data=pdf, color=palette["Human"], label='Human', s=10)
    plt.ylabel('Proportion of Clicks on Best Path')
    plt.xlabel('Number of Clicks per Trial')
    plt.xlim(-0.5,15.5)
    plt.ylim(0.48, 1.02)
    # plt.xticks([0, 15])
    plt.yticks([0.5, 0.75, 1])
    # sns.despine(trim=True);
    # plt.xlim(-0.05, 3.0)
    # plt.legend(loc='upper right')


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
# %% --------
# from statsmodels.formula.api import ols
# model = ols('best_first ~ n_click', data=pdf).fit()
# write_tex('best_first_click', fr'$B={model.params.n_click:.4f},\ {pval(model.pvalues.n_click)}$')

# %% --------
from scipy.stats import spearmanr
r, p = spearmanr(pdf.best_first, pdf.n_click)
write_tex('best_first_click', fr'$\rho={r:.3f},\ {pval(p)}$')

# %% --------
rdf = pdf[['n_click', 'best_first']]
%load_ext rpy2.ipython

# %% --------
%%R -i rdf -out
out = summary(lm(best_first ~ n_click, data=rdf))
# %% --------
out = %R -i rdf summary(lm(best_first ~ n_click, data=rdf))

