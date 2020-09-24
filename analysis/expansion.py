# %% ==================== EXPANSION ====================
from statsmodels.stats.proportion import proportion_confint

opt_exp = load_cf('OptimalPlus').groupby('wid').expand.mean()
opt_pure_exp = load_cf('OptimalPlusPure').groupby('wid').expand.mean()
cf = load_cf('Human')
@figure()
def expansion():
    fig, axes = setup_variance_plot()
    for ax, (_, d) in zip(axes.flat, cf.groupby(['variance'])):
        plt.sca(ax)
        g = d.expand.groupby('wid')
        est = g.mean()
        lo, hi = proportion_confint(g.apply(sum), g.apply(len))
        err = np.vstack([(est - lo).values, (hi - est).values])
        idx = np.argsort(est)
        plt.errorbar(np.arange(len(est)), est[idx], yerr=err[:, idx], color='k', label='Humans')
        
        plt.plot(np.arange(len(est)), opt_exp[est.index][idx], color=palette['Optimal'], label='Optimal')
        plt.plot(np.arange(len(est)), opt_pure_exp[est.index][idx], color=lb, label='Pure Optimal')

        plt.xticks([])
        plt.xlabel("Participant")
        plt.ylim(-0.05, 1.05)
        plt.ylabel('Expansion Rate')
        plt.axhline(1, label='Forward Search', color=palette['BreadthFirst'])
    axes.flat[0].legend()


# %% --------
edf = pd.DataFrame(get_result(VERSION, 'expansion.json')).set_index('wid')
edf['gain'] = edf.q_jump - edf.q_expand
edf['jump'] = ~edf['expand']


@figure()
def expansion_value():
    sns.regplot('gain', 'jump', data=edf, logistic=True, x_bins=np.linspace(-0.75, 0.75, 5), color='black')
    plt.xlabel('Q(Jump) - Q(Expand)')
    plt.ylabel('P(Jump)')

# %% --------

pdf['expanding'] = cf.groupby('wid')['expanding'].mean()
sns.swarmplot('variance', 'expanding', data=pdf, size=3)
plt.ylabel('Expansion Rate')
show()
# %% --------

# for w, x in :
    # proportion_confint(x.sum(), len(x))
    # cf.groupby('wid')

# cf = pd.DataFrame(get_result(VERSION, 'click_features.json')) \
#     .set_index('wid').loc[keep]


x = load_cf('OptimalPlusPure')



fig, axes = setup_variance_plot()
x = pdf[['variance', 'n_click']].join(x.groupby('wid').expand.mean()).set_index('variance')
for ax, var in zip(axes.flat, VARIANCES):
    ax.scatter('n_click', 'expand', data=x.loc[var])

show()

