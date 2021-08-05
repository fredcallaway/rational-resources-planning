# %% ==================== heatmaps ====================

def robust_mean(x):
    return np.mean(x)
    if len(x) < 5:
        return np.nan
    return np.mean(x)

def plot_term(df, x, y, **kws):
    base = matplotlib.cm.get_cmap('Blues', 512)
    cmap = matplotlib.colors.ListedColormap(base(np.linspace(0.2, 1, 512 * 0.8)))

    # df = df.query('term_reward >= -10')
    max_clicks = 16
    df = df.query('n_revealed < @max_clicks')    
    X = df.groupby([y, x]).is_term.apply(robust_mean).unstack()

    ax = sns.heatmap(X, cmap=cmap, vmin=0, vmax=1, linewidths=1, **kws)

    xlab = [int(float(t.get_text())) for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_xticklabels(xlab[::2])
    plt.xticks(rotation=0)
    
    ylab = [int(float(t.get_text())) for t in ax.get_yticklabels()]
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_yticklabels(ylab[::2])


    ax.invert_yaxis()
    plt.xlim(-0.5, X.shape[1] + 0.5)
    plt.ylim(-0.5, X.shape[0] + 0.5)
    figs.reformat_labels()
    return ax

def termination(x, y, height=3):
    agents = ['Human', 'OptimalPlusPure']
    # agents = ['Human', 'OptimalPlus']
    nc = len(agents)
    fig, axes = plt.subplots(1, nc+1, figsize=(4*nc, height),
                             gridspec_kw={'width_ratios': [*([16] * nc), 1]})

    cfs = {k: load_cf(k) for k in agents}
    multi_cf = pd.concat([load_cf(k) for k in agents])
    for k in (x, y):
        multi_cf[k] = pd.Categorical(multi_cf[k])

    for i, (name, cf) in enumerate(multi_cf.groupby('agent')):
        plt.sca(axes[i])
        if i == 0:
            plot_term(cf, x, y, cbar_ax=axes[-1])
        else:
            plot_term(cf, x, y, cbar=False)
            # plt.yticks(())
            plt.ylabel("")
        plt.title(figs.nice_name(name))
    axes[-1].set_ylabel('Stopping Probability', labelpad=10)

@figure(despine=True, tight=False)
def plot_termination():
    # with sns.axes_style('white'):
    termination('best_next', 'term_reward', height=4)
    # sns.despine(offset=10, trim=True)

# %% --------
@figure()
def plot_nextbest():
    X = pd.concat(load_cf(v) for v in ['Human', 'OptimalPlusPure'])
    sns.pointplot('best_next', 'is_term', data=X, hue='agent', palette=palette)
    plt.ylabel('Probability of Stopping')
    figs.reformat_legend()
    figs.reformat_labels()

# %% --------
@figure()
def plot_satisfice():
    X = pd.concat(load_cf(v) for v in ['Human', 'OptimalPlusPure'])
    sns.pointplot('term_reward', 'is_term', data=X, hue='agent', palette=palette)
    plt.ylabel('Probability of Stopping')
    figs.reformat_legend()
    figs.reformat_labels()

# %% ==================== stats  ====================

from statsmodels.formula.api import logit 

@do_if(True)
def this():
    cf = load_cf('Human').query('n_revealed < 16')
    m = logit(f'is_term.astype(int) ~ term_reward + best_next + prob_maximal', data=cf).fit()
    m.summary()
    for k in ['term_reward', 'best_next']:
        lo, hi = m.conf_int().loc[k]
        # write_tex(f'expansion_logistic', rf'$\beta = {m.params.gain_z:.3f} [{lo:.3f} {hi:.3f}], {pval(m.pvalues.gain_z)}$')
        write_tex(f'term_human_{k}', rf'$B = {m.params[k]:.3f}$, 95\% CI [{lo:.3f}, {hi:.3f}], ${pval(m.pvalues[k])}$')
        # write_tex(f'term_human_{k}', rf'$B = {m.params[k]:.3f}$, {label} [{lo:.3f}, {hi:.3f}], ${pval(m.pvalues[k])}$'))

@do_if(True)
def this():
    cf = load_cf('OptimalPlusPure').query('n_revealed < 16')
    m = logit(f'is_term.astype(int) ~ term_reward + best_next + prob_maximal', data=cf).fit()
    m.summary()
    for k in ['term_reward', 'best_next']:
        write_tex(f'term_optimal_{k}', f'{m.params[k]:.3f}')

# # %% --------
# preds = ['best_next', 'term_reward', 'n_revealed']
# models = {
#     pred: logit(f'is_term ~ {pred}', data=X).fit()
#     for pred in preds
# }
# for pred in preds:
#     models[f'no_{pred}'] = logit('is_term ~ ' + ' + '.join([p for p in preds if p != pred]), data=X).fit()
# models['full'] = logit('is_term ~ best_next + term_reward + n_revealed', data=X).fit()

# # %% --------
# X = pd.DataFrame({k: {
#         'R^2': round(m.prsquared, 3),
#         'LL': round(m.llf),
#     } 
#     for k, m in models.items()}).T
# X

# for p in preds:
#     print(p, X.LL.full - X.LL['no_' + p])
    


