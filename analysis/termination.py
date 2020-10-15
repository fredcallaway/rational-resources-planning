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
    ax.set_yticks(ax.get_yticks()[1::2])
    ax.set_yticklabels(ylab[1::2])

    ax.invert_yaxis()
    figs.reformat_labels()
    return ax


def termination(x, y, height=3):
    agents = ['Human', 'OptimalPlus']
    nc = len(agents)
    fig, axes = plt.subplots(1, nc+1, figsize=(4*nc, height),
                             gridspec_kw={'width_ratios': [*([15] * nc), 1]})

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
    axes[-1].set_ylabel('Stopping Probability')

@figure()
def plot_termination():
    termination('best_next', 'term_reward', height=4)

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

# sns.lmplot('best_next', 'is_term', data=cf, logistic=False, scatter=False)
# %% --------
sns.lmplot('best_next', 'is_term', data=cf, logistic=False, x_bins=10)
show()

# %% ==================== stats ====================
cf = load_cf('Human')
preds = ['n_revealed', 'best_next', 'term_reward']
X = cf[['is_term', *preds]].copy()
X.is_term = X.is_term.astype(int)
X[preds] -= X[preds].mean()
X[preds] /= X[preds].std()
X = X.reset_index()
# %% --------
%%R -i X
library(lme4)
library(lmerTest)
model = glmer(is_term ~ best_next + term_reward + (1|wid), family=binomial, data=X)
summary(model)

# %% --------
from statsmodels.formula.api import logit
def do_fit(X):
    m = logit('is_term ~ best_next + term_reward', data=X).fit(disp=False)
    return m.pvalues

# pd.DataFrame(do_fit(d) for _)
ind_fits = X.groupby('wid').apply(do_fit)

bn_sig = (ind_fits.best_next < .05)
tr_sig = (ind_fits.term_reward < .05)

sig = ind_fits < .05
types = sig.apply(lambda row: (int(row.best_next), int(row.term_reward)), axis=1).value_counts()

types.items
for k, n in types.items():
    i = ''.join(map(str, k))
    write_tex(f'term_nsig_{i}', f'($N={n}$)')
    
for k in ['best_next', 'term_reward']:
    sig = ind_fits[k] < .05





# %% --------
ind_fits



