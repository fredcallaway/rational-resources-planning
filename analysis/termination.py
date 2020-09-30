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
ax = plot_term(cf, 'best_next', 'term_reward')

# %% ==================== correlation ====================
X = all_cfs['Human'][['n_revealed', 'term_reward', 'potential_gain']]
sns.pairplot(X, kind='reg')
show()

# %% ==================== stats ====================
# %load_ext rpy2.ipython
# cf = all_cfs['OptimalPlus']
m = "PureOptimal"
cf = load_cf(m, group=False)
for k, v in cf.items():
    if v.dtype == bool:
        cf[k] = v.astype(int)

cf.to_csv(f'{m}-term.csv', index=False)
print(f'{m}-term.csv')

# %% --------
sns.factorplot('best_next', )

# %% ==================== ADAPTIVE SATISFICING ====================

termination = get_result(VERSION, 'termination.json')
etrs = list(map(int, termination['etrs']))
idx = 1+np.arange(len(etrs))
idx = idx[0::2]
etrs = etrs[0::2]

@figure()
def adaptive_satisficing():
    cols = ['OptimalPlus', 'Human', 'BestFirstNoBestNext']

    fig, axes = plt.subplots(1, 4, figsize=(12, 3),
                             gridspec_kw={'width_ratios': [15, 15, 15, 1]})

    for i, col in enumerate(cols):
        plt.sca(axes[i])
        X, N = map(np.array, termination[col])
        if i == 0:
            sns.heatmap(X.T/N.T, cmap='viridis', linewidths=1, cbar_ax=axes[3])
            plt.yticks(idx, etrs, rotation='horizontal')
            plt.ylabel("Expected Value")
        else:
            sns.heatmap(X.T/N.T, cmap='viridis', linewidths=1, cbar=False)
            plt.yticks(())
        axes[i].invert_yaxis()
        plt.xlabel('Number of Clicks Made')
        plt.title('Satisficing BestFirst' if col == "Heuristic" else col)
        # plt.title(col)

# %% ==================== EXPECTED VS MAX ====================


evmv = get_result(VERSION, 'evmv.json')


@figure()
def expected_vs_max():
    cols = ['OptimalPlus', 'Human', 'BestFirstNoBestNext']

    fig, axes = plt.subplots(1, 4, figsize=(12, 3),
                             gridspec_kw={'width_ratios': [15, 15, 15, 1]})

    for i, col in enumerate(cols):
        plt.sca(axes[i])
        X, N = map(np.array, evmv[col])
        if i == 0:
            sns.heatmap(X.T/N.T, cmap='viridis', linewidths=1, cbar_ax=axes[3])
            plt.yticks(idx, etrs, rotation='horizontal')
            plt.ylabel("Expected Value")
        else:
            sns.heatmap(X.T/N.T, cmap='viridis', linewidths=1, cbar=False)
            plt.yticks(())
        axes[i].invert_yaxis()
        plt.xticks(idx, etrs)
        plt.xlabel('Maximum Possible Value')
        plt.title('Satisficing BestFirst' if col == "Heuristic" else col)
        # plt.title(col)
