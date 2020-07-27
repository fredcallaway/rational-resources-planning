cf = pd.DataFrame(get_result(VERSION, 'click_features.json'))
for k, v in cf.items():
    if v.dtype == bool:
        cf[k] = v.astype(int)

cf['potential_gain'] = cf.max_competing - cf.term_reward

# %% --------
%%R -i cf
summary(glm(is_term ~ n_revealed + term_reward + potential_gain, data=cf))

# %% --------
def robust_mean(x):
    return np.mean(x)
    if len(x) < 5:
        return np.nan
    return np.mean(x)

def plot_adaptive(df, **kws):
    X = df.groupby(['term_reward', 'n_revealed']).is_term.apply(robust_mean).unstack()
    # X = df.groupby(['etr', 'n_revealed']).apply(len).unstack()
    sns.heatmap(X, cmap='Blues', linewidths=1, **kws).invert_yaxis()
    plt.xlabel('Number of Clicks Made')
#     plt.ylim(*lims['y'])
#     plt.xlim(*lims['x'])

plot_adaptive(cf)
# %% --------
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

# %% --------
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
