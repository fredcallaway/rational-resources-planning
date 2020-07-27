
# %% ==================== OLD ====================


df = pd.read_json(f'model/results/{VERSION}/click_features.json')
df.term_reward = df.term_reward.apply(int)

def get_agent(wid):
    return wid.split('-')[0] if '-' in wid else 'Human'

df['agent'] = df.wid.apply(get_agent)


def savefig(name):
    plt.tight_layout()
    plt.savefig(f'figs/{name}.png', dpi=300)


sns.catplot('n_revealed', 'is_term', data=df, kind='point', hue='agent', ci=False)
savefig('term_revealed')


sns.catplot('term_reward', 'is_term', data=df, kind='point', hue='agent', ci=False)
savefig('term_reward')


# x = df.groupby(['etr', 'n_revealed']).is_term.mean().reset_index()
def robust_mean(x):
    return np.mean(x)
    if len(x) < 5:
        return np.nan
    return np.mean(x)

# lims = {''}

def plot_adaptive(df, **kws):
    X = df.groupby(['term_reward', 'n_revealed']).is_term.apply(robust_mean).unstack()
    # X = df.groupby(['etr', 'n_revealed']).apply(len).unstack()
    sns.heatmap(X, cmap='Blues', linewidths=1, **kws).invert_yaxis()
    plt.xlabel('Number of Clicks Made')
#     plt.ylim(*lims['y'])
#     plt.xlim(*lims['x'])

    
fig, axes = plt.subplots(1, 4, figsize=(12, 3),
                         gridspec_kw={'width_ratios': [15, 15, 15, 1]})

# fig, axes = plt.subplots(1, 2, figsize=(8,4))

plt.sca(axes[0])
plot_adaptive(df.query('agent == "Optimal"'), cbar_ax=axes[3])
plt.ylabel("Best Expected Path Value")
plt.title("Optimal")

plt.sca(axes[1])
plot_adaptive(df.query('agent == "Human"'), cbar=False)
plt.title("Human")
plt.ylabel("")
plt.yticks(())

plt.sca(axes[2])
plot_adaptive(df.query('agent == "BestFirst"'), cbar=False)
plt.title("Best First")
plt.ylabel("")
plt.yticks(())
savefig('adaptive_satisficing')
