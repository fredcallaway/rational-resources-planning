# %% ==================== PARETO FRONT ====================

mdp2var = {}
for x in tdf[['variance', 'mdp']].itertuples():
    mdp2var[x.mdp] = x.variance

model_pareto = pd.concat((pd.read_csv(f) for f in glob('../model/mdps/pareto/*')), sort=True)
# model_pareto.model = model_pareto.model.str.replace('RandomSelection', 'Random')
model_pareto = model_pareto.set_index('mdp').loc[tdf.mdp.unique()].reset_index()
model_pareto['variance'] = model_pareto.mdp.apply(mdp2var.get)
model_pareto.set_index(['model', 'variance'], inplace=True)
model_pareto.sort_values('cost', inplace=True)
model_pareto.rename(columns={'clicks': 'n_click', 'reward': 'term_reward'}, inplace=True)
model_pareto.index.unique()

if EXPERIMENT == 1:
    PARETO_MODELS = ['Random', 'MetaGreedy', 'OptimalPlus', 'BestFirst', ]
else:
    PARETO_MODELS = ['OptimalPlus', 'BestFirst', 'BreadthFirst', 'DepthFirst', 'Random']

# PARETO_MODELS = [m for m in MODELS if not (m.endswith('Expand') or m.endswith('NoPrune'))]
palette['BreadthFirst'] = do
palette['DepthFirst'] = dp
palette['BestFirst'] = dg
# %% --------

def plot_model_pareto(variance, model): 
   plt.plot('n_click', 'term_reward', data=model_pareto.loc[model, variance], 
        label=model, color=palette[model],
        lw=2,
         # marker='.',
         )

@figure()
def plot_pareto(axes=None, legend=True, fit_reg=False, models=PARETO_MODELS):
    if axes is None:
        _, axes = setup_variance_plot()
    X = tdf.reset_index().set_index('variance')
    for i, v in enumerate(VARIANCES):
        plt.sca(axes.flat[i])
        for model in models:
            if model == 'OptimalPlus':
                model = 'Optimal'
            plot_model_pareto(v, model)
            
        g = X.loc[v].groupby('wid'); x = 'n_click'; y = 'score'
        sns.regplot(g[x].mean(), g[y].mean(), fit_reg=fit_reg, lowess=True, 
            color=palette['Human'], label='Human', scatter=False).set_zorder(20)
        plt.scatter(g[x].mean(), g[y].mean(), label='Human', color=palette['Human'], s=5).set_zorder(21)

        # plt.errorbar(g[x].mean(), g[y].mean(), yerr=g[y].sem(), xerr=g[x].sem(), 
        #              label='Human', fmt='.', color='#333333', elinewidth=.5)

        plt.ylabel("Reward")
        plt.xlabel("Number of Clicks")
        if i == 0:
            if legend:
                figs.reformat_legend()
                plt.legend(loc='lower right')
        else:
            plt.ylabel('')

# %% --------

def linear_interpolate(x1, x2, y1, y2, x):
    assert x1 < x < x2
    d = (x - x1) / (x2 - x1)
    return y1 + d * (y2 - y1)

# x1, x2, y1, y2 = 1, 4, 2, 6
# plt.plot([x1, x2], [y1, y2])
# x = 2
# plt.plot([x], [linear_interpolate(x1, x2, y1, y2, x)], 'o')
# show()

optimal = model_pareto.loc['Optimal'].set_index('mdp')[['n_click', 'term_reward']].sort_values('n_click')
random = model_pareto.loc['RandomSelection'].set_index('mdp')[['n_click', 'term_reward']].sort_values('n_click')

def only(x):
    assert len(x) == 1
    return x[0]


# %% --------
def get_closest_term_reward(row, agent):
    comp = agent.loc[row.mdp]
    x_diff = (comp['n_click'] - row['n_click']).values
    cross_point = (x_diff > 0).argmax()
    
    if cross_point == 0:
        o = comp.iloc[0]
        assert row['n_click'] > o['n_click']
        return o['term_reward']
    else:
        assert x_diff[cross_point - 1] < 0
        o = comp.iloc[cross_point-1:cross_point+1]
        return linear_interpolate(*o['n_click'], *o['term_reward'], row.n_click)


df = tdf.groupby('wid')[['n_click', 'term_reward', 'score']].mean()
df['mdp'] = tdf.groupby('wid').mdp.unique().apply(only)
df['optimal'] = df.apply(get_closest_term_reward, agent=optimal, axis=1)
df['random'] = df.apply(get_closest_term_reward, agent=random, axis=1)

gain = df.score - df.random
loss = df.optimal - df.score
pct_loss = loss / df.optimal
relative = (df.score - df.random) / (df.optimal - df.random)

write_tex(f'pareto_loss', f'{loss.mean():.2f} ({pct_loss.mean()*100:.0f}\%)')
write_tex(f'pareto_gain', f'{gain.mean():.2f}')
write_tex(f'pareto_relative', f'{100*relative.mean():.1f}\%')
write_tex(f'pareto_gain_wilcoxon', f'${pval(wilcoxon(gain).pvalue)}$')

# %% --------
x = 2.5
a = 1
b = 3
(x - a) / (b - a)

# %% --------

loss, pct_loss = -human.apply(pareto_relative, comparison=optimal, axis=1, result_type='expand').values.T
gain, pct_gain = human.apply(pareto_relative, comparison=random, axis=1, result_type='expand').values.T


loss, pct_loss = -human.apply(pareto_relative, comparison=optimal, 
    axis=1, result_type='expand').mean().values.T

gain, pct_gain = human.apply(pareto_relative, comparison=random, 
    axis=1, result_type='expand').mean().values.T
gain
