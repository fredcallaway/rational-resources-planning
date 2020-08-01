

# %% ==================== PARETO FRONT ====================

mdp2var = {}
for x in tdf[['variance', 'mdp']].itertuples():
    mdp2var[x.mdp] = x.variance

model_pareto = pd.concat((pd.read_csv(f) for f in glob('model/mdps/pareto/*')), sort=True)
model_pareto = model_pareto.set_index('mdp').loc[tdf.mdp.unique()].reset_index()
model_pareto['variance'] = model_pareto.mdp.apply(mdp2var.get)
model_pareto.set_index(['model', 'variance'], inplace=True)
model_pareto.sort_values('cost', inplace=True)
model_pareto.rename(columns={'clicks': 'n_click', 'reward': 'term_reward'}, inplace=True)
model_pareto.index.unique()

# %% --------

plt.rc('legend', fontsize=10, handlelength=2)

def plot_model(variance, model):
    plt.plot('n_click', 'term_reward', data=model_pareto.loc[model, variance], 
        label=model, marker='.', color=palette[model])

@figure()
def pareto():
    X = tdf.reset_index().set_index('variance')
    fig, axes = setup_variance_plot()
    for i, v in enumerate(variances):
        plt.sca(axes.flat[i])
        for model in MODELS:
            if model == 'OptimalPlus':
                continue
            plot_model(v, model)
            
        g = X.loc[v].groupby('wid'); x = 'n_click'; y = 'term_reward'
        sns.regplot(g[x].mean(), g[y].mean(), lowess=True, color=palette['Human'], label='Human',
                   scatter=False).set_zorder(20)
        plt.scatter(g[x].mean(), g[y].mean(), color=palette['Human'], s=5).set_zorder(21)

        # plt.errorbar(g[x].mean(), g[y].mean(), yerr=g[y].sem(), xerr=g[x].sem(), 
        #              label='Human', fmt='.', color='#333333', elinewidth=.5)

        if ncol > 1:
            plt.title(f'{v.title()} Variance')
        plt.ylabel("Expected Reward")
        plt.xlabel("Number of Clicks")
        if i == 0:
            plt.legend(loc='lower right')

# %% --------

def linear_interpolate(x1, x2, y1, y2, x):
    d = (x - x1) / (x2 - x1)
    return y1 + d * (y2 - y1)

# x1, x2, y1, y2 = 1, 4, 2, 6
# plt.plot([x1, x2], [y1, y2])
# x = 2
# plt.plot([x], [linear_interpolate(x1, x2, y1, y2, x)], 'o')
# show()

mdps = tdf.mdp.unique()
optimal = model_pareto.loc['Optimal'].set_index('mdp').loc[mdps].set_index('cost')[['n_click', 'term_reward']]
human = tdf.groupby('wid')[['n_click', 'term_reward']].mean()

def pareto_loss(row):
    xvar = 'n_click'; yvar = 'term_reward'
    xvar = human.columns.drop(yvar)[0]
    x_diff = (optimal[xvar] - row[xvar]).values

    cross_point = (x_diff < 0).argmax()
    assert cross_point > 0
    assert x_diff[cross_point - 1] > 0
    opt = optimal.iloc[cross_point-1:cross_point+1]
    opt_xs = list(opt[xvar])
    hum_x = row[xvar]

    assert opt_xs[0] > hum_x > opt_xs[1]
        
    opt_ys = list(opt[yvar])
    opt_y = linear_interpolate(*opt_xs, *opt_ys, hum_x)
    hum_y = row[yvar]

    if opt_y == 0:
        return np.nan
    d = opt_y - hum_y
    return d, d / opt_y

loss, pct_loss = human.apply(pareto_loss, axis=1, result_type='expand').mean().values.T
write_tex(f'pareto_loss', f'{loss:.2f} ({pct_loss*100:.0f}\%)')
