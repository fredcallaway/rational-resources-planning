
# %% ==================== Second click ====================

def breadth_depth_data(models=MODELS, noclick='drop'):
    dfs = []
    d = tdf.set_index('variance').copy()
    d['agent'] = 'Human'
    dfs.append(d)
    for k in models:
        d = model_trial_features(k)
        dfs.append(d)
    data = pd.concat(dfs, sort=False)
    data = data[['agent', 'first_revealed', 'second_same']].copy()
    data = data.loc[~data.first_revealed.isna()]  # no first click

    data.first_revealed = data.first_revealed.astype(int)
    if noclick == 'drop':
        data = data.loc[~data.second_same.isna()]
    elif noclick == 'zero':
        data.second_same = data.second_same.fillna(0.)
    else:
        raise ValueError("noclick")
    return data

@figure()
def plot_second_click(axes=None, models=['OptimalPlus', 'BestFirst', 'BreadthFirst', 'DepthFirst']):
    noclick = 'drop'
    bdd = breadth_depth_data(models, noclick=noclick)
    if axes is None:
        fig, axes = setup_variance_plot(title=True)
    for i, v in enumerate(VARIANCES):
        plt.sca(axes.flat[i])
        agents = [*reversed(models), 'Human']
        data = bdd.loc[v].query('agent == @agents')
        sns.pointplot('first_revealed', 'second_same', hue='agent', hue_order=agents,
            palette=palette, data=data, legend=False)
        plt.xlabel('First Revealed Value')
        plt.ylim(-0.05, 1.05)
        if i == 0:
            if noclick == 'zero':
                plt.ylabel('Probability of Second\nClick on Same Path')
            else:
                plt.ylabel('Proportion of Second\nClicks on Same Path')
            plt.legend()
            figs.reformat_legend()
            plt.yticks([0,1])
        else:
            plt.legend().remove()
            plt.ylabel('')

# %% ==================== Depth curve ====================


@figure()
def plot_depth_curve(axes=None):
    # dcd = pd.concat(load_depth_curve(k) for k in ['Human', 'OptimalPlus', 'BestFirst']).set_index('variance')
    dcd = load_depth_curve('Human')
    if axes is None:
        fig, axes = setup_variance_plot(title=True)
    for i, v in enumerate(VARIANCES):
        plt.sca(axes.flat[i])
        sns.lineplot('click', 'cumdepth', hue='agent', 
            palette=palette, data=dcd.loc[v])
        plt.xlabel('Click Number')
        # plt.ylim(-0.05, 1.05)
        if i == 0:
            plt.ylabel('Maximum Depth Clicked')
            plt.legend()
            figs.reformat_legend()
        else:
            plt.legend().remove()
            plt.ylabel('')




# %% ==================== HEATMAP ====================

def plot_depth_heat(agent, axes):
    # base = matplotlib.cm.get_cmap('Blues', 512)
    # cmap = matplotlib.colors.ListedColormap(base(np.linspace(0.2, 1, 512 * 0.8)))
    cmap = 'Blues'

    dcd = load_depth_curve(agent)
    D = dcd.set_index('variance')

    for i, (ax, var) in enumerate(zip(axes.flat, VARIANCES)):
        plt.sca(ax)
        X = D.loc[var].groupby(['depth', 'click']).apply(len).unstack()
        # n_trial = (tdf.variance == var).sum()
        X /= X[1].sum()
        g = sns.heatmap(X, cmap=cmap, vmin=0, vmax=1, linewidths=1, cbar=True)
        g.invert_yaxis()
        if i > 0:
            plt.ylabel('')
        figs.reformat_labels()
    return g

@figure()
def depth_heat():
    fig, axes = setup_variance_plot(2, title=True)
    plot_depth_heat('Human', axes[0])
    plot_depth_heat('OptimalPlus', axes[1])

# %% ==================== BARS ====================

@figure()
def first_click_depth(axes=None):
    if axes is None:
        fig, axes = setup_variance_plot(title=True)

    agents = ['Human', 'OptimalPlus', 'OptimalPlusExpand', ]
    dd = pd.concat([load_depth_curve(h).query('click == 1')
        for h in agents]).set_index(['agent', 'variance']).depth

    def make_hist(m, var):
        x = dd.loc[m, var].value_counts(sort=False)
        x /= x.sum()
        return x

    for i, (ax, var) in enumerate(zip(axes.flat, VARIANCES)):
        d = pd.DataFrame({m: make_hist(m, var) for m in agents})
        d.plot.bar(color=[palette[m] for m in agents], 
            rot=0, ax=ax, legend=i==0)
        ax.set_ylim(0, 1)
        ax.set_xlabel('First Clicked Depth')
        if i == 0:
            ax.set_ylabel('Proportion of Trials')
            figs.reformat_legend(ax=ax, OptimalPlusExpand='Optimal + Forward')


# %% --------
# X = pd.concat(load_cf(x) for x in ['Human', ])

x = load_cf('Human')
rate = x.query('variance == "constant" and not is_term').groupby('wid').expand.mean()
write_tex(f'expansion_Human', mean_std(100*rate, fmt='pct', digits=0))

rate = load_cf('OptimalPlus').query('variance == "constant" and not is_term').expand.mean()
write_tex('expansion_OptimalPlus', f'{rate*100:.1f}\\%')
rate.mean()



# # %% --------
# dd.groupby(['variance']).apply(make_hist)


# def make_thing():
#     for m in ['Human', 'OptimalPlus']:
#         dd.loc[m, var].value_counts()
#         x = dd.loc[m, var].value_counts(sort=False)
#         x /= x.sum()
#         yield x
# fig, axes = setup_variance_plot(title=True)
# for i, (ax, var) in enumerate(zip(axes.flat, VARIANCES)):

#         # sns.distplot(dd.loc[m, var], kde=False, norm_hist=True, ax=ax)
# show()



# # %% --------

# sns.distplot(dd.loc[m, var], kde=False, norm_hist=True)
# show()

# dd.value_counts().plot.bar()
# show()

# # %% --------

# X = d.groupby(['click', 'cumdepth']).apply(len).unstack()
# sns.heatmap(X.T)
# show()

# # %% --------

# # def unroll(df):
# #     rows = []
# #     for row in df.itertuples():
# #         cm = pd.Series(depth[c] for c in row.clicks).cummax()
# #         for i, c in enumerate(cm):
# #             rows.append([row.variance, i,])
# #     return pd.DataFrame(rows, columns=['variance', 'idx', 'depth'])

# # clicks = unroll(df)


# # # In[614]:


# # sns.lineplot('idx', 'depth', hue='variance', data=clicks)
# # plt.xlabel('Click Number')
# # plt.ylabel('Maximum Depth Clicked')
# # savefig('maxdepth')


# # # In[488]:


# # tree = [[1, 5, 9, 13], [2], [3, 4], [], [], [6], [7, 8], [], [], [10], [11, 12], [], [], [14], [15, 16], [], []]
# depth = [0, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3]

# def first_revealed(row):
#     if len(row.clicks) < 1:
#         return 0
#     return row.state_rewards[row.clicks[0]]
    
# def second_click(row):
#     if len(row.clicks) < 2:
#         return 'none'
#     c1 = row.clicks[1]
#     if depth[c1] == 1:
#         return 'breadth'
#     if depth[c1] == 2:
#         return 'depth'

# tdf['second_click'] = tdf.apply(second_click, axis=1)
# tdf['first_revealed'] = tdf.apply(first_revealed, axis=1)
# X = tdf.groupby(['variance', 'first_revealed', 'second_click']).apply(len)
# N = tdf.groupby(['variance', 'first_revealed']).apply(len)
# X = (X / N).rename('rate').reset_index()

# fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
# order = ['decreasing', 'constant', 'increasing']
# for i, (var, d) in enumerate(X.query('second_click != "none"').groupby('variance')):
#     i = order.index(var)
#     ax = axes[i]; plt.sca(ax)
#     sns.barplot('first_revealed', 'rate', hue='second_click', data=d)
#     plt.title(f'{var.title()} Variance')
#     plt.xlabel('First Revealed Value')
#     if i == 0:
#         plt.ylabel('Proportion')
#     else:
#         plt.ylabel('')
#     if i == 2:
#         ax.legend().set_label('Second Click')
#     else:
#         ax.legend().remove()
# savefig('breadth-depth')


# # In[ ]:


# import json

# def parse_sim_clicks(x):
#     if x == "Int64[]":
#         return []
#     else:
#         return json.loads(x)
    
# sdf = pd.concat(
#     pd.read_csv(f"model/results/{code}/simulations.csv")
#     for code in CODES
# )
# sdf['clicks'] = sdf.clicks.apply(parse_sim_clicks)
# sdf.state_rewards = sdf.state_rewards.apply(json.loads)
# sdf['second_click'] = sdf.apply(second_click, axis=1)
# sdf['first_revealed'] = sdf.apply(first_revealed, axis=1)
# sdf['model'] =  sdf.wid.str.split('-').str[0]

# sdf.set_index('mdp', inplace=True)
# sdf['variance'] = mdps.variance
# sdf.reset_index(inplace=True)
# sdf.set_index('model', inplace=True)


# # In[547]:


# def plot_breadth_depth_model(model):
#     df = sdf.loc[model]
#     X = df.groupby(['variance', 'first_revealed', 'second_click']).apply(len)
#     N = df.groupby(['variance', 'first_revealed']).apply(len)
#     X = (X / N).rename('rate').reset_index()

#     fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
#     order = ['decreasing', 'constant', 'increasing']
#     for i, (var, d) in enumerate(X.query('second_click != "none"').groupby('variance')):
#         i = order.index(var)
#         ax = axes[i]; plt.sca(ax)
#         sns.barplot('first_revealed', 'rate', hue='second_click', data=d)
#         plt.title(f'{var.title()} Variance')
#         plt.xlabel('First Revealed Value')
#         if i == 0:
#             plt.ylabel(f'{model} Proportion')
#         else:
#             plt.ylabel('')
#         if i == 2:
#             ax.legend().set_label('Second Click')
#         else:
#             ax.legend().remove()
#     savefig(f'breadth-depth-{model}')

# # plot_breadth_depth_model('Optimal')
# # plot_breadth_depth_model('BestFirst')
# plot_breadth_depth_model('BreadthFirst')
# plot_breadth_depth_model('DepthFirst')

