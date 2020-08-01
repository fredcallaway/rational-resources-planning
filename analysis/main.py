%run setup



# %% ==================== BACKWARD ====================

leaves = {20, 15, 10, 5}
tdf['first_click'] = tdf.clicks.apply(lambda x: x[0] if x else 0)
tdf['backward'] = tdf.first_click.isin(leaves)
pdf['backward'] = tdf.groupby('wid').backward.mean()

@figure(reformat_labels=True)
def backwards():
    sns.swarmplot('variance', 'backward', data=pdf)
    plt.xlabel('a')





# %% ==================== FITTING DIAGNOSTICS ====================

g = sns.FacetGrid(row='variance', col='model', data=fits, aspect=1, margin_titles=True)
g.map(sns.scatterplot, 'nll', 'cv_nll')
show()

d = cv_fits.query('model == "BreadthFirst" and variance == "decreasing"').reset_index()
plt.figure(figsize=(10,4))

sns.stripplot('wid', 'ε', data=d, palette=sns.color_palette()[:3])
plt.xticks([])
show()

d.set_index('wid').β_depth.sort_values()

# %% --------
F = fits.set_index(['wid', 'model']).cv_nll.unstack()
F.loc['w016ae6c']

fits.set_index(['wid', 'model']).cost

# %% ==================== PARAMETERS ====================

cv_fits = pd.concat([pd.read_csv(f'model/results/{VERSION}/mle/{model}-cv.csv') 
                     for model in MODELS], sort=False).set_index('wid')

cv_fits['click_delay'] = pdf.click_delay
cv_fits['variance'] = pdf.variance
# cv_fits.to_csv('cv_fits.csv')
# cv_fits.rename(columns=lambda x: x.replace('ε', 'eps').replace('β', 'beta').replace('θ', 'theta')).to_csv('cv_fits.csv')
# %% --------

sns.scatterplot('β_term', 'β_select', data=fits.query('model == "OptimalPlus"'))
plt.plot([0,50], [0,50], c='k')
plt.axis('square')
show()

# %% --------
f = cv_fits.query('model == "Optimal"')
f.groupby('wid').cost.std()

# %% --------

g = sns.FacetGrid(col='variance', data=f)
g.map(sns.scatterplot, 'ε', 'β')
show()

# %% --------

g = sns.FacetGrid(row='variance', sharex=False, aspect=3, palette=sns.color_palette()[:3],
    data=cv_fits.query('model == "BreadthFirst"').reset_index())

g.map(sns.stripplot, 'wid', 'β_depth')
g.set_xticklabels([])
show()


# %% --------


# %% ==================== CLICK DELAY ====================

@figure()
def click_delay_click():
    sns.swarmplot('click_delay', 'n_click', data=pdf,
        hue='variance',
        order='1.0s 2.0s 3.0s'.split())

    plt.xlabel('Click Delay')
    plt.ylabel('Number of Clicks')



# %% ==================== LEARNING ====================

sns.lineplot('i', 'n_click', hue='variance', data=tdf)
show()

sns.lineplot('i', 'score', hue='variance', data=tdf)
show()

# %% ==================== BREADTH DEPTH ====================



# # In[611]:


# def unroll(df):
#     rows = []
#     for row in df.itertuples():
#         cm = pd.Series(depth[c] for c in row.clicks).cummax()
#         for i, c in enumerate(cm):
#             rows.append([row.variance, i,])
#     return pd.DataFrame(rows, columns=['variance', 'idx', 'depth'])

# clicks = unroll(df)


# # In[614]:


# sns.lineplot('idx', 'depth', hue='variance', data=clicks)
# plt.xlabel('Click Number')
# plt.ylabel('Maximum Depth Clicked')
# savefig('maxdepth')


# # In[488]:


# tree = [[1, 5, 9, 13], [2], [3, 4], [], [], [6], [7, 8], [], [], [10], [11, 12], [], [], [14], [15, 16], [], []]
depth = [0, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3]

def first_revealed(row):
    if len(row.clicks) < 1:
        return 0
    return row.state_rewards[row.clicks[0]]
    
def second_click(row):
    if len(row.clicks) < 2:
        return 'none'
    c1 = row.clicks[1]
    if depth[c1] == 1:
        return 'breadth'
    if depth[c1] == 2:
        return 'depth'

tdf['second_click'] = tdf.apply(second_click, axis=1)
tdf['first_revealed'] = tdf.apply(first_revealed, axis=1)
X = tdf.groupby(['variance', 'first_revealed', 'second_click']).apply(len)
N = tdf.groupby(['variance', 'first_revealed']).apply(len)
X = (X / N).rename('rate').reset_index()

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
order = ['decreasing', 'constant', 'increasing']
for i, (var, d) in enumerate(X.query('second_click != "none"').groupby('variance')):
    i = order.index(var)
    ax = axes[i]; plt.sca(ax)
    sns.barplot('first_revealed', 'rate', hue='second_click', data=d)
    plt.title(f'{var.title()} Variance')
    plt.xlabel('First Revealed Value')
    if i == 0:
        plt.ylabel('Proportion')
    else:
        plt.ylabel('')
    if i == 2:
        ax.legend().set_label('Second Click')
    else:
        ax.legend().remove()
savefig('breadth-depth')


# In[ ]:


import json

def parse_sim_clicks(x):
    if x == "Int64[]":
        return []
    else:
        return json.loads(x)
    
sdf = pd.concat(
    pd.read_csv(f"model/results/{code}/simulations.csv")
    for code in CODES
)
sdf['clicks'] = sdf.clicks.apply(parse_sim_clicks)
sdf.state_rewards = sdf.state_rewards.apply(json.loads)
sdf['second_click'] = sdf.apply(second_click, axis=1)
sdf['first_revealed'] = sdf.apply(first_revealed, axis=1)
sdf['model'] =  sdf.wid.str.split('-').str[0]

sdf.set_index('mdp', inplace=True)
sdf['variance'] = mdps.variance
sdf.reset_index(inplace=True)
sdf.set_index('model', inplace=True)


# In[547]:


def plot_breadth_depth_model(model):
    df = sdf.loc[model]
    X = df.groupby(['variance', 'first_revealed', 'second_click']).apply(len)
    N = df.groupby(['variance', 'first_revealed']).apply(len)
    X = (X / N).rename('rate').reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    order = ['decreasing', 'constant', 'increasing']
    for i, (var, d) in enumerate(X.query('second_click != "none"').groupby('variance')):
        i = order.index(var)
        ax = axes[i]; plt.sca(ax)
        sns.barplot('first_revealed', 'rate', hue='second_click', data=d)
        plt.title(f'{var.title()} Variance')
        plt.xlabel('First Revealed Value')
        if i == 0:
            plt.ylabel(f'{model} Proportion')
        else:
            plt.ylabel('')
        if i == 2:
            ax.legend().set_label('Second Click')
        else:
            ax.legend().remove()
    savefig(f'breadth-depth-{model}')

# plot_breadth_depth_model('Optimal')
# plot_breadth_depth_model('BestFirst')
plot_breadth_depth_model('BreadthFirst')
plot_breadth_depth_model('DepthFirst')



# # Scratch

# In[286]:


def plot_line(MODELS, d, val):
    plt.plot(MODELS, d.set_index('full_model').loc[MODELS][val], color='k', lw=1, alpha=0.3)

def plot_participants(val, MODELS=MODELS):
    ax = sns.swarmplot('full_model', val, data=ind, order=MODELS, palette=pal)
    for w, d in ind.groupby('wid'):
        plot_line(MODELS, d, val)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlabel('')
    plt.ylabel('Log Likelihood')
#         plot_line(MODELS[:3], d, val)
#         plot_line(MODELS[3:], d, val)
        
plot_participants('logp', MODELS=MODELS[:3])
plt.tight_layout()
plt.savefig('individual_likelihood_nobias.pdf')


# In[287]:


plot_participants('logp', MODELS=MODELS)
plt.tight_layout()
plt.savefig('individual_likelihood.pdf')


# In[ ]:


import json
# VERSION = 'webofcash-pilot-1.1'
VERSION = 'webofcash-1.2'
def load_trials(experiment):
    with open(f'data/{experiment}/trials.json') as f:
        data = json.load(f)

    for wid, trials in data.items():
        for t in trials:
            t['wid'] = wid
            yield t
            
data = pd.DataFrame(load_trials(VERSION))
data['n_click'] = data.reveals.apply(len)
data['raw_reward'] = data.score + data.n_click  # assumes cost = 1

