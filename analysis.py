
# %% ==================== LOAD DATA AND SET UP ====================

from analysis_utils import *

EXPERIMENT = 1
VERSION = 'exp1'
MODELS = ['BreadthFirst',  'DepthFirst', 'BestFirst', 'Optimal']
VARIANCES = ['decreasing', 'constant', 'increasing']

pdf, tdf = load_data(VERSION)
pdf.variance = pd.Categorical(pdf.variance, categories=VARIANCES)

# Drop incomplete
pdf = pdf.query('complete').copy()
tdf = tdf.loc[list(pdf.index)]
# assert all(tdf.reset_index().wid.value_counts() == 25)

mdp2var = {}
for x in tdf[['variance', 'mdp']].itertuples():
    mdp2var[x.mdp] = x.variance

figs = Figures(f'figs/{VERSION}')
figs.add_names({
    'backward': 'Proportion Planning Backward',
    'BestFirstNoBestNext': 'Satisficing\nBestFirst',
    'BestFirst': 'Adaptive\nBestFirst'
})
figure = figs.figure; show = figs.show; figs.watch()
write_tex = TeX(path=f'stats/{EXPERIMENT}').write

tdf['i'] = list(tdf.trial_index - tdf.trial_index.groupby('wid').min() + 1)
assert all(tdf.groupby(['wid', 'i']).apply(len) == 1)
trial_features = pd.read_json(f'model/results/{VERSION}/trial_features.json').set_index(['wid', 'i'])
tdf = tdf.join(trial_features, on=['wid', 'i'])

if EXPERIMENT == 1:
    variances = ['constant']
    MODELS = ['RandomSelection', 'Optimal', 'OptimalPlus', 'BestFirst', 'BestFirstNoBestNext']
else:
    MODELS = 'Optimal BestFirst BreadthFirst DepthFirst '.split()
    variances = ['decreasing', 'constant', 'increasing']

keep = tdf.groupby('wid').n_click.mean() >= 1
tdf = tdf.loc[keep]
pdf = pdf.loc[keep]

%load_ext rpy2.ipython

# %% --------
lb, db, lg, dg, lr, dr, lo, do, *_ = sns.color_palette("Paired")
gray = (0.5, 0.5, 0.5)
pal = [gray, db, lb, dg, lg]
palette = dict(zip(MODELS, pal))
palette['Human'] = '#333333'
# %% ==================== PAYMENT ====================

pdf['total_time'] = total_time = (pdf.time_end - pdf.time_start) / 1000 / 60
sns.distplot(pdf.total_time)
m = pdf.total_time.median()
plt.axvline(m)
plt.title(f'Median time: {m:.2f} minutes')
show()

# %% --------
pdf['total_time'] = total_time = (pdf.time_end - pdf.time_start) / 1000 / 60
pdf['instruct_time'] = (pdf.time_instruct - pdf.time_start) / 60000
pdf['test_time'] = (pdf.time_end - pdf.time_instruct) / 60000

pdf.total_time.mean()
pdf.instruct_time.median()
print(pdf.groupby('click_delay').total_time.mean())
print(pdf.groupby('click_delay').final_bonus.mean())
print(pdf.groupby('click_delay').n_click.mean())

# %% --------
pdf.final_bonus.loc[lambda x: x>0].median()
pdf

wage = 60 * (bonus + base_pay) / pdf.total_time
sns.distplot(wage)
m = wage.median()
plt.axvline(m)
plt.title(f'Median wage: ${m:.2f} per hour')
show()

# %% --------

pdf['wage'] = wage
sns.catplot('click_delay', 'wage', data=pdf, kind='swarm',
           order='1.0s 2.0s 3.0s 4.0s'.split())
show()


# %% ==================== PARETO FRONT ====================

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

def setup_variance_plot(nrow=1):
    ncol = len(variances)
    return plt.subplots(nrow, ncol, figsize=(4*ncol,4), squeeze=False)

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

# %% ==================== MODEL COMPARISON ====================

fits = load_fits(VERSION, MODELS)
fits = fits.join(pdf[['variance', 'click_delay']], on='wid')
pdf['cost'] = fits.query('model == "Optimal"').set_index('wid').cost.clip(upper=5)

cf = pd.read_json(f'model/results/{VERSION}/click_features.json').set_index('wid')
res = cf.apply(lambda d: {k: p[d.c] for k, p in d.predictions.items()}, axis=1)
logp = np.log(pd.DataFrame(list(res.values)))[MODELS]
logp.set_index(cf.index, inplace=True)
logp['variance'] = pdf.variance
logp['Random'] = np.log(cf.p_rand)
assert set(MODELS) < set(logp.columns)
# %% --------

@figure()
def bbd_individual_likelihood():
    def plot_participants(val, fits, MODELS):
        sns.swarmplot(y='model', x=val, data=fits, order=MODELS,
                      palette=palette)
        for w, d in fits.groupby('wid'):
            # c = palette[pdf.click_delay[w]]
            c = 'k'
            plt.plot(d.set_index('model').loc[MODELS][val], MODELS, color=c, lw=2, alpha=0.5)
        plt.ylabel('')
        plt.xlabel('Log Likelihood')

    X = fits.set_index('variance')

    fig, axes = plt.subplots(len(variances), 1, figsize=(8,4*len(variances)))
    for i, v in enumerate(variances):
        if i != 0:
            plt.legend().remove()
        try:
            plt.sca(axes.flat[i])
        except:
            pass
        plot_participants('cv_nll', X.loc[v], MODELS)
        plt.title(f'{v.title()} Variance')
        if i != len(variances) - 1:
            plt.xlabel('')

# %% --------

def plot_models(L, ylabel, axes=None, title=True):
    L = L.copy()
    if axes is None:
        fig, axes = setup_variance_plot()
    for i, v in enumerate(variances):
        plt.sca(axes.flat[i])

        # L.loc[v].plot.line(color=[f'C{i}' for i in range(len(MODELS))], rot=30)
        plt.axhline(L.pop('RandomSelection').loc[v], c='k')
        L.loc[v].plot.bar(color=[f'C{i}' for i in range(len(MODELS))], rot=30)
        
        plt.xlabel('')
        if i == 0:
            plt.ylabel(ylabel)
        if len(variances) > 1 and title:
            plt.title(f'{v.title()} Variance')

@figure()
def average_predictive_accuracy(axes=None):
    # fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(6,4))
    plot_models(
        np.exp(logp.groupby(['variance', 'wid']).mean()).groupby('variance').mean(),
        "Average Predictive Accuracy",
        axes
    )

# %% --------
@figure()
def individual_predictive_accuracy():
    L = np.exp(logp.groupby('wid').mean())
    L = L.loc[keep]

    lm = L.mean().loc[MODELS]
    plt.scatter(lm, lm.index, s=100, color=[palette[m] for m in MODELS]).set_zorder(20)

    sns.stripplot(y='Model', x='value',
        data=pd.melt(L, var_name='Model'),
        order=MODELS,  jitter=False, 
        palette=palette,
        alpha=0.1)

    for wid, row in L.iterrows():
        # c = palette[pdf.click_delay[w]]
        c = 'k'
        plt.plot(row.loc[MODELS], MODELS, color=c, lw=1, alpha=0.1)
    plt.xlabel('Predictive Accuracy')

# %% --------
@figure()
def full_likelihood(axes=None):
    plot_models(
        -logp.groupby('variance').sum(),
        'Cross Validated\nNegative Log-Likelihood',
        axes
    )

@figure()
def geometric_mean_likelihood(axes=None):
    plot_models(
        np.exp(logp.groupby('variance').mean()),
        "Geometric Mean Likelihood",
        axes
    )

# %% --------

fits.query('variance == "increasing"').set_index(['wid', 'model']).sort_index()

@figure()
def pareto_fit(reformat_legend=False):
    X = tdf.reset_index().set_index('variance')
    L = np.exp(logp.groupby(['variance', 'wid']).mean()).groupby('variance').mean()
    fig, axes = plt.subplots(2, 3, figsize=(12,8))
    for i, v in enumerate(variances):
        plt.sca(axes[0, i])
        for model in MODELS:
            plot_model(v, model, title=False)
            
        # g = X.loc[v].groupby('wid'); x = 'n_click'; y = 'term_reward'
        # plt.errorbar(g[x].mean(), g[y].mean(), yerr=g[y].sem(), xerr=g[x].sem(), 
        #              label='Human', fmt='.', color='#333333', elinewidth=.5)

        plt.title(f'{v.title()} Variance')
        plt.ylabel("Expected Reward")
        plt.xlabel("Number of Clicks")

        plt.sca(axes[1, i])
        L.loc[v].plot.bar(color=[f'C{i}' for i in range(len(MODELS))], rot=30)
        plt.xlabel('')
        plt.ylabel("Average Predictive Accuracy")



# %% ==================== BACKWARD ====================

leaves = {20, 15, 10, 5}
tdf['first_click'] = tdf.clicks.apply(lambda x: x[0] if x else 0)
tdf['backward'] = tdf.first_click.isin(leaves)
pdf['backward'] = tdf.groupby('wid').backward.mean()

@figure(reformat_labels=True)
def backwards():
    sns.swarmplot('variance', 'backward', data=pdf)
    plt.xlabel('a')

# %% ==================== BEST FIRST ====================
best_first = get_result(VERSION, 'bestfirst.json')
bfo = pd.Series(best_first['optimal'])
bfo.index = bfo.index.astype(float)
bfo = bfo.sort_index().iloc[:-1]  # drop 100
pdf['best_first'] = pd.Series(best_first['human'])


def mean_std(x, digits=1, pct=False):
    if pct:
        x *= 100
        return fr'{x.mean().round(digits)}\% \pm {x.std().round(digits)}\%'
    else:
        return fr'{x.mean().round(digits)} \pm {x.std().round(digits)}'

write_tex("best_first", f"{pdf.best_first.mean()*100:.1f}\\%")

# %% --------
rdf = pdf[['cost', 'best_first']]
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
%load_ext rpy2.ipython

# %% --------
%%R -i rdf
summary(lm(best_first ~ cost, data=rdf))


# %% --------o
ut.head()

def write_lm_var(model, var, name):
    beta = np.round(model.params[var], 2)
    se = np.round(model.bse[var], 2)
    p = model.pvalues[var]
    if p <.001:
        p_desc = 'p < 0.001'
    elif p < .01:
        p_desc = 'p = {}'.format(np.round(p, 3))
    else :
        p_desc = 'p = {}'.format(np.round(p, 3))

    writevar('{}_BETA'.format(name), beta)
    writevar('{}_SE'.format(name), se)
    writevar('{}_P'.format(name), p)
    
    writevar(
        '{}_RESULT'.format(name),
        r'$\\beta = %s,\\ \\text{SE} = %s,\\ %s$' % (beta, se, p_desc)
    )

# %% --------

@figure()
def cost_best_first():
    bfo.plot(label="Optimal", color=palette["Optimal"], lw=2)
    # sns.regplot('cost', 'best_first', lowess=True, data=pdf, color=palette["Human"])
    plt.scatter('cost', 'best_first', data=pdf, color=palette["Human"])
    plt.ylabel('Proportion of Clicks on Best Path')
    plt.xlabel('Click Cost')
    plt.legend()


# %% --------
"""
- perecent best first (human/optimal)
- errors broken down by termination
    - terminate early vs late?
- action error rate
- interaction for adaptive satisficing
"""

# %% ==================== TERMINATION ====================
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

