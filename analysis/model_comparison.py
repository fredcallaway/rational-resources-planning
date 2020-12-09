# %% ==================== Load predictions ====================
%run setup 1
preds = pd.DataFrame(get_result(VERSION, 'predictions.json')).set_index('wid')
res = preds.apply(lambda d: {k: p[d.c] for k, p in d.predictions.items()}, axis=1)
logp = np.log(pd.DataFrame(list(res.values)))
logp.set_index(preds.index, inplace=True)
logp['variance'] = pdf.variance
logp = logp.loc[keep]
logp = logp.reset_index().set_index(['variance', 'wid'])

# geometric mean likelihood
gml = np.exp(logp.groupby(['variance', 'wid']).mean())
mean_gml = gml.mean()
total = logp.sum()
total.sort_values()

total.OptimalPlus - total.Best_Satisfice_DepthLimit_Prune
total.max() - total.OptimalPlus

mean_gml.sort_values()

# %% ==================== Build table ====================
HEURISTICS = ['Breadth', 'Depth', 'Best']
MECHANISMS = ['Satisfice', 'BestNext', 'DepthLimit', 'Prune']
h = HEURISTICS[0]

param_counts = get_result(VERSION, 'param_counts.json')

def make_row(model):
    row = {m: m in model for m in MECHANISMS}
    row['# Params'] = param_counts[model]
    row['Total NLL'] = -int(round(total[model]))
    return row


models = list(total.index[total.index.str.startswith(h)])
t = pd.DataFrame(make_row(model) for model in models)
t = t.sort_values(['Satisfice', 'BestNext', 'DepthLimit', 'Prune'])
t.to_latex

total.sort_values()

# %% ==================== Choose models to plot ====================

MODELS = ['Random', 'MetaGreedy', 'OptimalPlus']
HEURISTIC = ['Breadth', 'Depth', 'Best']

best_heuristic = [
    mean_gml.filter(regex=f'^{model_class}*').idxmax()
    for model_class in ['Breadth', 'Depth', 'Best']
]
best_heuristic_simple = [
    # mean_gml.filter(regex=f'^{model_class}(_Satisfice)?(_BestNext)?(_Prune)?$').idxmax()
    mean_gml.filter(regex=f'^{model_class}(_Satisfice)?(_BestNext)?$').idxmax()
    for model_class in ['Breadth', 'Depth', 'Best']
]
full_heuristic = [f'{base}_Satisfice_BestNext_DepthLimit_Prune'
    for base in ['Breadth', 'Depth', 'Best']]

foo = [f'{base}_Satisfice_BestNext'
    for base in ['Breadth', 'Depth', 'Best']]

# MODELS.extend(best_heuristic)
print(best_heuristic_simple)
MODELS.extend(best_heuristic_simple)
if EXPERIMENT == 1:
    MODELS.extend([
        'Best_Satisfice_BestNext_DepthLimit_Prune',
        'Best_BestNext',
        'Best_Satisfice',
        'Best_DepthLimit',
        'Best_Prune',
        'Best'
    ])
# %% --------
for model_class in HEURISTIC:
    x = mean_gml.filter(regex=f'^{model_class}*')
    print(x.max() - x.loc[~x.index.str.contains('Prune')].max())


# %% ==================== Stats ====================
opt = total.OptimalPlus
no_opt = total.drop('OptimalPlus')
no_best = no_opt.loc[~no_opt.index.str.startswith('Best')]

write_tex('dnll_no_best', rf'all $\dnll > {opt - no_best.max():.0f}$')

no_best_next = total.loc[~total.index.str.contains('BestNext')]
best_no_best_next = no_best_next.drop('OptimalPlus').max()
write_tex('dnll_no_bestnext', rf'all $\dnll > {opt - best_no_best_next.max():.0f}$')

write_tex('dnll_best', rf'$\dnll = {total.max() - opt:.0f}$')

assert total.idxmax() == 'Best_Satisfice_BestNext_DepthLimit'
delta = total.Best_Satisfice_BestNext_DepthLimit_Prune - total.Best_Satisfice_BestNext_DepthLimit
write_tex('dnll_prune', rf'$\dnll = {delta:.0f}$')

total.loc[total.index.str.startswith('Best')].sort_values()


# %% ==================== Plots ====================

def plot_model_performance(L, label, axes=None):
    if EXPERIMENT == 1:
        return plot_model_performance_vertical(L, label, axes)
    if EXPERIMENT >= 3:
        return plot_model_performance_expansion(L, label, axes)
    if axes is None:
        L = L.groupby('variance').mean()
        fig, axes = setup_variance_plot()
    for i, v in enumerate(VARIANCES):
        plt.sca(axes.flat[i])
        pal = [palette[m] for m in MODELS]
        L.loc[v].loc[MODELS].plot.barh(color=pal)
        
        plt.xlabel(label)
        if i != 0:
            plt.yticks(())
    figs.reformat_ticks(yaxis=True, ax=axes.flat[0])


def plot_model_performance_vertical(L, label, ax=None):
    if ax is None:
        plt.figure(figsize=(8,4))
        ax = plt.gca()
    else:
        plt.sca(ax)
    pal = [palette[m] for m in MODELS]
    for i in range(6, len(pal)):
        pal[i] = lg

    x = L.groupby('variance').mean().loc['constant'].loc[MODELS]

    x.plot.bar(color=pal)

    stars = [i for i, k in enumerate(x.index) 
        if k != 'OptimalPlus' and ttest_ind(L.OptimalPlus, L[k]).pvalue < .05]
    plt.scatter(stars, x.iloc[stars] + 0.03, marker='*', c='k')
    plt.xticks(rotation=45, ha="right")

    figs.reformat_ticks()
    plt.ylabel(label)

def plot_model_performance_expansion(L, label, axes=None):
    L = L.groupby('variance').mean()
    if axes is None:
        if EXPERIMENT == 4:
            fig, axes = plt.subplots(1, 1, figsize=(6,4), squeeze=False)
        else:
            fig, axes = setup_variance_plot()
    
    top_models = [m for m in MODELS if not m.endswith('Expand')]
    bottom_models = [m for m in MODELS if m.endswith('Expand')]
    # del top_models[0]
    for i, v in enumerate(VARIANCES):
        plt.sca(axes.flat[i])

        for models in [bottom_models, top_models]:
            pal = [palette[m] for m in models]
            ax = L.loc[v].loc[models].plot.barh(color=pal)

        plt.xlabel(label)
        plt.xlim(0, 0.33)
        if i == 0:
            figs.reformat_ticks(yaxis=True)
        else:
            plt.yticks(())

@figure()
def plot_average_predictive_accuracy(axes=None):
    plot_model_performance(gml, 'Predictive Accuracy', axes)

# %% --------
@figure()
def plot_full_likelihood(axes=None):
    plot_model_performance(
        -logp.groupby('variance').sum(),
        'Cross Validated\nNegative Log-Likelihood',
        axes
    )
# %% --------
@figure()
def plot_full_predictive_accuracy(axes=None):
    plot_model_performance(
        np.exp(logp.groupby('variance').mean()),
        'Predictive Accuracy',
        axes
    )
# %% --------

@figure()
def individual_predictive_accuracy():
    models = MODELS
    fig = plt.figure(figsize=(8,4))
    L = np.exp(logp.groupby('wid').mean())
    L = L.loc[keep]

    lm = L.mean().loc[models]
    plt.scatter(lm, lm.index, s=100, color=[palette[m] for m in models]).set_zorder(20)

    sns.stripplot(y='Model', x='value',
        data=pd.melt(L, var_name='Model'),
        order=models,  jitter=False, 
        palette=palette,
        alpha=0.1)

    for wid, row in L.iterrows():
        # c = palette[pdf.click_delay[w]]
        c = 'k'
        plt.plot(row.loc[models], models, color=c, lw=1, alpha=0.1)
    plt.xlabel('Predictive Accuracy')
    plt.ylabel('')
    figs.reformat_ticks(yaxis=True)

# %% --------

# @figure()
# def bbd_individual_likelihood():
#     def plot_participants(val, fits, MODELS):
#         sns.swarmplot(y='model', x=val, data=fits, order=MODELS,
#                       palette=palette)
#         for w, d in fits.groupby('wid'):
#             # c = palette[pdf.click_delay[w]]
#             c = 'k'
#             plt.plot(d.set_index('model').loc[MODELS][val], MODELS, color=c, lw=2, alpha=0.5)
#         plt.ylabel('')
#         plt.xlabel('Log Likelihood')

#     X = fits.set_index('variance')

#     fig, axes = plt.subplots(len(VARIANCES), 1, figsize=(8,4*len(VARIANCES)))
#     for i, v in enumerate(VARIANCES):
#         if i != 0:
#             plt.legend().remove()
#         try:
#             plt.sca(axes.flat[i])
#         except:
#             pass
#         plot_participants('cv_nll', X.loc[v], MODELS)
#         plt.title(f'{v.title()} Variance')
#         if i != len(VARIANCES) - 1:
#             plt.xlabel('')

# # %% --------


# @figure()
# def geometric_mean_likelihood(axes=None):
#     plot_model_performance(
#         np.exp(logp.groupby('variance').mean()),
#         "Geometric Mean Likelihood",
#         axes
#     )

# # %% --------