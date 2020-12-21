# %% ==================== Load predictions ====================
preds = pd.DataFrame(get_result(VERSION, 'predictions.json')).set_index('wid')
res = preds.apply(lambda d: {k: p[d.c] for k, p in d.predictions.items()}, axis=1)
logp = np.log(pd.DataFrame(list(res.values)))
logp.set_index(preds.index, inplace=True)
logp['variance'] = pdf.variance
logp = logp.loc[list(pdf.index)]
logp = logp.reset_index().set_index(['variance', 'wid'])
total = logp.sum()

assert set(logp.reset_index().wid) == set(pdf.index)

# %% ==================== Choose models to plot ====================

MODELS = list(BASIC)

best_heuristic = [
    total.filter(regex=f'^{model_class}*').idxmax()
    for model_class in HEURISTICS
]
best_heuristic_no_depthlimit = [
    total.filter(regex=f'^{model_class}(_Satisfice)?(_BestNext)?(_Prune)?$').idxmax()
    for model_class in ['Breadth', 'Depth', 'Best']
]

# full_heuristic = [f'{base}_Satisfice_BestNext_DepthLimit_Prune'
#     for base in ['Breadth', 'Depth', 'Best']]

# foo = [f'{base}_Satisfice_BestNext'
#     for base in ['Breadth', 'Depth', 'Best']]

if EXPERIMENT == 2:
    MODELS.extend(best_heuristic_no_depthlimit)
elif EXPERIMENT == 3:
    MODELS.extend(['Expand', 'MetaGreedyExpand', 'OptimalPlusExpand'])
    no_expand = [m.replace('_Expand', '') for m in best_heuristic]
    MODELS.extend(no_expand)
    MODELS.extend(m + '_Expand' for m in no_expand)
else:
    MODELS.extend(best_heuristic)

if EXPERIMENT == 1:
    MODELS.extend([
        'Best_Satisfice_BestNext_DepthLimit_Prune',
        'Best_BestNext',
        'Best_Satisfice',
        'Best_DepthLimit',
        'Best_Prune',
        'Best'
    ])


    if EXPERIMENT == 3:
    MODELS = [m.replace('_Expand', '') for m in MODELS]
    with_expand = [m + '_Expand' for m in MODELS]
    with_expand[0] = 'Expand'  # Random_Expand
    MODELS.extend(with_expand)

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

        plt.xlabel(label.replace('\n', ' '))
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
    
    # def geometric_mean(x):
        # return np.exp(np.log(x).mean())
    # sns.barplot(data=np.exp(logp[MODELS]), palette=pal, estimator=geometric_mean)

    # up = [i for i, model in enumerate(x.index) if def_better[model]]
    # plt.scatter(up, x.iloc[up] + 0.03, marker='^', c='k', alpha=0.8, s=15)

    # all_better_or_worse = (def_better | def_worse).drop('OptimalPlus').all()
    # if not all_better_or_worse:
    #     down = [i for i, model in enumerate(x.index) if def_worse[model]]
    #     plt.scatter(down, x.iloc[down] + 0.03, marker='_', c='k', alpha=0.8, s=15)

    plt.xticks(rotation=35, ha="right")
    figs.reformat_ticks()
    plt.ylabel(label)

def plot_model_performance_expansion(L, label, axes=None):
    # L = L.groupby('variance').mean()
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

        plt.xlabel(label.replace('\n', ' '))
        if i == 0:
            figs.reformat_ticks(yaxis=True)
        else:
            plt.yticks(())

@figure()
def plot_geometric_mean_likelihood(axes=None):
    plot_model_performance(
        np.exp(logp.groupby('variance').mean()),
        'Geometric Mean\nLikelihood',
        # 'Predictive Accuracy',
        axes
    )

# # %% --------
# @figure()
# def plot_full_likelihood(axes=None):
#     plot_model_performance(
#         -logp.groupby('variance').sum(),
#         'Cross Validated\nNegative Log-Likelihood',
#         axes
#     )

# # %% --------

# @figure()
# def plot_average_predictive_accuracy(axes=None):
#     plot_model_performance(np.exp(logp.groupby(['variance', 'wid']).mean()),
#         'Predictive Accuracy', axes)
# # %% --------

@figure()
def individual_predictive_accuracy():
    models = MODELS
    fig = plt.figure(figsize=(8,4))
    L = np.exp(logp.groupby('wid').mean())
    L = L.loc[pdf.index]

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



# %% ==================== Stats ====================

def get_best_with_delta(lp):
    top2 = lp.sort_values(ascending=False).iloc[:2]
    return top2.index[0], top2.iloc[0] - top2.iloc[1]

def fmt_min_delta(x):
    assert x > 0
    return rf'all $\dnll > {np.floor(x):.0f}$'

if EXPERIMENT == 1:
    opt = total.OptimalPlus
    no_opt = total.drop('OptimalPlus')
    no_best = no_opt.loc[~no_opt.index.str.startswith('Best')]
    write_tex('dnll_no_best', fmt_min_delta(opt - no_best.max()))

    no_best_next = total.loc[~total.index.str.contains('BestNext')]
    best_no_best_next = no_best_next.drop('OptimalPlus').max()
    write_tex('dnll_no_bestnext', fmt_min_delta(opt - best_no_best_next.max()))

    write_tex('dnll_best', rf'$\dnll = {opt - total.drop("OptimalPlus").max():.0f}$')

    assert total.drop('OptimalPlus').idxmax() == 'Best_Satisfice_BestNext_DepthLimit'
    delta = total.Best_Satisfice_BestNext_DepthLimit_Prune - total.Best_Satisfice_BestNext_DepthLimit
    write_tex('dnll_prune', rf'$\dnll = {delta:.0f}$')

    THRESHOLD = np.log(100)
    def_better = total - total.OptimalPlus > THRESHOLD
    def_worse = total - total.OptimalPlus < -THRESHOLD

    # write_tex('min_dnll_plot', f'{total[MODELS].sort_values().diff().min():.0f}')
    total[MODELS].sort_values().diff()
    if total[MODELS].sort_values().diff().min() < THRESHOLD:
        print("ERROR: warn_small_dnll")
        write_tex('warn_small_dnll', r'\red{WARNING: SMALL DIFFERENCE IN LIKELIHOOD}')
    else:
        write_tex('warn_small_dnll', '')

    best_fit = logp.groupby('wid').sum().T.idxmax()
    pdf['best_fit'] = best_fit
    pdf['best_fit_class'] = best_fit.apply(lambda x: x.split('_')[0])
    pdf.groupby('variance').best_fit_class.value_counts()
    nbf = best_fit.apply(lambda x: x.split('_')[0]).value_counts()

    write_tex(f'nfit_best', nbf.Best)
    write_tex(f'nfit_optimal', nbf.OptimalPlus)
    write_tex(f'nfit_other', nbf.drop(['Best', 'OptimalPlus']).sum())

if EXPERIMENT == 2:
    # heuristics only
    deltas = []
    total = logp.groupby('variance').sum()
    for v in VARIANCES:
        predicted = {'decreasing': 'Breadth', 'constant': 'Best', 'increasing': 'Depth'}[v]
        non_opt = total[MODELS].drop('OptimalPlus', axis=1).T
        lp = non_opt[v]
        top2 = lp.sort_values(ascending=False).iloc[:2]
        assert top2.index[0].startswith(predicted)
        deltas.append(top2.iloc[0] - top2.iloc[1])

    write_tex('heuristic_min_dnll', fmt_min_delta(min(deltas)))

    # best fit without restrictions
    reported_best = dict(zip(VARIANCES, ['OptimalPlus', 'Best_BestNext', 'OptimalPlus']))
    for v in VARIANCES:
        lp = total.loc[v]
        best_model, delta = get_best_with_delta(lp)
        assert best_model == reported_best[v]
        if best_model != 'OptimalPlus':
            delta = lp[best_model] - lp['OptimalPlus']
        write_tex(f'total_dnll_{v}', rf'$\dnll = {delta:.0f}$')



# # %% --------
# best_fit = logp.groupby('wid').sum().T.idxmax()
# pdf['best_fit'] = best_fit
# pdf['best_fit_class'] = best_fit.apply(lambda x: x.split('_')[0])
# pdf.groupby('variance').best_fit_class.value_counts()


# best_fit = logp.groupby('wid').sum().T.drop('OptimalPlus').idxmax()
# pdf['best_fit'] = best_fit
# pdf['best_fit_class'] = best_fit.apply(lambda x: x.split('_')[0])
# pdf.groupby('variance').best_fit_class.value_counts()
# # %% --------
# logp.groupby('variance').mean().T.idxmax()



# %% ==================== Build table ====================
BASIC = ['Random', 'MetaGreedy', 'OptimalPlus']
HEURISTICS = ['Breadth', 'Depth', 'Best']
MECHANISMS = ['Satisfice', 'BestNext', 'DepthLimit', 'Prune']

param_counts = get_result(VERSION, 'param_counts.json')

best_fit = logp.groupby('wid').sum().T.idxmax()
nbf = best_fit.value_counts()

def make_row(model):
    row = {'Class': model.split('_')[0].replace('Plus', '')}
    for mech in MECHANISMS:
        row[mech] = mech in model
    row['# Params'] = param_counts[model]
    row['# Participants'] = nbf.get(model, 0)
    row['Total NLL'] = -int(round(total[model]))

    return row

# h = HEURISTICS[0]
# models = list(total.index[total.index.str.startswith(h)])
models = list(total.index)
t = pd.DataFrame(make_row(model) for model in models)
t.Class = pd.Categorical(t.Class, ['Random', 'MetaGreedy', 'Optimal', *HEURISTICS])
t = t.sort_values(['Class', 'Satisfice', 'BestNext', 'DepthLimit', 'Prune'])

write_tex('modelcomp_table', t.to_latex(index=False, 
    formatters={
        mech: lambda x: {False: '', True: 'X'}[x]
    for mech in MECHANISMS}))


# %% ==================== Old figures ====================



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