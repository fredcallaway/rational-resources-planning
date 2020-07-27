
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