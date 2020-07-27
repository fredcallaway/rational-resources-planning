# %% ==================== compare cross validation ====================
rand_cv = pd.concat([pd.read_csv(f'model/results/{EXPERIMENT}-randomfolds/mle/{model}-cv.csv')
                     for model in MODELS], sort=False)

fits = fits.join(rand_cv.groupby(['model', 'wid']).test_nll.sum().rename('rand_nll'), on=['model', 'wid'])

# %% --------
g = sns.FacetGrid(row='variance', col='model', data=fits, aspect=1, margin_titles=True)
g.map(sns.scatterplot, 'cv_nll', 'rand_nll')

for ax in g.axes.flat:
    ax.plot([0, 0], [500, 500], c='k')
show()


# %% --------
MODELS = 'BreadthFirst DepthFirst BestFirst Optimal'.split()
fits = load_fits(exp, MODELS, path='mle')
fits = fits.join(pdf[['variance', 'click_delay']], on='wid')
pdf['cost'] = fits.query('model == "Optimal"').set_index('wid').cost.clip(upper=5)

cf = pd.read_json(f'model/results/{exp}/click_features.json').set_index('wid')
res = cf.apply(lambda d: {k: p[d.c] for k, p in d.predictions.items()}, axis=1)
logp = np.log(pd.DataFrame(list(res.values)))[MODELS]
logp.set_index(cf.index, inplace=True)
logp['variance'] = pdf.variance
# %% --------
r = tdf.state_rewards.iloc[0]

tdf['max_possible'] = tdf.state_rewards.apply(lambda r:
    max(sum(r[i:i+5]) for i in range(1, 17, 5))
)


indmax = tdf.groupby('wid').max_possible.mean()
achieved = tdf.groupby('wid').score.mean()
all(achieved <= indmax)

tdf.query('variance == "increasing"').groupby('wid').score.mean()