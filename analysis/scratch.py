
import hashlib
def hash_id(worker_id):
    return 'w' + hashlib.md5(worker_id.encode()).hexdigest()[:7]

wid = hash_id('5f292bd0f94a2428215919a1')


tdf.loc[wid]

30 * 7 - tdf.score.clip(lower=-30).groupby('wid').sum()

# %% --------
trial_bonus = (30 - tdf.score).clip(lower=0) / 1000
trial_bonus.groupby('wid').sum()
tdf.score
tdf.loc[wid]
# %% --------
@figure()
def exp2_big():
    fig, axes = setup_variance_plot(4, label_offset=-0.4)
    for v, ax in zip(VARIANCES, axes[0, :]):
        ax.imshow(task_image(v))
        ax.axis('off')

    plot_second_click(axes[1, :], models=['OptimalPlus', 'BestFirst'])
    plot_pareto(axes[2, :], legend=False, fit_reg=False)
    plot_average_predictive_accuracy(axes[3, :])
    figs.reformat_ticks(yaxis=True, ax=axes[2,0])

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