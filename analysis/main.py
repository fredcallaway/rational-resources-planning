# %%
%run setup 3
# figs.nosave = True
if EXPERIMENT == 2:
    %run -i breadth_depth
%run -i pareto
%run -i model_comparison
figs.nosave = False


# %% ==================== Basic stuff ====================
write_tex("recruited", len(full_pdf))

write_tex("excluded", "TODO")
write_tex("incomplete", "TODO")
write_tex("N", len(pdf))

write_tex("bonus", mean_std(pdf.bonus, fmt='$'))
write_tex("time", mean_std(pdf['total_time']))

# %% --------

wage = 60 * (pdf.bonus + 1.50) / pdf.total_time
pdf['wage'] = wage
np.mean(pdf.wage < pdf.set_index('worker_id').loc['5f4912fc3c25512e73761c48'].wage)

# %% ==================== Main figures  ====================

@figure()
def exp2_main():
    fig, axes = setup_variance_plot(2)
    for v, ax in zip(VARIANCES, axes[0, :]):
        ax.imshow(task_image(v))
        ax.axis('off')
    plot_second_click(axes[1, :], 
        # models=['OptimalPlus', 'BestFirst']
        )

@figure()
def pareto_fit():
    fig, axes = setup_variance_plot(2, label_offset=-0.4)
    plot_pareto(axes[0, :], legend=False, fit_reg=False)
    plot_average_predictive_accuracy(axes[1, :])
    # figs.reformat_ticks(yaxis=True, ax=axes[1,0])


# %% ==================== BACKWARD ====================

tf = model_trial_features('OptimalPlusPure')
d = pd.DataFrame([pdf.variance, tf.groupby('wid').backward.mean()]).T
d.backward = d.backward.astype(int)
d.query('variance == "increasing"').join(pdf.query('variance == "increasing"').cost)

# %% --------
pdf['optimal_backward'] = model_trial_features('OptimalPlusPure').groupby('wid').backward.mean()

@figure(reformat_labels=True)
def backwards():
    sns.stripplot('variance', 'backward', data=pdf, order=VARIANCES,
        size=3, color=palette['Human'], alpha=0.5)
    sns.pointplot('variance', 'backward', data=pdf, order=VARIANCES, color=palette['Human'])
    plt.plot([0,1,2], [0, 0, 1], lw=2, ls='--', color=palette['Optimal'], label='Optimal')
    plt.plot([0,1,2], [0, 0, 0], lw=2, ls='--', color=palette['BreadthFirst'], label='Forward')
    plt.plot([0,1,2], [1, 1, 1], lw=2, ls='--', color=palette['DepthFirst'], label='Backward')
    plt.plot([0,1,2], [.5, .5, .5] , lw=2, ls='--', color=palette['RandomSelection'], label='Random')
    plt.xlabel('')

# %% --------

@figure(reformat_labels=True)
def backwards():
    sns.stripplot('variance', 'backward', data=pdf, order=VARIANCES,
        size=3, color=palette['Human'], alpha=0.5)

    for m in ['OptimalPlus']:
        y = model_trial_features(m).groupby('variance').backward.mean().loc[VARIANCES]
        y.plot(lw=2, ls='--', color=palette[m], label=figs.nice_name(m))
    
    sns.pointplot('variance', 'backward', data=pdf, order=VARIANCES, color=palette['Human'])

    plt.xlabel('')

# %% --------
opt_back = model_trial_features('OptimalPlus').groupby('wid').backward.mean()
opt_pure_back = model_trial_features('OptimalPlusPure').groupby('wid').backward.mean()

@figure()
def backwards_complex():
    fig, axes = setup_variance_plot()
    for ax, (_, d) in zip(axes.flat, tdf.groupby(['variance'])):
        plt.sca(ax)
        g = d.backward.groupby('wid')

        est = g.mean()
        lo, hi = proportion_confint(g.apply(sum), g.apply(len))
        err = np.vstack([(est - lo).values, (hi - est).values])
        idx = np.argsort(est)
        plt.errorbar(np.arange(len(est)), est[idx], yerr=err[:, idx], color=palette['Human'], label='Humans')
        plt.plot(np.arange(len(est)), opt_back[est.index][idx], color=palette['Optimal'], label='Optimal')
        plt.plot(np.arange(len(est)), opt_pure_back[est.index][idx], color=lb, label='Optimal')
        plt.xticks([])
        plt.xlabel("Participant")
        plt.ylabel('Backward Planning Rate')
        plt.ylim(-0.05, 1.05)
        plt.axhline(0, label='Forward Search', color=palette['BreadthFirst'])
        plt.axhline(1, label='Backward Search', color=palette['DepthFirst'])
    axes.flat[0].legend()


# %% ==================== EXPANSION ====================
from statsmodels.stats.proportion import proportion_confint

# cf = pd.DataFrame(get_result(VERSION, 'click_features.json')) \
#     .set_index('wid').loc[keep]


x = load_cf('OptimalPlusPure')




fig, axes = setup_variance_plot()
x = pdf[['variance', 'n_click']].join(x.groupby('wid').expand.mean()).set_index('variance')
for ax, var in zip(axes.flat, VARIANCES):
    ax.scatter('n_click', 'expand', data=x.loc[var])

show()



# %% --------

opt_exp = load_cf('OptimalPlus').groupby('wid').expand.mean()
opt_pure_exp = load_cf('OptimalPlusPure').groupby('wid').expand.mean()
cf = load_cf('Human')
@figure()
def expansion():
    fig, axes = setup_variance_plot()
    for ax, (_, d) in zip(axes.flat, cf.groupby(['variance'])):
        plt.sca(ax)
        g = d.expand.groupby('wid')
        est = g.mean()
        lo, hi = proportion_confint(g.apply(sum), g.apply(len))
        err = np.vstack([(est - lo).values, (hi - est).values])
        idx = np.argsort(est)
        plt.errorbar(np.arange(len(est)), est[idx], yerr=err[:, idx], color='k', label='Humans')
        
        plt.plot(np.arange(len(est)), opt_exp[est.index][idx], color=palette['Optimal'], label='Optimal')
        plt.plot(np.arange(len(est)), opt_pure_exp[est.index][idx], color=lb, label='Pure Optimal')

        plt.xticks([])
        plt.xlabel("Participant")
        plt.ylim(-0.05, 1.05)
        plt.ylabel('Expansion Rate')
        plt.axhline(1, label='Forward Search', color=palette['BreadthFirst'])
    axes.flat[0].legend()

# %% --------

pdf['expanding'] = cf.groupby('wid')['expanding'].mean()
sns.swarmplot('variance', 'expanding', data=pdf, size=3)
plt.ylabel('Expansion Rate')
show()
# %% --------

# for w, x in :
    # proportion_confint(x.sum(), len(x))
    # cf.groupby('wid')




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

cv_fits = pd.concat([pd.read_csv(f'../model/results/{VERSION}/mle/{model}-cv.csv') 
                     for model in MODELS], sort=False).set_index('wid')

cv_fits.query('model == "OptimalPlus"').columns
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


sns.distplot(pdf.n_click); show()
# %% ==================== CLICK DELAY ====================

@figure()
def click_delay_click():
    sns.swarmplot('click_delay', 'n_click', data=pdf,
        hue='variance',
        order='1.0s 2.0s 3.0s 4.0s'.split())

    plt.xlabel('Click Delay')
    plt.ylabel('Number of Clicks')

@figure()
def click_delay_time():
    sns.swarmplot('click_delay', 'test_time', data=pdf,
        hue='variance',
        order='1.0s 2.0s 3.0s 4.0s'.split())

    plt.xlabel('Click Delay')
    plt.ylabel('Test Time')



# %% ==================== LEARNING ====================

sns.lineplot('i', 'n_click', hue='variance', data=tdf)
show()

sns.lineplot('i', 'score', hue='variance', data=tdf)
show()

# %% ==================== BREADTH DEPTH ====================



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

