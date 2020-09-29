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
# @figure()
def exp2_make_base():
    label_offset = -0.4
    fig, axes = plt.subplots(4, 3, figsize=(12, (4+3+3+3)), constrained_layout=True,
                             gridspec_kw={'height_ratios': [4,3,3,3]})

    # letter labels
    for char, ax in zip('ABCDEFG', axes[:, 0]):
        ax.annotate(char, (label_offset, 1), xycoords='axes fraction', size=32, va='bottom')
    # task image
    for v, ax in zip(VARIANCES, axes[0, :]):
        ax.set_title(f'{v.title()} Variance', fontdict=dict(fontsize=20))
        # ax.imshow(task_image(v))
        ax.axis('off')
    plot_pareto(axes[1, :], legend=False, fit_reg=False)
    for i, ax in enumerate(axes[1, :]):
        ax.set_ylim(-1, 25)
        if i != 0:
            ax.set_yticks([])
    plot_average_predictive_accuracy(axes[2, :])
    plot_second_click(axes[3, :])
    axes[3,0].legend().remove()
    for i, ax in enumerate(axes[3, :]):
        if i != 0:
            ax.set_yticks([])

    plt.savefig('fighist/exp2a.png', dpi=figs.dpi, bbox_inches='tight')

exp2_make_base()

# %% --------
def exp2_add_task():
    base = Image.open('fighist/exp2a.png')
    w, h = base.size

    task = task_image('decreasing')
    wt, ht = task.size
    scaling = 0.19 * w / wt
    task = task.resize((int(wt * scaling), int(ht * scaling)))

    base.paste(task, (400, 110))

    dt = datetime.now().strftime('%m-%d-%H-%M-%S')
    base.save(f'fighist/{dt}-exp2b.png')
    base.save('figs/4/exp2_main.png')


# # exp2_make_base()
# exp2_add_task()
    

# %% --------
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

# %% --------
@figure()
def exp2_main_alt():
    fig, axes = setup_variance_plot(4)
    for v, ax in zip(VARIANCES, axes[0, :]):
        ax.imshow(task_image(v))
        ax.axis('off')
    plot_second_click(axes[1, :], 
        # models=['OptimalPlus', 'BestFirst']
        )
    plot_pareto(axes[2, :], legend=False, fit_reg=False)
    plot_average_predictive_accuracy(axes[3, :])


# %% --------

@figure()
def exp3_main():
    fig, axes = setup_variance_plot(2, height=3, label_offset=-0.5)
    first_click_depth(axes[0])
    plot_average_predictive_accuracy(axes[1])

# %% --------

def exp4_make_base():
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))
    gs = fig.add_gridspec(2, 6)
    
    plt.sca(fig.add_subplot(gs[:, 0:4]))
    # img = Image.open(f'imgs/roadtrip.png')
    # plt.imshow(img)
    plt.axis('off')
    plt.annotate('A', (-0.4, 1), xycoords='axes fraction', size=32, va='bottom')
    
    ax = fig.add_subplot(gs[0, 4:6])
    plot_average_predictive_accuracy(np.array(ax))
    ax.annotate('B', (-0.4, 1), xycoords='axes fraction', size=32, va='bottom')
    
    ax = fig.add_subplot(gs[1, 4:6])
    plt.sca(ax)
    expansion_value()
    ax.annotate('C', (-0.4, 1), xycoords='axes fraction', size=32, va='bottom')
    
    plt.savefig('fighist/exp4a.png', dpi=figs.dpi, bbox_inches='tight')

def exp4_add_task():
    base = Image.open('fighist/exp4a.png')
    w, h = base.size

    task = Image.open('imgs/roadtrip.png')
    wt, ht = task.size
    scaling = 0.66 * w / wt
    task = task.resize((int(wt * scaling), int(ht * scaling)))

    base.paste(task, (0, 100))

    dt = datetime.now().strftime('%m-%d-%H-%M-%S')
    base.save(f'fighist/exp4b-{dt}.png')
    base.save('figs/4/exp4_main.png')

exp4_make_base()
exp4_add_task()




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
from statsmodels.stats.proportion import proportion_confint
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

