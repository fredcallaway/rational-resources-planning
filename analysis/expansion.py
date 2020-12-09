# %% ==================== expansion rate by participant ====================
from statsmodels.stats.proportion import proportion_confint

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

human_rate = load_cf('Human').query('variance == "constant" and not is_term').groupby('wid').expand.mean()
write_tex(f'expansion_Human', mean_std(human_rate, fmt='pct', digits=0))

model_rate = load_cf('OptimalPlus').query('variance == "constant" and not is_term').expand.mean()
write_tex('expansion_OptimalPlus', f'{model_rate*100:.1f}\\%')

model_rate
# %% --------
cfh = load_cf('Human').query('variance == "constant" and not is_term')
cfo  = load_cf('OptimalPlus').query('variance == "constant" and not is_term')

# %% --------
from statsmodels.stats.proportion import proportions_ztest
z, p = proportions_ztest(cfh.expand.sum(), len(cfh.expand), value=model_rate)
write_tex('expand_proportion', rf'$z={z:.2f},\ {pval(p)}$')
    

prop_p = cfh.groupby('wid').expand.apply(
    lambda x: proportions_ztest(x.sum(), len(x), value=model_rate, alternative='larger')[1]
)
(prop_p < .05).sum()

cfh.groupby('wid').expand.mean()

# %% ==================== logistic curve ====================
edf = pd.DataFrame(get_result(VERSION, 'expansion.json')).set_index('wid')
edf['gain'] = edf.q_jump - edf.q_expand
edf['jump'] = ~edf['expand']

write_tex("jump", mean_std(edf.groupby('wid').jump.mean()*100, fmt='pct'))

# write_tex("best_first", mean_std(pdf.best_first, fmt='pct'))

@figure()
def expansion_value():
    sns.regplot('gain', 'jump', data=edf, logistic=True, x_bins=np.linspace(-0.75, 0.75, 5), color='black')
    plt.xlabel('Value of Violating\nForward Search')
    plt.ylabel('Probability of Violating\nForward Search')

# %% --------
rdf = edf[['gain', 'jump']].reset_index().dropna()
rdf.jump = rdf.jump.astype(int)
rdf.gain -= rdf.gain.mean()
rdf.gain /= rdf.gain.std()
# %% --------
%%R -i rdf
library(lme4)
library(lmerTest)
m = glmer(jump ~ gain + (1|wid), family=binomial, data=rdf)
print(summary(m));


# %% ==================== histogram ====================

for k, d in edf.groupby('expand'):
    sns.distplot(d.gain, label='frontier' if k else 'not frontier', kde=0, norm_hist=True)

plt.legend()
show()

# %% ==================== bar ====================
sns.barplot('jump', 'gain', data=edf)
show()


# %% --------

pdf['expanding'] = cf.groupby('wid')['expanding'].mean()
sns.swarmplot('variance', 'expanding', data=pdf, size=3)
plt.ylabel('Expansion Rate')
show()
# %% --------


# %% --------
# for w, x in :
    # proportion_confint(x.sum(), len(x))
    # cf.groupby('wid')

# cf = pd.DataFrame(get_result(VERSION, 'click_features.json')) \
#     .set_index('wid').loc[keep]


x = load_cf('OptimalPlusPure')



fig, axes = setup_variance_plot()
x = pdf[['variance', 'n_click']].join(x.groupby('wid').expand.mean()).set_index('variance')
for ax, var in zip(axes.flat, VARIANCES):
    ax.scatter('n_click', 'expand', data=x.loc[var])

show()

