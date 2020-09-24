best_first = get_result(VERSION, 'bestfirst.json')

bfo = pd.Series(best_first['optimal'])
bfo.index = bfo.index.astype(float)
bfo = bfo.sort_index().iloc[:-1]  # drop 100
pdf['best_first'] = pd.Series(best_first['human'])

write_tex("best_first", mean_std(pdf.best_first, fmt='pct'))

# %% ==================== plot ====================

@figure()
def cost_best_first():
    plt.figure(figsize=(4,4))
    plt.axhline([1], label='BestFirst', color=palette['BestFirst'], lw=3)
    bfo.plot(label="Optimal", color=palette["Optimal"], lw=3)
    # sns.regplot('cost', 'best_first', lowess=True, data=pdf, color=palette["Human"])
    plt.scatter('cost', 'best_first', data=pdf, color=palette["Human"], label='Human')
    plt.ylabel('Proportion of Clicks on Best Path')
    plt.xlabel('Click Cost')
    plt.xlim(-0.05, 3.0)
    plt.legend()

# %% ==================== alt ====================

cf = load_cf('Human')
pdf['best_first'] = cf.is_best.groupby('wid').mean()
plt.scatter(pdf.cost, pdf.best_first)

cf = load_cf('OptimalPlusPure')
pdf['opt_best_first'] = cf.is_best.groupby('wid').mean()
plt.scatter(pdf.cost, pdf.opt_best_first)
plt.ylim(0, 1.)
show()

# %% --------
cf = load_cf('Human')
pdf['best_first'] = cf.is_best.groupby('wid').mean()
plt.scatter(pdf.cost, pdf.best_first)

cf = load_cf('OptimalPlus')
pdf['opt_best_first'] = cf.is_best.groupby('wid').mean()
plt.scatter(pdf.cost, pdf.opt_best_first)
plt.ylim(0, 1.)
show()


# %% ==================== stats ====================


# %% --------
"""
- perecent best first (human/optimal)
- errors broken down by termination
    - terminate early vs late?
- action error rate
- interaction for adaptive satisficing
"""

# %% --------
rdf = pdf[['cost', 'best_first']]
%load_ext rpy2.ipython

# %% --------
%%R -i rdf
summary(lm(best_first ~ cost, data=rdf))


pdf.best_first
# %% --------