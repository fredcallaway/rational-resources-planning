best_first = get_result(VERSION, 'bestfirst.json')

bfo = pd.Series(best_first['optimal'])
bfo.index = bfo.index.astype(float)
bfo = bfo.sort_index().iloc[:-1]  # drop 100
pdf['best_first'] = pd.Series(best_first['human'])

def mean_std(x, digits=1, pct=False):
    if pct:
        x = x * 100
        return fr'{x.mean().round(digits)}\% $\pm$ {x.std().round(digits)}\%'
    else:
        return fr'{x.mean().round(digits)} $\pm$ {x.std().round(digits)}'

write_tex("best_first", mean_std(pdf.best_first, pct=True))

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