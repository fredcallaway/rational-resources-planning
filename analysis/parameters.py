cv_fits = pd.concat([pd.read_csv(f'../model/results/{VERSION}/mle/{model}-cv.csv') 
                     for model in MODELS], sort=False).set_index('wid')

X = load_fits(VERSION, ['Breadth_BestNext'])
sns.regplot(X.β_best_next, X.β_depth)
show()
# %% --------
full_fits.query('β_depthlim != 1e5').model.value_counts()

x = full_fits.query('model == "Best_DepthLimit"')
x0 = full_fits.query('model == "Best"')
x.β_depthlim
x.θ_depthlim

sns.distplot(x0.nll - x.nll); show()

# %% --------
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
