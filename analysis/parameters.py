

total = logp.sum()


# %% --------

fits = load_fits(VERSION, [*total.filter(regex='^Best').index, *total.filter(regex='^Fancy_Best').index])
fits['fancy'] = fits.model.str.startswith('Fancy')

x = fits.groupby('model').cv_nll.sum().sort_values()
np.exp(-x/len(logp))

load_fits(VERSION, ["OptimalPlus"]).cv_nll.sum()

# %% --------

fits.model = fits.model.str.strip('Fancy_')
X = fits.query('model == "Best_Satisfice_BestNext"').set_index(['wid', 'fancy']).cv_nll.unstack('fancy')
sns.distplot(X[0] - X[1]); show()
X[0].mean() - X[1].mean()

# %% --------
X = fits.set_index(['model', 'wid'])
wid = 'w04ef2b5'
X.loc['Fancy_Best_BestNext_DepthLimit'].loc[wid].nll
X.loc['Best_Satisfice_BestNext_DepthLimit'].loc[wid].nll


# %% --------

def show_param(k):
    x = fits.query('not fancy')[k]
    x = x[abs(x) < 1000]
    plt.plot(x.sort_values().values, np.linspace(0, 1, len(x)), 'o')

show_param('β_depthlim'); show()
show_param('β_satisfice'); show()
show_param('θ_prune'); show()
show_param('β_best_next'); show()
show_param('β_prune'); show()
# show_param('θ_depthlim')
# show_param('β_depth')
# show_param('β_expand')
# show_param('β_best')
# show_param('θ_term')
# show_param('θ_satisfice')
show_param('α_term')
show()
# %% --------

X.loc["Best_Satisfice_BestNext_DepthLimit"].value

sns.catplot('fancy', 'nll', data=fits)
show()
# %% --------

total = fits.groupby('model').nll.sum().reset_index()
total
total['fancy'] = total.model.str.startswith('Fancy')
sns.catplot('fancy', 'nll', data=total)
show()
# %% --------

x = fits.groupby(['fancy', 'model']).cv_nll.sum().reset_index().set_index('model')
fancy = x.query('fancy').cv_nll
base = x.query('not fancy').cv_nll
X = pd.DataFrame({'value': base, 'probability': fancy})

plt.plot(np.exp(-X.values.T/len(logp)), color=palette['Best'])
plt.axhline(np.exp(-load_fits(VERSION, ["OptimalPlus"]).cv_nll.sum() / len(logp)), color=palette['Optimal'])
plt.xticks([0,1], ['Value', 'Probability'])
plt.xlim(-0.5, 1.5)
show()

# %% -------
base = fits.query('model == "Best_Satisfice_BestNext"')
fancy = fits.query('model == "Fancy_Best_Satisfice_BestNext"')

# %% --------
plt.plot(X.query('probability < 23000').values.T, color='k')
show()

X.query('probability < 23000')

# %% --------

fits.query('wid == "w046f5c6" and model == "Fancy_Best_Satisfice_BestNext"').nll


# %% ==================== Old ====================


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
