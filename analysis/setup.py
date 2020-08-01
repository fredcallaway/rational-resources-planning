from utils import *

# %% ==================== GLOBALS AND FLAGS ====================

EXPERIMENT = 1
VERSION = 'exp1'

if EXPERIMENT == 1:
    VARIANCES = ['constant']
    MODELS = ['RandomSelection', 'OptimalPlus', 'BestFirst', 'BestFirstNoBestNext']
else:
    VARIANCES = ['decreasing', 'constant', 'increasing']
    MODELS = 'Optimal BestFirst BreadthFirst DepthFirst '.split()

# %% ==================== LOAD DATA ====================

pdf, tdf = load_data(VERSION)
full_pdf = pdf.copy()
pdf.variance = pd.Categorical(pdf.variance, categories=VARIANCES)

# %% ==================== EXCLUSION ====================

pdf = pdf.query('complete').copy()
tdf = tdf.loc[list(pdf.index)]

keep = tdf.groupby('wid').n_click.mean() >= 1
tdf = tdf.loc[keep]
pdf = pdf.loc[keep]

# %% ==================== LOAD MODEL FITS ====================

fits = load_fits(VERSION, MODELS)
fits = fits.join(pdf[['variance', 'click_delay']], on='wid')
pdf['cost'] = fits.query('model == "OptimalPlus"').set_index('wid').cost.clip(upper=5)

# %% ==================== ADD COLUMNS ====================

tdf['i'] = list(tdf.trial_index - tdf.trial_index.groupby('wid').min() + 1)
assert all(tdf.groupby(['wid', 'i']).apply(len) == 1)
trial_features = pd.DataFrame(get_result(VERSION, 'trial_features.json')).set_index(['wid', 'i'])
tdf = tdf.join(trial_features, on=['wid', 'i'])

# %% ==================== PLOTTING ====================

figs = Figures(f'figs/{VERSION}')
figs.add_names({
    'backward': 'Proportion Planning Backward',
    'BestFirstNoBestNext': 'Satisficing\nBestFirst',
    'BestFirst': 'Adaptive\nBestFirst',
    'best_next': 'Best - Second Best Path Value',
    'term_reward': 'Best Path Value',
    'OptimalPlus': 'Optimal'
})
figure = figs.figure; show = figs.show; figs.watch()

lb, db, lg, dg, lr, dr, lo, do, *_ = sns.color_palette("Paired")
palette = {
    'Human': (0.1, 0.1, 0.1),
    'RandomSelection': (0.5, 0.5, 0.5),
    'OptimalPlus': db,
    'Optimal': db,
    'BestFirst': dg,
    'BestFirstNoBestNext': lg,
}

write_tex = TeX(path=f'stats/{EXPERIMENT}').write

def setup_variance_plot(nrow=1):
    ncol = len(VARIANCES)
    return plt.subplots(nrow, ncol, figsize=(4*ncol,4), squeeze=False)
