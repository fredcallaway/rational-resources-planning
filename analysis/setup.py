# %% ==================== LOAD DATA AND SET UP ====================
from utils import *

EXPERIMENT = 1
VERSION = 'exp1'
MODELS = ['BreadthFirst',  'DepthFirst', 'BestFirst', 'Optimal']
VARIANCES = ['decreasing', 'constant', 'increasing']

pdf, tdf = load_data(VERSION)
pdf.variance = pd.Categorical(pdf.variance, categories=VARIANCES)

# Drop incomplete
pdf = pdf.query('complete').copy()
tdf = tdf.loc[list(pdf.index)]
# assert all(tdf.reset_index().wid.value_counts() == 25)

mdp2var = {}
for x in tdf[['variance', 'mdp']].itertuples():
    mdp2var[x.mdp] = x.variance

figs = Figures(f'figs/{VERSION}')
figs.add_names({
    'backward': 'Proportion Planning Backward',
    'BestFirstNoBestNext': 'Satisficing\nBestFirst',
    'BestFirst': 'Adaptive\nBestFirst'
})
figure = figs.figure; show = figs.show; figs.watch()
write_tex = TeX(path=f'stats/{EXPERIMENT}').write

tdf['i'] = list(tdf.trial_index - tdf.trial_index.groupby('wid').min() + 1)
assert all(tdf.groupby(['wid', 'i']).apply(len) == 1)
trial_features = pd.DataFrame(get_result(VERSION, 'trial_features.json')).set_index(['wid', 'i'])
tdf = tdf.join(trial_features, on=['wid', 'i'])

if EXPERIMENT == 1:
    variances = ['constant']
    MODELS = ['RandomSelection', 'Optimal', 'OptimalPlus', 'BestFirst', 'BestFirstNoBestNext']
else:
    MODELS = 'Optimal BestFirst BreadthFirst DepthFirst '.split()
    variances = ['decreasing', 'constant', 'increasing']

keep = tdf.groupby('wid').n_click.mean() >= 1
tdf = tdf.loc[keep]
pdf = pdf.loc[keep]

lb, db, lg, dg, lr, dr, lo, do, *_ = sns.color_palette("Paired")
gray = (0.5, 0.5, 0.5)
pal = [gray, db, lb, dg, lg]
palette = dict(zip(MODELS, pal))
palette['Human'] = '#333333'
