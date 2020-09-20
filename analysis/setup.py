from utils import *
import sys
from PIL import Image
EXPERIMENT = int(sys.argv[1])

# %% ==================== GLOBALS AND FLAGS ====================

print('Setting up experiment', EXPERIMENT)
VERSION = f'exp{EXPERIMENT}'

VARIANCES = ['decreasing', 'constant', 'increasing'] if EXPERIMENT in (2,3) else ['constant']
MODELS = ['RandomSelection', 'MetaGreedy', 'OptimalPlus', 'BestFirst']
if EXPERIMENT == 1:
    MODELS.append('BestFirstNoBestNext')
else:
    MODELS.extend(['BreadthFirst', 'DepthFirst'])
if EXPERIMENT >= 3:
    MODELS.extend([m + 'Expand' for m in MODELS if not m.startswith('Random')])

MODELS

# %% ==================== LOAD DATA ====================
pdf, tdf = load_data(VERSION)
pdf = pdf.rename(columns={'final_bonus': 'bonus'})

full_pdf = pdf.copy()
pdf.variance = pd.Categorical(pdf.variance, categories=VARIANCES)

# %% ==================== EXCLUSION ====================
pdf = pdf.query('complete').copy()
tdf = tdf.loc[list(pdf.index)]
pdf = pdf.query('n_click >= 1')
if EXPERIMENT == 3:
    pdf = pdf.query('click_delay == "3.0s"')
    print('DROPPING PARTICIPANTS: click_delay == "3.0s"')

keep = list(pdf.index)
tdf = tdf.loc[keep]
# %% ==================== LOAD MODEL RESULTS ====================

fits = load_fits(VERSION, MODELS)
fits = fits.join(pdf[['variance', 'click_delay']], on='wid')
pdf['cost'] = fits.query('model == "OptimalPlus"').set_index('wid').cost.clip(upper=5)


def model_trial_features(model):
    tf = pd.DataFrame(get_result(VERSION, f'{model}-trial_features.json'))
    tf.wid = tf.wid.apply(lambda x: x.split('-')[1])
    tf.set_index('wid', inplace=True)
    tf['variance'] = pdf.variance
    tf['agent'] = model
    return tf.loc[keep].reset_index().set_index('variance')

def load_cf(k, group=False):
    # VERSION = 'exp1' if k in bfs_cols or k in normal_cols else 'exp1-bfs'
    mod = '' if k == 'Human' else k + '-'
    if group and mod:
        mod = 'group-' + mod
    cf = pd.DataFrame(get_result(VERSION, f'{mod}click_features.json'))
    cf['potential_gain'] = (cf.max_competing - cf.term_reward).clip(0)
    cf['competing'] = cf.term_reward - cf.best_next

    if k != 'Human':
        cf.wid = cf.wid.apply(lambda x: x.split('-')[1])

    cf = cf.set_index('wid').loc[keep]
    cf['variance'] = pdf.variance
    # for k, v in cf.items():
    #     if v.dtype == float:
    #         cf[k] = v.astype(int)
    return cf.rename(columns={'expanding': 'expand'})

def load_depth_curve(k):
    mod = '' if k == 'Human' else k + '-'
    d = pd.DataFrame(get_result(VERSION, f'{mod}depth_curve.json'))
    if k != 'Human':
        d.wid = d.wid.apply(lambda x: x.split('-')[1])
    d.set_index('wid', inplace=True)
    d['variance'] = pdf.variance
    d['agent'] = k
    return d.loc[keep]

# %% ==================== ADD COLUMNS ====================

tdf['i'] = list(tdf.trial_index - tdf.trial_index.groupby('wid').min() + 1)
assert all(tdf.groupby(['wid', 'i']).apply(len) == 1)
trial_features = pd.DataFrame(get_result(VERSION, 'trial_features.json'))
assert set(pdf.index).issubset(set(trial_features.wid))
tdf = tdf.join(trial_features.set_index(['wid', 'i']), on=['wid', 'i'])

# print("FIX ME setup.py line 55")
# pdf['total_time'] = (pdf.time_end - pdf.time_start) / 1000 / 60
# pdf['instruct_time'] = (pdf.time_instruct - pdf.time_start) / 60000
# pdf['test_time'] = (pdf.time_end - pdf.time_instruct) / 60000

# pdf['backward'] = tdf.groupby('wid').backward.mean()

# %% ==================== PLOTTING ====================

figs = Figures(f'figs/{EXPERIMENT}')
figs.add_names({
    'backward': 'Proportion Planning Backward',
    'best_next': 'Best - Next Best Path Value',
    'term_reward': 'Best Path Value',
    
    'RandomSelection': 'Random',
    'MetaGreedy': 'Meta-Greedy',
    'BestFirst': 'Best-First',
    'OptimalPlus': 'Optimal',
    'BreadthFirst': 'Breadth-First',
    'DepthFirst': 'Backward' if EXPERIMENT >= 3 else 'Depth-First',

    'BestFirstNoBestNext': 'Simple\nBest-First',
    'MetaGreedyExpand': 'Meta-Greedy + Expansion',
    'BestFirstExpand': 'Best-First + Expansion',
    'OptimalPlusExpand': 'Optimal + Expansion',
    'BreadthFirstExpand': 'Breadth-First + Expansion',
    'DepthFirstExpand': 'Depth-First + Expansion',
})

figure = figs.figure; show = figs.show; figs.watch()

lb, db, lg, dg, lr, dr, lo, do, lp, dp, *_ = sns.color_palette("Paired")
# lb, db, lg, dg, lr, dr, lo, do, *_ = 
palette = {
    'Human': (0.1, 0.1, 0.1),
    'RandomSelection': (0.5, 0.5, 0.5),
    'MetaGreedy': dr,
    'OptimalPlus': db,
    'Optimal': db,
    'BestFirst': dg,
    'BreadthFirst': do,
    'DepthFirst': dp,
    
    'BestFirstNoBestNext': lg,
    'MetaGreedyExpand': lr,
    'OptimalPlusExpand': lb,
    'BestFirstExpand': lg,
    'BreadthFirstExpand': lo,
    'DepthFirstExpand': lp,
}
plt.rc('legend', fontsize=10, handlelength=2)

write_tex = TeX(path=f'stats/{EXPERIMENT}').write

def setup_variance_plot(nrow=1, title=True, label=True, label_offset=-0.3, **kws):
    titlesize = kws.pop('titlesize', 20)
    ncol = len(VARIANCES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4*ncol,4*nrow), squeeze=False, **kws)
    if len(VARIANCES) > 1 and title:
        for v, ax in zip(VARIANCES, axes[0, :]):
            ax.set_title(f'{v.title()} Variance', fontdict=dict(fontsize=titlesize))
    if nrow > 1 and label:
        for char, ax in zip('ABCDEFG', axes[:, 0]):
            ax.annotate(char, (label_offset, 1), xycoords='axes fraction', size=32, va='bottom')
    return fig, axes

def task_image(variance):
    img = Image.open(f'imgs/{variance}.png')
    w, h = img.size
    return img.crop((700, 730, w-700, h-850))
