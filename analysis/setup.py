from utils import *
import sys
from PIL import Image
EXPERIMENT = int(sys.argv[1])

# %% ==================== GLOBALS AND FLAGS ====================

print('Setting up experiment', EXPERIMENT)
VERSION = f'exp{EXPERIMENT}'

VARIANCES = ['decreasing', 'constant', 'increasing'] if EXPERIMENT in (2,3) else ['constant']

# MODELS = ['Random', 'MetaGreedy', 'OptimalPlus']
if EXPERIMENT == 1:
    MODELS = ("""
        Random MetaGreedy OptimalPlus
        Breadth_Full_NoDepthLimit Depth_Full_NoDepthLimit Best_Satisfice_BestNext
        Best_Full Best_BestNext Best_Satisfice Best_DepthLimit Best_Prune Best
    """.split())

if EXPERIMENT == 2:
    MODELS = ("""
        Random MetaGreedy OptimalPlus
        Breadth_Full Depth_Full Best_Full
    """.split())
        # Breadth_Full_NoDepthLimit Depth_Full_NoDepthLimit Best_Satisfice_BestNext 

if EXPERIMENT >= 3:
    MODELS = ("""
        Random MetaGreedy OptimalPlus
        Best_Satisfice_BestNext Breadth_Satisfice_BestNext Depth_Satisfice_BestNext 
        Expand MetaGreedyExpand OptimalPlusExpand
        Best_Satisfice_BestNext_Expand Breadth_Satisfice_BestNext_Expand Depth_Satisfice_BestNext_Expand 
    """.split())



# else:
#     MODELS.extend(['BreadthFirst', 'DepthFirst'])
#     # MODELS.extend(['BestFirstNoPrune', 'DepthFirstNoPrune', 'BreadthFirstNoPrune'])
# if EXPERIMENT >= 3:
#     MODELS.extend([m + 'Expand' for m in MODELS])

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
    cf['agent'] = k
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
tf = pd.DataFrame(get_result(VERSION, 'trial_features.json'))
n_click = tdf.pop('n_click')  # this is already in tf, we check that it's the same below
assert set(pdf.index).issubset(set(tf.wid))
tdf = tdf.join(tf.set_index(['wid', 'i']), on=['wid', 'i'])
if hasattr(tdf, 'n_click'):
    assert all(tdf.n_click == n_click)

pdf['total_time'] = (pdf.time_end - pdf.time_start) / 1000 / 60
pdf['instruct_time'] = (pdf.time_instruct - pdf.time_start) / 60000
pdf['test_time'] = (pdf.time_end - pdf.time_instruct) / 60000
# pdf['backward'] = tdf.groupby('wid').backward.mean()

# %% ==================== PLOTTING ====================

figs = Figures(f'figs/{EXPERIMENT}')
figs.add_names({
    'backward': 'Proportion Planning Backward',
    'best_next': 'Best - Next Best Path Value',
    'term_reward': 'Best Path Value',
    
    'OptimalPlus': 'Optimal',
    'OptimalPlusPure': 'Optimal',
    'MetaGreedy': 'MetaGreedy',
    # 'Best': 'BestFirst',
    # 'Breadth': 'BreadthFirst',
    # 'Depth': 'DepthFirst',
    'Best_Satisfice' : 'Best +Satisfice',
    'Best_BestNext' : 'Best +BestNext',
    'Best_DepthLimit' : 'Best +DepthLimit',
    'Best_Prune' : 'Best +Prune',
    'Best_Full': 'Best +All',
    'Best_Full_NoPrune': 'Best +All -Prune',
    'Breadth_Full': 'Breadth +All',
    'Breadth_Full_NoPrune': 'Breadth +All -Prune',
    'Depth_Full': 'Depth +All',
    'Depth_Full_NoPrune': 'Depth +All -Prune',
})

if EXPERIMENT > 1:
    figs.add_names({
        'Breadth_Full_NoDepthLimit': 'Breadth',
        'Depth_Full_NoDepthLimit': 'Depth',
        'Best_Satisfice_BestNext': 'Best',
        'BreadthFirst': 'Breadth',
        'DepthFirst': 'Depth',
        'BestFirst': 'Best',
        'Breadth_Satisfice_BestNext': 'Breadth',
        'Depth_Satisfice_BestNext': 'Depth',

    })

figure = figs.figure; show = figs.show; figs.watch()

lb, db, lg, dg, lr, dr, lo, do, lp, dp, *_ = sns.color_palette("Paired")
# lb, db, lg, dg, lr, dr, lo, do, *_ = 

# %% --------
palette = {
    'BestFirst': dg,
    'Human': (0.1, 0.1, 0.1),
    'Random': (0.5, 0.5, 0.5),
    'MetaGreedy': dr,
    'OptimalPlus': db,
    'OptimalPlusPure': db,
    'Optimal': db,
    'Best': lg,
    'Best_Satisfice': lg,
    'Best_BestNext': lg,
    'Best_DepthLimit': lg,
    'Best_Prune': lg,
    'Best_Full': lg,
    'Best_Full_NoPrune': dg,
    'Breadth_Full': do,
    'Breadth_Full_NoPrune': lo,
    'Depth_Full': dp,

    'Depth_Full_NoPrune': lp,
    'Breadth_Full_NoDepthLimit': do,
    'Depth_Full_NoDepthLimit': dp,
    
    'Best_Satisfice_BestNext': dg,
    'Breadth_Satisfice_BestNext': do,
    'Depth_Satisfice_BestNext': dp,

    'OptimalPlusExpand': lb,
    'MetaGreedyExpand': lr,
    'Expand': (0.7, 0.7, 0.7),
    'Best_Satisfice_BestNext_Expand': lg,
    'Breadth_Satisfice_BestNext_Expand': lo,
    'Depth_Satisfice_BestNext_Expand': lp,

}
for m in MODELS:
    assert m in palette, m

# %% --------
plt.rc('legend', fontsize=10, handlelength=2)
sns.set_style('ticks')

write_tex = TeX(path=f'stats/{EXPERIMENT}').write

def setup_variance_plot(nrow=1, title=True, label=True, label_offset=-0.3, height=4, width=4, **kws):
    titlesize = kws.pop('titlesize', 20)
    ncol = len(VARIANCES)
    fig, axes = plt.subplots(nrow, ncol, figsize=(width*ncol,height*nrow), squeeze=False, **kws)
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
    img = img.crop((700, 730, w-700, h-850))
    img.save(f'imgs/{variance}-cropped.png')
    return img

