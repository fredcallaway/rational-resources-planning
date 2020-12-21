from utils import *
import sys
from PIL import Image
EXPERIMENT = int(sys.argv[1])

# %% ==================== GLOBALS AND FLAGS ====================

print('Setting up experiment', EXPERIMENT)
VERSION = f'exp{EXPERIMENT}'

VARIANCES = ['decreasing', 'constant', 'increasing'] if EXPERIMENT in (2,3) else ['constant']

# if EXPERIMENT == 1:
#     MODELS = ("""
#         Random MetaGreedy OptimalPlus
#         Breadth_Full Depth_Full
#         Best_Full_NoPrune
#         Best_Full Best_BestNext Best_Satisfice Best_DepthLimit Best_Prune Best
#     """.split())

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
if EXPERIMENT == 1:
    pdf = pdf.drop('w29dd261')  # extra participant approved after reaching 100

full_pdf = pdf.copy()
pdf = pdf.query('complete')
tdf = tdf.loc[list(pdf.index)]
pdf.variance = pd.Categorical(pdf.variance, categories=VARIANCES)

assert (tdf.index.value_counts() == 25).all()
assert set(tdf.index) == set(pdf.index)


# # %% ==================== EXCLUSION ====================
# pdf = pdf.query('complete').copy()
# tdf = tdf.loc[list(pdf.index)]
# pdf = pdf.query('n_click >= 1')
# # if EXPERIMENT == 3:
# #     pdf = pdf.query('click_delay == "3.0s"')
# #     print('DROPPING PARTICIPANTS: click_delay == "3.0s"')

# list(pdf.index) = list(pdf.index)
# tdf = tdf.loc[list(pdf.index)]
# %% ==================== LOAD MODEL RESULTS ====================

# fits = load_fits(VERSION, MODELS)
# fits = fits.join(pdf[['variance', 'click_delay']], on='wid')
# pdf['cost'] = fits.query('model == "OptimalPlus"').set_index('wid').cost.clip(upper=5)


def model_trial_features(model):
    tf = pd.DataFrame(get_result(VERSION, f'trial_features/{model}.json'))
    tf.wid = tf.wid.apply(lambda x: x.split('-')[1])
    tf.set_index('wid', inplace=True)
    tf['variance'] = pdf.variance
    tf['agent'] = model
    return tf.loc[list(pdf.index)].reset_index().set_index('variance')

def load_cf(k):
    # VERSION = 'exp1' if k in bfs_cols or k in normal_cols else 'exp1-bfs'
    cf = pd.DataFrame(get_result(VERSION, f'click_features/{k}.json'))
    try:
        cf['potential_gain'] = (cf.max_competing - cf.term_reward).clip(0)
    except:
        print('Error computing potential_gain')
    cf['competing'] = cf.term_reward - cf.best_next

    if k != 'Human':
        cf.wid = cf.wid.apply(lambda x: x.split('-')[1])

    cf = cf.set_index('wid').loc[list(pdf.index)]
    cf['variance'] = pdf.variance
    cf['agent'] = k
    # for k, v in cf.items():
    #     if v.dtype == float:
    #         cf[k] = v.astype(int)
    return cf.rename(columns={'expanding': 'expand'})

def load_depth_curve(k):
    d = pd.DataFrame(get_result(VERSION, f'depth_curve/{k}.json'))
    if k != 'Human':
        d.wid = d.wid.apply(lambda x: x.split('-')[1])
    d.set_index('wid', inplace=True)
    d['variance'] = pdf.variance
    d['agent'] = k
    return d.loc[list(pdf.index)]

# %% ==================== ADD COLUMNS ====================

tdf['i'] = list(tdf.trial_index - tdf.trial_index.groupby('wid').min() + 1)
assert all(tdf.groupby(['wid', 'i']).apply(len) == 1)
tf = pd.DataFrame(get_result(VERSION, 'trial_features/Human.json'))
n_click = tdf.pop('n_click')  # this is already in tf, we check that it's the same below

assert set(pdf.index) <= set(tf.wid)
tdf = tdf.join(tf.set_index(['wid', 'i']), on=['wid', 'i'])
assert all(tdf.n_click == n_click)

pdf['total_time'] = (pdf.time_end - pdf.time_start) / 1000 / 60
pdf['instruct_time'] = (pdf.time_instruct - pdf.time_start) / 60000
pdf['test_time'] = (pdf.time_end - pdf.time_instruct) / 60000
# pdf['backward'] = tdf.groupby('wid').backward.mean()

# %% ==================== FIGURES ====================

figs = Figures(f'figs/{EXPERIMENT}', pdf=True)
figure = figs.figure; show = figs.show; figs.watch()
model_names = get_result(VERSION, 'param_counts.json').keys()

all_extra = ['Satisfice', 'BestNext', 'DepthLimit', 'Prune']

def prettify_name(name):
    spec = base, *extra = name.split('_')
    if EXPERIMENT > 1:
        return base
    if len(extra) == len(all_extra):
        return base + ' +All'
    elif len(extra) == len(all_extra) - 1:
        missing = [e for e in all_extra if e not in extra]
        assert len(missing) <= 1
        return base + ' -' + missing[0]
    else:
        return ' +'.join(spec)

figs.add_names({
    'backward': 'Proportion Planning Backward',
    'best_next': 'Best - Next Best Path Value',
    'term_reward': 'Best Path Value',
    'OptimalPlus': 'Optimal',
    'OptimalPlusPure': 'Optimal',
    'MetaGreedy': 'Myopic',
    'Random': 'Random'
})
figs.add_names({name: prettify_name(name) 
    for name in model_names if name not in figs.names})

# if EXPERIMENT > 1:
#     figs.add_names({
#         'Breadth_Full_NoDepthLimit': 'Breadth',
#         'Depth_Full_NoDepthLimit': 'Depth',
#         'Best_Satisfice_BestNext': 'Best',
#         'BreadthFirst': 'Breadth',
#         'DepthFirst': 'Depth',
#         'BestFirst': 'Best',
#         'Breadth_Satisfice_BestNext': 'Breadth',
#         'Depth_Satisfice_BestNext': 'Depth',
#     })

# %% --------
lb, db, lg, dg, lr, dr, lo, do, lp, dp, *_ = sns.color_palette("Paired")

palette = {
    # 'BestFirst': dg,
    'Human': (0.1, 0.1, 0.1),
    'Random': (0.5, 0.5, 0.5),
    'MetaGreedy': dr,
    'OptimalPlus': db,
    'OptimalPlusPure': db,
    'Optimal': db,

    'OptimalPlusExpand': lb,
    'MetaGreedyExpand': lr,
    'Expand': (0.7, 0.7, 0.7),

    # 'Best_Satisfice_BestNext_Expand': lg,
    # 'Breadth_Satisfice_BestNext_Expand': lo,
    # 'Depth_Satisfice_BestNext_Expand': lp,
}

def pick_color(name):
    if name.startswith('Best'):
        return lg if name.endswith('Expand') else dg
    if name.startswith('Breadth'):
        return lo if name.endswith('Expand') else do
    if name.startswith('Depth'):
        return lp if name.endswith('Expand') else dp

for m in model_names:
    if m not in palette:
        palette[m] = pick_color(m)

# lb, db, lg, dg, lr, dr, lo, do, *_ = 


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

