import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from glob import glob
import json

sns.set_context('notebook', font_scale=1.3)
sns.set_style('white')

def str_join(args, sep=' '):
    return sep.join(map(str, args))

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_result(version, name):
    return load_json(f'../model/results/{version}/{name}')

def load_data(exp):
    tdf = pd.read_pickle(f'../data/{exp}/trials.pkl').query('block == "test"')
    pdf = pd.read_pickle(f'../data/{exp}/participants.pkl')
    pdf.rename(columns={'clickDelay': 'click_delay'}, inplace=True)
    tdf['click_delay'] = pdf['click_delay'] = pdf.click_delay.apply(lambda x: str(x/1000)+'s')
    if 'variance' not in pdf:
        pdf['variance'] = 'constant'
    tdf['mdp'], tdf['trial_id'] = tdf.trial_id.str.split('-', expand=True).values.T
    tdf['variance'] = pdf.variance
    tdf['n_click'] = tdf.clicks.apply(len)

    for k in ['n_click', 'score']:
        pdf[k] = tdf[k].groupby('wid').mean()
    
    return pdf, tdf

def load_fits(exp, models, path='mle'):
    cv_fits = pd.concat([pd.read_csv(f'../model/results/{exp}/{path}/{model}-cv.csv') 
                         for model in models], sort=False)
    fits = pd.concat([pd.read_csv(f'../model/results/{exp}/{path}/{model}.csv')
                      for model in models], sort=False)
    fits.set_index(['model', 'wid'], inplace=True)
    fits['cv_nll'] = cv_fits.groupby(['model', 'wid']).test_nll.sum()
    fits['overfitting'] = fits.cv_nll - fits.nll
    return fits.reset_index()


from datetime import datetime
class Figures(object):
    """Plots and saves figures."""
    def __init__(self, path='figs', hist_path='fighist', dpi=300):
        self.path = path
        self.hist_path = hist_path
        self.dpi = dpi
        self.names = {}
        self._last = None

        os.makedirs(path, exist_ok=True)
        os.makedirs(hist_path, exist_ok=True)

    def add_names(self, names):
        self.names.update(names)

    def nice_name(self, name):
        return self.names.get(name, name.title())

    def open(self):
        latest = max(glob(f'fighist/*'), key=os.path.getctime)
        os.system(f'open {latest}')

    def reformat_labels(self, ax=None):
        ax = ax or plt.gca()
        labels = [t.get_text() for t in ax.get_xticklabels()]
        new_labels = [self.nice_name(lab) for lab in labels]
        ax.set_xticklabels(new_labels)

        ax.set_ylabel(self.nice_name(ax.get_ylabel()))
        ax.set_xlabel(self.nice_name(ax.get_xlabel()))
        
    def reformat_legend(self, ax=None):
        ax = ax or plt.gca()
        if ax.legend_:
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [self.names.get(l, l.title()).replace('\n', ' ') for l in labels]
            ax.legend(handles=handles, labels=new_labels)

    def watch(self):
        from watcher import Watcher
        if hasattr(self, 'watcher'):
            self.watcher.start()
        self.watcher = Watcher(self.hist_path)

    def show(self, name='tmp', tight=True, reformat_labels=False, reformat_legend=False):
        try:
            if tight:
                plt.tight_layout()
            if reformat_labels:
                self.reformat_labels()
            if reformat_legend:
                self.reformat_legend()

            dt = datetime.now().strftime('%m-%d-%H-%M-%S')
            p = f'{dt}-{name}.png'
            tmp = f'{self.hist_path}/{p}'
            plt.savefig(tmp, dpi=self.dpi, bbox_inches='tight')

            if name != 'tmp':
                name = name.lower()
                path = f'{self.path}/{name}.png'
                os.system(f'cp {tmp} {path}')
                print(f"Wrote {path}")
        finally:
            plt.close('all')

    def figure(self, save=True, reformat_labels=False, reformat_legend=False, **kwargs):
        """Decorator that calls a plotting function and saves the result."""
        def decorator(func):
            params = [v for v in kwargs.values() if v is not None]
            param_str = '_' + str_join(params).rstrip('_') if params else ''
            name = func.__name__ + param_str
            if name.startswith('plot_'):
                name = name[len('plot_'):].lower()
            try:
                plt.figure()
                func(**kwargs)
                self.show(name, reformat_labels=reformat_labels, reformat_legend=reformat_legend)
            finally:
                plt.close('all')
            return func
        return decorator

def plot_params(fits):
    d = fits.query('model == "BestFirst"')
    betas = ['β_depth', 'β_sat', 'β_lead', 'β_click']
    thetas = ['θ_depth', 'θ_sat', 'θ_lead']
    params = [*betas, *thetas, 'ε']
    upper = [3, 3, 3, 3, 4, 30, 30, 1]
    X = d[params] / upper
    X = X[betas]
    X.T.plot(legend=False, color='k', alpha=0.4)


class TeX(object):
    """Saves tex files."""
    def __init__(self, path='stats', clear=False):
        self.path = path
        if clear:
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


    def write(self, name, tex):
        file = f"{self.path}/{name}.tex"
        with open(file, "w+") as f:
            f.write(str(tex) + r"\unskip")
        print(f'wrote "{tex}" to "{file}"')


# def load_data_multi(codes):
#     pdfs, tdfs = zip(*map(load_data, codes))
#     return pd.concat(pdfs, sort=False), pd.concat(tdfs, sort=False)

# mx = 'DepthFirst'; my = 'BestFirst'
# d = fits.query('variance == "increasing"')
# x = d.groupby(['wid', 'model']).nll.mean().reset_index().set_index('model').nll
# plt.scatter(x.loc[mx], x.loc[my])
# plt.plot([0, 400], [0, 400])
# plt.xlabel(f'{mx} NLL')
# plt.ylabel(f'{my} NLL')