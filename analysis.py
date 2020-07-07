import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def load_data(exp):
    tdf = pd.read_pickle(f'data/{exp}/trials.pkl').query('block == "test"')
    pdf = pd.read_pickle(f'data/{exp}/participants.pkl')
    tdf['click_delay'] = pdf['click_delay'] = pdf.pop('clickDelay').apply(lambda x: str(x/1000)+'s')
    if 'variance' not in pdf:
        pdf['variance'] = 'constant'
    tdf['variance'] = pdf.variance
    tdf['n_click'] = tdf.clicks.apply(len)

    for k in ['n_click', 'score']:
        pdf[k] = tdf[k].groupby('wid').mean()
    
    return pdf, tdf

def load_fits(exp, models, path='mle'):
    cv_fits = pd.concat([pd.read_csv(f'model/results/{exp}/{path}/{model}-cv.csv') 
                         for model in models], sort=False)
    fits = pd.concat([pd.read_csv(f'model/results/{exp}/{path}/{model}.csv')
                      for model in models], sort=False)
    fits.set_index(['model', 'wid'], inplace=True)
    fits['cv_nll'] = cv_fits.groupby(['model', 'wid']).test_nll.sum()
    fits['overfitting'] = fits.cv_nll - fits.nll
    return fits.reset_index()

def load_pareto(exp, models, path='pareto'):
    mdps = pd.read_csv(f"model/results/{exp}/mdps.csv").set_index('mdp')
    
    def load(model):
        d = pd.read_csv(f'model/results/{exp}/pareto/{model}.csv')
        d['model'] = model
        return d

    d = pd.concat(map(load, models), sort=True)
    return d.join(mdps, on='mdp').set_index(['variance', 'model'])


class Figures(object):
    """Plots and saves figures."""
    def __init__(self, path='figs/'):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def savefig(self, name, tight=True):
        if tight:
            plt.tight_layout()
        name = name.lower()
        path = f'{self.path}/{name}.png'
        print(path)
        plt.savefig(path, dpi=400, bbox_inches='tight')

    # def plot(self, **kwargs1):
    #     """Decorator that calls a plotting function and saves the result."""
    #     def decorator(func):
    #         def wrapped(*args, **kwargs):
    #             kwargs.update(kwargs1)
    #             params = [v for v in kwargs1.values() if v is not None]
    #             param_str = '_' + str_join(params).rstrip('_') if params else ''
    #             name = func.__name__ + param_str
    #             if name.startswith('plot_'):
    #                 name = name[len('plot_'):]
    #             func(*args, **kwargs)
    #             self.savefig(name)
    #         wrapped()
    #         return wrapped

    #     return decorator

def plot_params(fits):
    d = fits.query('model == "BestFirst"')
    betas = ['β_depth', 'β_sat', 'β_lead', 'β_click']
    thetas = ['θ_depth', 'θ_sat', 'θ_lead']
    params = [*betas, *thetas, 'ε']
    upper = [3, 3, 3, 3, 4, 30, 30, 1]
    X = d[params] / upper
    X = X[betas]
    X.T.plot(legend=False, color='k', alpha=0.4)




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