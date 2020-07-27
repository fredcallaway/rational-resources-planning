# %% ====================  ====================


MODELS = "Optimal BestFirst MetaGreedy BiasedOptimal BiasedBestFirst BiasedMetaGreedy".split()
MODELS = "BestFirst Optimal MetaGreedy BiasedBestFirst BiasedOptimal BiasedMetaGreedy".split()

paired = sns.palettes.color_palette('Paired')
pal = dict(zip(MODELS, [*paired[1:7:2], *paired[0:6:2]]))


# In[5]:


df = pd.read_csv('julia/real_results/likelihoods.csv')
df['full_model'] = df.biased.apply(lambda x: 'Biased' if x else '') + df.model

no_click = df.query('full_model == "Optimal"').groupby(['map', 'wid']).apply(len) < 2
usually_click = no_click.groupby('wid').sum() <= 2
print(usually_click.mean())
df = df.set_index('wid').loc[usually_click]
len(df.reset_index().wid.unique())


# In[9]:


d = df.set_index('full_model').logp
diff = d['BiasedOptimal'] - d['BiasedBestFirst']


# In[10]:


stim = df.groupby(['full_model', 'map']).logp.sum()
diff = stim['BiasedOptimal'] - stim['BiasedBestFirst']
diff


# In[337]:


d = ind.set_index(['full_model', 'wid']).logp
diff = d['BiasedOptimal'] - d['BiasedBestFirst']
data = pdf.set_index('wid')[['logp', 'n_reveal', 'cost']]
data['diff'] = diff
sns.lmplot('n_reveal', 'diff', data=data)


# In[12]:


ind = df.groupby(['full_model', 'wid']).logp.sum().reset_index()
n_obs = df.groupby('wid').apply(len) / len(MODELS)
ind['avg_logp'] = list(ind.set_index('wid').logp / n_obs)


# In[16]:


# sns.pointplot('full_model', 'logp', data=ind, )


# In[20]:


val = 'logp'
avg = ind.set_index('full_model').loc[MODELS].groupby('full_model')[val].mean()
plt.plot(avg.values, avg.index)


# In[49]:


# horizontal orientation

def plot_line(MODELS, d, val):
    plt.plot(MODELS, d.set_index('full_model').loc[MODELS][val], color='k', lw=1, alpha=0.3)

def plot_participants(val, MODELS=MODELS):
    plt.figure(figsize=(8,4))
    sns.swarmplot(y='full_model', x=val, data=ind, order=MODELS, palette=pal, )
    for w, d in ind.groupby('wid'):
        plt.plot(d.set_index('full_model').loc[MODELS][val], MODELS, color='k', lw=1, alpha=0.3)
#     plt.plot(ind.set_index('full_model').loc[MODELS][val])
    avg = ind.set_index('full_model').loc[MODELS].groupby('full_model')[val].mean()
    plt.scatter(avg.values, avg.index, s=2000, marker='|', c='k')
    plt.ylabel('')
    plt.xlabel('Log Likelihood')

plot_participants('logp', MODELS=MODELS[:3])
plt.tight_layout()
plt.savefig('individual_likelihood_nobias.pdf')


# In[44]:


alt_models = MODELS[::3] + MODELS[1::3] + MODELS[2::3]


# In[48]:


plot_participants('logp', MODELS=MODELS)
plt.tight_layout()
plt.savefig('individual_likelihood.pdf')


# In[53]:


stim = df.groupby(['full_model', 'map']).logp.sum().reset_index()

def plot_participants(val, MODELS=MODELS):
    plt.figure(figsize=(8,4))
    sns.swarmplot(y='full_model', x=val, data=stim, order=MODELS, palette=pal, )
    for w, d in stim.groupby('map'):
        plt.plot(d.set_index('full_model').loc[MODELS][val], MODELS, color='k', lw=1, alpha=0.3)
#     plt.plot(stim.set_index('full_model').loc[MODELS][val])
#     avg = stim.set_index('full_model').loc[MODELS].groupby('full_model')[val].mean()
#     plt.scatter(avg.values, avg.index, s=2000, marker='|', c='k')
    plt.ylabel('')
    plt.xlabel('Log Likelihood')
        
plot_participants('logp', MODELS=MODELS[3:])
plt.tight_layout()
plt.savefig('map_likelihood.pdf')

