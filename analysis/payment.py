# %% ==================== Make bonus ====================
prolific = pd.read_csv('prolific_export_5f0e7b7db485881bcb6b11b5.csv')
pp = prolific.query('status == "APPROVED"')
pp.columns

full_pdf.workerId
full_pdf
total_score = tdf.loc[list(pdf.index)].query('block == "test"').groupby('wid').score.sum()
# bonus = (total_score + 100) * .005
# bonus = bonus.rename('bonus').round(2).to_frame()
# bonus['workerid'] = pdf.workerid

# %% --------
from datetime import datetime
pdf['start'] = pdf.time_start.dropna().apply(lambda x: datetime.fromtimestamp(x/1000))
pdf = pdf.loc[pdf.start.dt.day == 15]
bonus = pdf[['workerid', 'final_bonus']].dropna().query('final_bonus > 0')
bonus.to_csv('bonus.csv', index=False, header=False)
!cat bonus.csv | pbcopy
# less bonus.csv


# %% ==================== PAYMENT ====================



pdf['total_time'] = total_time = (pdf.time_end - pdf.time_start) / 1000 / 60
sns.distplot(pdf.total_time)
m = pdf.total_time.median()
plt.axvline(m)
plt.title(f'Median time: {m:.2f} minutes')
show()

# %% --------
pdf['total_time'] = total_time = (pdf.time_end - pdf.time_start) / 1000 / 60
pdf['instruct_time'] = (pdf.time_instruct - pdf.time_start) / 60000
pdf['test_time'] = (pdf.time_end - pdf.time_instruct) / 60000

pdf.total_time.mean()
pdf.instruct_time.median()
print(pdf.groupby('click_delay').total_time.mean())
print(pdf.groupby('click_delay').final_bonus.mean())
print(pdf.groupby('click_delay').n_click.mean())

# %% --------
pdf.final_bonus.loc[lambda x: x>0].median()
pdf

wage = 60 * (bonus + base_pay) / pdf.total_time
sns.distplot(wage)
m = wage.median()
plt.axvline(m)
plt.title(f'Median wage: ${m:.2f} per hour')
show()

# %% --------

pdf['wage'] = wage
sns.catplot('click_delay', 'wage', data=pdf, kind='swarm',
           order='1.0s 2.0s 3.0s 4.0s'.split())
show()
