

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
