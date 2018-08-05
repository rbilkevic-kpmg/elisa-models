import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr



data = pd.read_excel('BMA analysis.xlsx', index_col=0, na_values=np.float64('nan'))

data['Is_Healthy'] = [1 if row['Category'] == 'Healthy' else 0 for ix, row in data.iterrows()]
data['Is_Obese'] = [1 if row['Category'] == 'Obese' else 0 for ix, row in data.iterrows()]
data['Is_Severe'] = [1 if row['Category'] == 'Severe Obesity' else 0 for ix, row in data.iterrows()]
data['Is_Morbid'] = [1 if row['Category'] == 'Morbidly Obese' else 0 for ix, row in data.iterrows()]


data['Female'] = [1 if row['Gender'] == 'F' else 0 for ix, row in data.iterrows()]
data = data.dropna(axis=0)

#print(data)

# Primary analysis
dpi = 200

data.boxplot(column='Age', by='Gender')
plt.grid(False)
plt.savefig('graphs\\boxplot_age_gender', bbox_inches='tight', dpi=dpi)

data.boxplot(column='Body Fat, %', by='Gender')
plt.grid(False)
plt.savefig('graphs\\boxplot_fat_gender', bbox_inches='tight', dpi=dpi)

data.boxplot(column='Body Fat, %', by='Is_Healthy')
plt.grid(False)
plt.savefig('graphs\\boxplot_fat_healthy', bbox_inches='tight', dpi=dpi)

data.boxplot(column='BMI', by='Is_Healthy')
plt.grid(False)
plt.savefig('graphs\\boxplot_bmi_healthy', bbox_inches='tight', dpi=dpi)

data.boxplot(column='WHR', by='Is_Healthy')
plt.grid(False)
plt.savefig('graphs\\boxplot_whr_healthy', bbox_inches='tight', dpi=dpi)

data.boxplot(column='8OHDG', by='Is_Healthy')
plt.grid(False)
plt.savefig('graphs\\boxplot_8ohdg_healthy', bbox_inches='tight', dpi=dpi)

data.boxplot(column='8OHDG', by='Gender')
plt.grid(False)
plt.savefig('graphs\\boxplot_8ohdg_gender', bbox_inches='tight', dpi=dpi)

data.boxplot(column='CRP', by='Is_Healthy')
plt.grid(False)
plt.savefig('graphs\\boxplot_crp_healthy', bbox_inches='tight', dpi=dpi)

data.boxplot(column='CRP', by='Gender')
plt.grid(False)
plt.savefig('graphs\\boxplot_crp_gender', bbox_inches='tight', dpi=dpi)

data.boxplot(column='DNA DAMAGE, %', by='Is_Healthy')
plt.grid(False)
plt.savefig('graphs\\boxplot_dnadmg_healthy', bbox_inches='tight', dpi=dpi)

data.boxplot(column='DNA DAMAGE, %', by='Gender')
plt.grid(False)
plt.savefig('graphs\\boxplot_dnadmg_gender', bbox_inches='tight', dpi=dpi)

x_n = 'DNA DAMAGE, %'
y_n = '8OHDG'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_dnadmg_8ohdg', bbox_inches='tight', dpi=dpi)

x_n = 'CRP'
y_n = '8OHDG'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_crp_8ohdg', bbox_inches='tight', dpi=dpi)

x_n = 'BMI'
y_n = '8OHDG'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_bmi_8ohdg', bbox_inches='tight', dpi=dpi)

x_n = 'Body Fat, %'
y_n = '8OHDG'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_fat_8ohdg', bbox_inches='tight', dpi=dpi)

x_n = 'WHR'
y_n = '8OHDG'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_whr_8ohdg', bbox_inches='tight', dpi=dpi)


x_n = 'DNA DAMAGE, %'
y_n = 'CRP'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_dnadmg_crp', bbox_inches='tight', dpi=dpi)

x_n = 'BMI'
y_n = 'CRP'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_bmi_crp', bbox_inches='tight', dpi=dpi)

x_n = 'Body Fat, %'
y_n = 'CRP'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_fat_crp', bbox_inches='tight', dpi=dpi)

x_n = 'WHR'
y_n = 'CRP'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_whr_crp', bbox_inches='tight', dpi=dpi)



x_n = 'BMI'
y_n = 'DNA DAMAGE, %'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_bmi_dnadmg', bbox_inches='tight', dpi=dpi)

x_n = 'Body Fat, %'
y_n = 'DNA DAMAGE, %'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_fat_dnadmg', bbox_inches='tight', dpi=dpi)

x_n = 'WHR'
y_n = 'DNA DAMAGE, %'
data.plot(x=x_n, y=y_n, kind='scatter')
corr = pearsonr(data[x_n], data[y_n])
x_pos = min(data[x_n]) #+ (max(data[x_n]) - min(data[x_n])) * 0.05
y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])
plt.savefig('graphs\\scatter_whr_dnadmg', bbox_inches='tight', dpi=dpi)

#
# X = data[["Is_Healthy", "Female", "DNA DAMAGE, %", "CRP", "WHR", "Age", "BMI"]]
# y = data["8OHDG"]

# X = data[["Is_Healthy", "Female", "DNA DAMAGE, %", "8OHDG", "WHR", "Age", "BMI"]]
# y = data["CRP"]

X = data[["Body Fat, %", "Age"]]
y = data["DNA DAMAGE, %"]

X = sm.add_constant(X)
# Note the difference in argument order
model = sm.OLS(y, X, missing='drop').fit()
# predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())
