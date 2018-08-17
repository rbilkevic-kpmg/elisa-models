import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import re
from scipy.stats import pearsonr


def draw_boxplot(col, categories, dpi=200):
    data.boxplot(column=col, by=categories)
    pattern = re.compile('[\W_]+')
    col = pattern.sub('', col)
    categories = pattern.sub('', categories)

    plt.grid(False)
    plt.savefig('graphs\\boxplot_{}_{}'.format(col.lower(), categories.lower()), dpi=dpi)


def draw_scatter(x_n, y_n, dpi=200):
    data.plot(x=x_n, y=y_n, kind='scatter')
    corr = pearsonr(data[x_n], data[y_n])
    x_pos = min(data[x_n])  # + (max(data[x_n]) - min(data[x_n])) * 0.05
    y_pos = max(data[y_n]) - (max(data[y_n]) - min(data[y_n])) * 0.05
    plt.text(x_pos, y_pos, r"$\rho_{xy}=%0.2f$" % corr[0] + "\n" + r"${p-value}=%0.3f$" % corr[1])

    pattern = re.compile('[\W_]+')
    x_n = pattern.sub('', x_n)
    y_n = pattern.sub('', y_n)
    plt.savefig('graphs\\scatter_{}_{}'.format(x_n.lower(), y_n.lower()), dpi=dpi)


data = pd.read_excel('BMA analysis.xlsx', index_col=0)

data['Is_Healthy'] = [1 if row == 'Healthy' else 0 for row in data['Category']]
data['Is_Obese'] = [1 if row == 'Obese' else 0 for row in data['Category']]
data['Is_Severe'] = [1 if row == 'Severe Obesity' else 0 for row in data['Category']]
data['Is_Morbid'] = [1 if row == 'Morbidly Obese' else 0 for row in data['Category']]
data['Female'] = [1 if row == 'F' else 0 for row in data['Gender']]
data['CRP2'] = [row ** 2 for row in data['CRP']]

data = data.dropna(axis=0)

# Primary analysis
draw_boxplot(col='Age', categories='Gender')
draw_boxplot(col='Body Fat, %', categories='Gender')
draw_boxplot(col='Body Fat, %', categories='Is_Healthy')
draw_boxplot(col='BMI', categories='Is_Healthy')
draw_boxplot(col='WHR', categories='Is_Healthy')
draw_boxplot(col='WHR', categories='Is_Healthy')
draw_boxplot(col='8OHDG', categories='Is_Healthy')
draw_boxplot(col='8OHDG', categories='Gender')
draw_boxplot(col='CRP', categories='Is_Healthy')
draw_boxplot(col='CRP', categories='Gender')
draw_boxplot(col='BMA, %', categories='Is_Healthy')
draw_boxplot(col='BMA, %', categories='Gender')

draw_scatter(x_n='BMA, %', y_n='8OHDG')
draw_scatter(x_n='CRP', y_n='8OHDG')
draw_scatter(x_n='BMI', y_n='8OHDG')
draw_scatter(x_n='Body Fat, %', y_n='8OHDG')
draw_scatter(x_n='WHR', y_n='8OHDG')
draw_scatter(x_n='Age', y_n='8OHDG')
draw_scatter(x_n='BMA, %', y_n='CRP')
draw_scatter(x_n='BMI', y_n='CRP2')
draw_scatter(x_n='Body Fat, %', y_n='CRP')
draw_scatter(x_n='WHR', y_n='CRP')
draw_scatter(x_n='Age', y_n='CRP')
draw_scatter(x_n='BMI', y_n='BMA, %')
draw_scatter(x_n='Body Fat, %', y_n='BMA, %')
draw_scatter(x_n='WHR', y_n='BMA, %')
draw_scatter(x_n='Age', y_n='BMA, %')

##
#
# X = data[["Is_Severe", "CRP", "Age"]]
# y = data["8OHDG"]

# X = data[["8OHDG", "Age"]]
# y = data["CRP"]

X = data[["Is_Morbid", "Body Fat, %"]]
y = data["BMA, %"]

X = sm.add_constant(X)
# Note the difference in argument order
model = sm.OLS(y, X).fit()
# predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())
