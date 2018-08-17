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

data['Is_Healthy'] = [1 if row['Category'] == 'Healthy' else 0 for ix, row in data.iterrows()]
data['Is_Obese'] = [1 if row['Category'] == 'Obese' else 0 for ix, row in data.iterrows()]
data['Is_Severe'] = [1 if row['Category'] == 'Severe Obesity' else 0 for ix, row in data.iterrows()]
data['Is_Morbid'] = [1 if row['Category'] == 'Morbidly Obese' else 0 for ix, row in data.iterrows()]
data['Female'] = [1 if row['Gender'] == 'F' else 0 for ix, row in data.iterrows()]

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
draw_boxplot(col='DNA DAMAGE, %', categories='Is_Healthy')
draw_boxplot(col='DNA DAMAGE, %', categories='Gender')

draw_scatter(x_n='DNA DAMAGE, %', y_n='8OHDG')
draw_scatter(x_n='CRP', y_n='8OHDG')
draw_scatter(x_n='BMI', y_n='8OHDG')
draw_scatter(x_n='Body Fat, %', y_n='8OHDG')
draw_scatter(x_n='WHR', y_n='8OHDG')
draw_scatter(x_n='DNA DAMAGE, %', y_n='CRP')
draw_scatter(x_n='BMI', y_n='CRP')
draw_scatter(x_n='Body Fat, %', y_n='CRP')
draw_scatter(x_n='WHR', y_n='CRP')
draw_scatter(x_n='BMI', y_n='DNA DAMAGE, %')
draw_scatter(x_n='Body Fat, %', y_n='DNA DAMAGE, %')
draw_scatter(x_n='WHR', y_n='DNA DAMAGE, %')

##
# X = data[["Is_Severe", "CRP", "Age"]]
# y = data["8OHDG"]

# X = data[["8OHDG", "Age"]]
# y = data["CRP"]

X = data[["Is_Morbid", "Body Fat, %"]]
y = data["DNA DAMAGE, %"]

X = sm.add_constant(X)
# Note the difference in argument order
model = sm.OLS(y, X).fit()
# predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())
