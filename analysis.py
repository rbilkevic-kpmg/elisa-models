import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import re
from scipy.stats import pearsonr, ttest_ind
from itertools import combinations


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


def get_significance(x):
    if x < 0.0001:
        return '***'
    elif x < 0.001:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return ''


data = pd.read_excel('BMA analysis.xlsx', index_col=0)

graph_data = data[
    ['Gender', 'BMI', 'Category', 'WHR', 'Body Fat, %', 'CRP', '8OHDG', 'BMA, %']
]
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(graph_data, hue="Category", plot_kws=
    {
    "s":30,
    "alpha":0.8,
    'lw':0.1,
    'edgecolor':'k'
    })
g.fig.set_size_inches(12, 10)
g.fig.subplots_adjust(bottom=0.1, left=0.07, right=0.86)
plt.show()

data['Control'] = [1 if row == 'Control' else 0 for row in data['Category']]
data['Obese'] = [1 if row == 'Obese' else 0 for row in data['Category']]
data['Severely Obese'] = [1 if row == 'Severe Obesity' else 0 for row in data['Category']]
data['Morbidly Obese'] = [1 if row == 'Morbidly Obese' else 0 for row in data['Category']]
data['Female'] = [1 if row == 'F' else 0 for row in data['Gender']]

data = data.dropna(axis=0)

# Primary analysis
draw_boxplot(col='Age', categories='Gender')
draw_boxplot(col='Age', categories='Control')
draw_boxplot(col='Body Fat, %', categories='Gender')
draw_boxplot(col='Body Fat, %', categories='Control')
draw_boxplot(col='BMI', categories='Gender')
draw_boxplot(col='BMI', categories='Control')
draw_boxplot(col='WHR', categories='Gender')
draw_boxplot(col='WHR', categories='Control')
draw_boxplot(col='8OHDG', categories='Gender')
draw_boxplot(col='8OHDG', categories='Control')
draw_boxplot(col='CRP', categories='Gender')
draw_boxplot(col='CRP', categories='Control')
draw_boxplot(col='BMA, %', categories='Gender')
draw_boxplot(col='BMA, %', categories='Control')

draw_scatter(x_n='BMA, %', y_n='8OHDG')
draw_scatter(x_n='CRP', y_n='8OHDG')
draw_scatter(x_n='BMI', y_n='8OHDG')
draw_scatter(x_n='Body Fat, %', y_n='8OHDG')
draw_scatter(x_n='WHR', y_n='8OHDG')
draw_scatter(x_n='Age', y_n='8OHDG')
draw_scatter(x_n='BMA, %', y_n='CRP')
draw_scatter(x_n='BMI', y_n='CRP')
draw_scatter(x_n='Body Fat, %', y_n='CRP')
draw_scatter(x_n='WHR', y_n='CRP')
draw_scatter(x_n='Age', y_n='CRP')
draw_scatter(x_n='BMI', y_n='BMA, %')
draw_scatter(x_n='Body Fat, %', y_n='BMA, %')
draw_scatter(x_n='WHR', y_n='BMA, %')
draw_scatter(x_n='Age', y_n='BMA, %')


# ======================== Study participant characteristics =================================

data_control = data[data['Category'] == 'Control'].select_dtypes('number')
data_obese = data[data['Category'] != 'Control'].select_dtypes('number')
data_ttest = np.array([
    [vals for col, vals in data_control.iteritems()],
    [vals for col, vals in data_obese.iteritems()]
]).T
data_ttest = [ttest_ind(row[0], row[1], equal_var=False) for row in data_ttest]
data_ttest = [get_significance(el[1]) for el in data_ttest]

data_primary = pd.concat([data_control.describe().T[['mean', 'std']],
                          data_obese.describe().T[['mean', 'std']]],
                         axis=1)

data_primary = np.round(data_primary, 2)
data_primary['Welch Test'] = data_ttest
data_primary['Control'] = ["{} ± {}".format(row[0],
                                            row[1]) for ix, row in data_primary.iterrows()]
data_primary['Obese'] = ["{} ± {} {}".format(row[2],
                                             row[3],
                                             row[4]) for ix, row in data_primary.iterrows()]
data_primary = data_primary.ix[:-5]

with pd.ExcelWriter('Means.xlsx') as wr:
    data_primary[['Control', 'Obese']].to_excel(wr)

# ============================ Correlation coefficients =======================================

target_list = ['BMA, %', '8OHDG', 'CRP']
df_list = []
for t in target_list:
    target = [
        data[t],
        data_control[t],
        data_obese[t]
    ]

    data_prep = np.array([
        [vals for col, vals in data.select_dtypes('number').iteritems()],
        [vals for col, vals in data_control.iteritems()],
        [vals for col, vals in data_obese.iteritems()]
    ])
    z = [pearsonr(target[2], el) for el in data_prep[2]]
    data_prep = [[pearsonr(target[ix], series) for series in el] for ix, el in enumerate(data_prep)]
    data_prep = [[(round(row[0], 3), get_significance(row[1])) for row in col] for col in data_prep]

    data_corr = pd.DataFrame(index=data.select_dtypes('number').T.index, columns=['All Subjects', 'Control', 'Obese'])

    for i in range(0, len(data_prep)):
        data_corr.iloc[:, i] = ["{} {}".format(el[0], el[1]) for el in data_prep[i]]

    df_list.append((t, data_corr.ix[:-5]))

with pd.ExcelWriter('Correlation.xlsx') as wr:
    for t, df in df_list:
        df[['All Subjects', 'Control', 'Obese']].to_excel(wr, sheet_name=t)

# ========================================== Models ==================================================

target_dict = {
    'BMA, %': ['Female', 'Obese', 'Severely Obese', 'Morbidly Obese', 'Age', 'BMI', 'WHR', 'Body Fat, %', 'CRP',
               '8OHDG'],
    '8OHDG': ['Female', 'Obese', 'Severely Obese', 'Morbidly Obese', 'Age', 'BMI', 'WHR', 'Body Fat, %', 'CRP',
              'Micronuclei', 'NBUD', 'Binucleated'],
    'CRP': ['Female', 'Obese', 'Severely Obese', 'Morbidly Obese', 'Age', 'BMI', 'WHR', 'Body Fat, %', '8OHDG',
            'Micronuclei', 'NBUD', 'Binucleated']
}

for key in target_dict:
    y = data[key]
    X = data[target_dict[key]]
    p = range(1, len(X.columns) + 1)
    z = []
    for r in range(1, len(p) - 1):
        z += list(combinations(p, r))
    p = []
    for combo in z:
        combo = [0] + list(combo)
        X_mod = sm.add_constant(X).iloc[:, combo]
        fit = sm.OLS(y, X_mod).fit()
        p.append((combo, 1-fit.rsquared_adj, fit.aic))

    p.sort(key=lambda x: (round(x[1], 2), round(x[2], 0)))

    df_list = []
    for config in p[:5]:
        X_mod = sm.add_constant(X).iloc[:, config[0]]
        model = sm.OLS(y, X_mod).fit()

        results = pd.DataFrame(
            data=list(model.params.values) + [model.rsquared, model.rsquared_adj, model.aic, model.fvalue, model.f_pvalue],
            index=list(model.params.index.values) + ['Rsq', 'RsqA', 'AIC', 'FVal', 'FpVal']
        )
        pvals = model.pvalues
        bse = model.bse
        results = np.round(pd.concat([results, pvals, bse], ignore_index=True, axis=1, sort=False), 3)
        results.columns = ['Model', 'pval', 'se']

        results['Model'] = [
            "{:0.3f} ({:0.3f}) {}".format(
                row['Model'],
                row['se'],
                get_significance(row['pval'])
            ) if not np.isnan(row['se']) else "{:0.3f}".format(row['Model']) for ix, row in results.iterrows()]

        df_list.append(results['Model'])

    with pd.ExcelWriter('Regression {}.xlsx'.format(key)) as wr:
        for ix, df in enumerate(df_list):
            df.to_excel(wr, sheet_name=str(ix))
