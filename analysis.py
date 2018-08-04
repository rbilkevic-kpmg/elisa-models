import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_excel('BMA analysis.xlsx', index_col=0, na_values=np.float64('nan'))

data.plot(x='CRP', y='8OHDG', kind='scatter')
data['Is_Healthy'] = [1 if row['Category'] == 'Healthy' else 0 for ix, row in data.iterrows()]
data['Female'] = [1 if row['Gender'] == 'Female' else 0 for ix, row in data.iterrows()]
data = data.dropna(axis=0)
print(data)

# data.boxplot(column='Body Fat, %', by='Gender')
plt.yscale('log')
plt.xscale('log')
plt.show()

X = data[["Is_Healthy", "Female", "DNA DAMAGE, %", "CRP", "WHR", "Age", "BMI"]]
y = data["8OHDG"]

# X = data[["Is_Healthy", "Female", "DNA DAMAGE, %", "8OHDG", "WHR", "Age", "BMI"]]
# y = data["CRP"]

# X = data[["Is_Healthy", "Female", "8OHDG", "CRP", "WHR", "Age", "BMI"]]
# y = data["DNA DAMAGE, %"]

X = sm.add_constant(X)
# Note the difference in argument order
model = sm.OLS(y, X).fit()
# predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())
