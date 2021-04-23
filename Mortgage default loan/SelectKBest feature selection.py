import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#import data
dataframe = pd.read_csv("../Mortgage.csv")
array = dataframe.values
names = ['Age', 'Ed', 'Employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'outcome']

#format y
y_pos = np.arange(len(names))

#selct all attributes except the outcome is empty
result = dataframe[dataframe.outcome != "' '"]
features = result.loc[:, result.columns != 'outcome']
outcome = result.loc[:, result.columns == 'outcome']
outcome = pd.to_numeric(outcome['outcome'])

# select the best feature
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(result, outcome)
# Summarize scores
print(fit.scores_)

# Summarize selected features and display
plt.xticks(y_pos,names)
plt.bar([i for i in range(len(fit.scores_))], fit.scores_)
plt.show()


