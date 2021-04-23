from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs

#load the data and categorize them , then train the data
dataframe = arff.loadarff('Heart.arff')
df = pd.DataFrame(dataframe[0])
names = ['age','gender','chest pain type','resting blood pressure','cholestoral','blood sugar','ECG results','max heart rate','chest pain after exercise','peak heart rate after exercise','heart rate variation','status of blood vessels','blood supply status','class']
X = df.loc[:,df.columns != 'class']
Y = df.loc[:,df.columns == 'class']
Y = LabelEncoder().fit_transform(np.ravel(Y))

#set the random state = 999
pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=0.25,random_state=999)

# feature extraction by using Decision tree
model = ExtraTreesClassifier()

#10 fold cross validation
scores = cross_val_score(model, pred_test, tar_test, cv=10).mean()
print("accuracy score ", scores )
model = model.fit(X, Y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')

#display the data
plt.show()
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');