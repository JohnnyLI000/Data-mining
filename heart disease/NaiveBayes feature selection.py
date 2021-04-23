from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#load the data
dataframe = arff.loadarff('Heart.arff')
df = pd.DataFrame(dataframe[0])
print(df)
#categorize each attributes  , select every attributes except the class attribute
names = ['age','gender','chest pain type','resting blood pressure','cholestoral','blood sugar','ECG results','max heart rate','chest pain after exercise','peak heart rate after exercise','heart rate variation','status of blood vessels','blood supply status']
X = df.loc[:,df.columns != 'class']
Y = df.loc[:,df.columns == 'class']
Y = LabelEncoder().fit_transform(np.ravel(Y))

#train the data
pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=0.25,random_state=999)

#use guassian naive bayes
gaussian_nb = GaussianNB()
gaussian_nb.fit(pred_train,tar_train)

#get each attribute importance
imps = permutation_importance(gaussian_nb, pred_test, tar_test)
print(imps.importances_mean)

#display and print out each attribute
for i,v in enumerate(imps.importances_mean):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(imps.importances_mean))], imps.importances_mean)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12],names, rotation=90)
plt.show()
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
