from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns

#qestion B
#load data
dataframe = pd.read_csv("../Mortgage.csv")
names = ['Age', 'Ed', 'Employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'outcome']
y_pos = np.arange(len(names))

#selct all attributes except the outcome is empty
result = dataframe[dataframe.outcome != "' '"]
features = result.loc[:, result.columns != 'outcome']

outcome = result.loc[:, result.columns == 'outcome']
outcome = pd.to_numeric(outcome['outcome'])
print(outcome)

#select the four most influential attributes from question 1 , by dropping the least influential attributes
dataframe.drop(['Age', 'Ed', 'income', 'othdebt'], axis='columns', inplace=True)

#train the data
pred_train, pred_test, tar_train, tar_test  = train_test_split(features, outcome, test_size=0.2)

#First parameter , by setting the max_leaf_node = 15
model = tree.DecisionTreeClassifier(max_leaf_nodes=15)
model.fit(pred_train,tar_train)
score = cross_val_score(model, pred_test, tar_test, cv=10).mean()
print("cross validation score ",score)
print("leaf nodes count  ",model.tree_.node_count)
print("max leaf nodes   ",model.max_leaf_nodes)
print("model depth " ,model.get_depth())
print("model params" , model.get_params())
print("max_leaf_nodes = 15 result  : " , model.score(pred_test,tar_test))
print("*************************")

#Second parameter , by setting the max_depth =4
model2 = tree.DecisionTreeClassifier(max_depth=4)
model2.fit(pred_train,tar_train)
score2 = cross_val_score(model2, pred_test, tar_test, cv=10).mean()
print("cross validation score ",score2)
print("leaf nodes count  ",model2.tree_.node_count)
print("max leaf nodes   ",model2.max_leaf_nodes)
print("model depth " ,model2.get_depth())
print("model params" , model2.get_params())
print("max_depth_4 reuslt : " , model2.score(pred_test,tar_test))
print("*************************")

#get the best outcome by using grid search CV
print("*****best outcome*********")
max_depth = []
for i in range(1,4):
    max_depth.append(i)
param1 = {'max_depth': max_depth}
grid1 = GridSearchCV(DecisionTreeClassifier(), param_grid=param1, cv=10)
grid1.fit(pred_train, tar_train)
print('Best params:', grid1.best_params_, 'Best score:', grid1.best_score_)

#check the leaf nodes
max_leaf_nodes = []
for i in range(2,15):
    max_leaf_nodes.append(i)
param2 = {'max_leaf_nodes': max_leaf_nodes,}
grid2 = GridSearchCV(DecisionTreeClassifier(), param_grid=param2, cv=10)
grid2.fit(pred_train, tar_train)
print('Best params2 :', grid2.best_params_, 'Best score:', grid2.best_score_)

#question d
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import precision_score, recall_score

predictions = model.predict(pred_test)
print("prediction : " , predictions)
prob = model.predict_proba(pred_test)
TN, FP, FP, TP = confusion_matrix(tar_test,predictions).ravel()
cm = confusion_matrix(tar_test,predictions) #
print("True Negatives: ",TN)
print("False Positives: ",FP)
print("False Negatives: ",FP)
print("True Positives: ",TP)
ls =["false ","true"]
f,ax=plt.subplots()
sns.heatmap(cm, annot=True,ax=ax)
ax.set_title('cm')
ax.set_xlabel('predict')
ax.set_ylabel('true')
# plt.show()
print("Accuracy score of our model with Decision Tree:", accuracy_score(tar_test, predictions))
for x in range(2):
    precision = precision_score(y_true=tar_test, y_pred=predictions,average='binary', pos_label=x)
    print("Precision score for class", x, "with Decision Tree :", precision)
    recall = recall_score(y_true=tar_test, y_pred=predictions,average='binary', pos_label=x)
    print("Recall score for class", x, " with Decision Tree :", recall)
figure,ax = plot_confusion_matrix(conf_mat=cm,class_names=ls,show_absolute=False,show_normed=True,colorbar=True)
# plt.show()

#question e
print("Accuracy score of our model with decision tree is :", accuracy_score(tar_test, predictions))