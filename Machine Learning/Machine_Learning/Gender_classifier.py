import numpy
import scipy 
import joblib
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics

clf_tree = tree.DecisionTreeClassifier()
clf_knn=KNeighborsClassifier()
clf_per=Perceptron()
clf_svc=sklearn.svm.SVC()

# [height, weight, shoe_size]
# https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

data_for_prediction=[[190,70,43]]

# fitting data
clf_tree.fit(X,Y)
clf_knn.fit(X,Y)
clf_per.fit(X,Y)
clf_svc.fit(X,Y)
             
results_clf_tree = clf_tree.predict(data_for_prediction)
print(results_clf_tree)

tree_res=clf_tree.predict(X)
tree_score=sklearn.metrics.accuracy_score(Y,tree_res)

knn_res=clf_knn.predict(X)
knn_score=sklearn.metrics.accuracy_score(Y,knn_res)

per_res=clf_per.predict(X)
per_score=sklearn.metrics.accuracy_score(Y, per_res)

svc_res=clf_svc.predict(X)
svc_score=sklearn.metrics.accuracy_score(Y, svc_res)

res=[("Tree",tree_res,tree_score),("K Neighbors", knn_res,knn_score),("Perceptron",per_res,per_score),("SVC",svc_res,svc_score)]

print(res)

sorted_res=sorted(res,key=lambda x:x[2])

print("Best score:\n")
best_scr=[i[0] for i in sorted_res]
print(best_scr)