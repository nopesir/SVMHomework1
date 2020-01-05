from matplotlib.colors import Normalize
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB

from lib import *
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt


X, y = datasets.load_wine(return_X_y=True)

pipe = make_pipeline(StandardScaler(), PCA(n_components=2))

X = pipe.fit_transform(X)

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2857, random_state=0)

print("\n********************************")
print("TASK: kNN classification")
print("********************************\n")

best_score = 0
best_k = 1
knn_models = []
knn_accuracies = []

for k in [1, 3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    knn_models.append(knn)
    accuracy = knn.score(X_val, y_val)
    knn_accuracies.append(accuracy)
    if accuracy > best_score:
        best_score = accuracy
        best_k = k

knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn.fit(X_trainval, y_trainval)
test_score = knn.score(X_test, y_test)
predictions = knn.predict(X_test)


print('Prediction accuracy on validation set found with k = ' + str(best_k))
print('{:.2%}\n'.format(best_score))

print('Prediction accuracy on test set for k=' + str(best_k))
print('{:.2%}\n'.format(test_score))
print('report: \n' + classification_report(y_test, predictions))

print("\n********************************")
print("TASK: Linear SVM classification")
print("********************************\n")

best_score = 0
best_C = 0
linearsvc_models = []
linearsvc_accuracies = []

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    svc = SVC(C=C, kernel='linear')
    svc.fit(X_train, y_train)
    linearsvc_models.append(svc)
    accuracy = svc.score(X_val, y_val)
    linearsvc_accuracies.append(accuracy)
    if accuracy > best_score:
        best_score = accuracy
        best_C = C


svc = SVC(C=best_C, kernel='linear')
svc.fit(X_trainval, y_trainval)
test_score = svc.score(X_test, y_test)
predictions = svc.predict(X_test)

print('Prediction accuracy on validation set found with C = ' + str(best_C))
print('{:.2%}\n'.format(best_score))

print('Prediction accuracy on test set for C=' + str(best_C))
print('{:.2%}\n'.format(test_score))
print('report: \n' + classification_report(y_test, predictions))

print("\n********************************")
print("TASK: RBF SVM classification")
print("********************************\n")

best_score = 0
best_C = 0
rbfsvc_models = []
rbfsvc_accuracies = []

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    svc = SVC(C=C, kernel='rbf', gamma='auto')
    svc.fit(X_train, y_train)
    rbfsvc_models.append(svc)
    accuracy = svc.score(X_val, y_val)
    rbfsvc_accuracies.append(accuracy)
    if accuracy > best_score:
        best_score = accuracy
        best_C = C


svc = SVC(C=best_C, kernel='rbf', gamma='auto')
svc.fit(X_trainval, y_trainval)
test_score = svc.score(X_test, y_test)
predictions = svc.predict(X_test)

print('Prediction accuracy on validation set found with C = ' + str(best_C))
print('{:.2%}\n'.format(best_score))

print('Prediction accuracy on test set for C=' + str(best_C))
print('{:.2%}\n'.format(test_score))
print('report: \n' + classification_report(y_test, predictions))

print("\n********************************")
print("TASK: RBF SVM classification with grid-search")
print("********************************\n")

best_score = 0
best_C = 0
best_gamma = 0
gridsvc_models = []
gridsvc_accuracies = []

for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        svc = SVC(C=C, kernel='rbf', gamma=gamma)
        svc.fit(X_train, y_train)
        gridsvc_models.append(svc)
        accuracy = svc.score(X_val, y_val)
        gridsvc_accuracies.append(accuracy)
        if accuracy > best_score:
            best_score = accuracy
            best_C = C
            best_gamma = gamma

svc = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
svc.fit(X_trainval, y_trainval)
test_score = svc.score(X_test, y_test)
predictions = svc.predict(X_test)

print('Prediction accuracy on validation set found with C = ' + str(best_C) + ' and gamma = ' + str(best_gamma))
print('{:.2%}\n'.format(best_score))

print('Prediction accuracy on test set for C = ' + str(best_C) + ' and gamma = ' + str(best_gamma))
print('{:.2%}\n'.format(test_score))
print('report: \n' + classification_report(y_test, predictions))


print("\n********************************")
print("TASK: RBF SVM classification with k-fold")
print("********************************\n")

X, y = datasets.load_wine(return_X_y=True)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

pipe = Pipeline([
    ('mm', MinMaxScaler()),
    ('pca', PCA(n_components=2)),
    ('svc', SVC(kernel='rbf'))
])

params = {
    'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'svc__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

kfold = KFold(5)

search = GridSearchCV(
    estimator=pipe,
    param_grid=params,
    n_jobs=-1,
    iid=False,
    cv=kfold.split(X=X_trainval, y=y_trainval),
    return_train_score=True
)

kfoldSVC = search.fit(X_trainval, y_trainval)

result = kfoldSVC.best_estimator_.score(X_test, y_test)
predictions = kfoldSVC.best_estimator_.predict(X_test)

print("Best average score found for C = " + str(kfoldSVC.best_params_['svc__C']) + " and gamma = " + str(
    kfoldSVC.best_params_['svc__gamma'])
      + "\n-> " + '{:.2%}\n'.format(kfoldSVC.best_score_))

print("Score applying it on the test set -> " + '{:.2%}\n'.format(result))
print('report: \n' + classification_report(y_test, predictions))
