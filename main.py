from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, KFold
from lib import *
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


# 1. Load Wine dataset
# 2. Select the first two attributes for a 2D representation of the image.
X, y = datasets.load_wine(return_X_y=True)
X = X[:, 0:2]

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

print('Best prediction accuracy on validation set found with k = ' + str(best_k))
print('{:.2%}\n'.format(best_score))

plotModels(fignum=0, X=X_train, Y=y_train, models=knn_models, titles=[
           'k=1', 'k=3', 'k=5', 'k=7'], n=2)

plotAccuracyComparison(fignum=1, the_list=[1, 3, 5, 7], accuracy_list=knn_accuracies,
                       x_label='k parameter', title='kNN: Accuracy on validation set for any k', type='linear')


knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
knn.fit(X_trainval, y_trainval)
test_score = knn.score(X_test, y_test)
predictions = knn.predict(X_test)



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

print('Prediction accuracy on validation set found with C = ' + str(best_C))
print('{:.2%}\n'.format(best_score))

plotModels(fignum=10, X=X_train, Y=y_train, models=linearsvc_models, titles=['C=0.001', 'C=0.01', 'C=0.1', 'C=1', 'C=10', 'C=100', 'C=1000'], n=3)

plotAccuracyComparison(fignum=11, the_list=[0.001, 0.01, 0.1, 1, 10, 100, 1000], accuracy_list=linearsvc_accuracies,
                       x_label='C parameter', title='Linear SVM: Accuracy on validation set for any C', type='log')


svc = SVC(C=best_C, kernel='linear')
svc.fit(X_trainval, y_trainval)
test_score = svc.score(X_test, y_test)
predictions = svc.predict(X_test)



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

print('Prediction accuracy on validation set found with C = ' + str(best_C))
print('{:.2%}\n'.format(best_score))

plotModels(fignum=12, X=X_train, Y=y_train, models=rbfsvc_models, titles=['C=0.001', 'C=0.01', 'C=0.1', 'C=1', 'C=10', 'C=100', 'C=1000'], n=3)

plotAccuracyComparison(fignum=13, the_list=[0.001, 0.01, 0.1, 1, 10, 100, 1000], accuracy_list=rbfsvc_accuracies,
                       x_label='C parameter', title='RBF SVM: Accuracy on validation set for any C', type='log')


svc = SVC(C=best_C, kernel='rbf', gamma='auto')
svc.fit(X_trainval, y_trainval)
test_score = svc.score(X_test, y_test)
predictions = svc.predict(X_test)


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

print('Best prediction accuracy on validation set found with C = ' + str(best_C) + ' and gamma = ' + str(best_gamma))
print('{:.2%}\n'.format(best_score))


scores = np.array(gridsvc_accuracies).reshape(len([0.001, 0.01, 0.1, 1, 10, 100, 1000]), len([0.001, 0.01, 0.1, 1, 10, 100, 1000]))

plt.figure(num=9, figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
           norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len([0.001, 0.01, 0.1, 1, 10, 100, 1000])), [0.001, 0.01, 0.1, 1, 10, 100, 1000], rotation=45)
plt.yticks(np.arange(len([0.001, 0.01, 0.1, 1, 10, 100, 1000])), [0.001, 0.01, 0.1, 1, 10, 100, 1000])
plt.title('Grid-search: Validation accuracy')

svc = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
svc.fit(X_trainval, y_trainval)
test_score = svc.score(X_test, y_test)
predictions = svc.predict(X_test)



print('Prediction accuracy on test set for C = ' + str(best_C) + ' and gamma = ' + str(best_gamma))
print('{:.2%}\n'.format(test_score))
print('report: \n' + classification_report(y_test, predictions))

print("\n********************************")
print("TASK: RBF SVM classification with k-fold")
print("********************************\n")

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

pipe = Pipeline([
    ('mm', MinMaxScaler()),
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



print("Best average score found for C = " + str(kfoldSVC.best_params_['svc__C']) + " and gamma = " + str(kfoldSVC.best_params_['svc__gamma'])
+ "\n-> " + '{:.2%}\n'.format(kfoldSVC.best_score_))

print("Score applying it on the test set -> " + '{:.2%}\n'.format(result))
print('report: \n' + classification_report(y_test, predictions))

print("\n********************************")
print("TASK: REPEATING USING 2-PCA AND NORMALIZATION")
print("********************************\n")

import withPCA

plt.show()
exit()
