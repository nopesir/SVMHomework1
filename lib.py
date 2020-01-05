import math
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm, neighbors
import seaborn as sns



def splitDataIntoTVT(SamplesMatrix, LabelsMatrix, train_perc, val_perc, test_perc):
    try:
        if train_perc + val_perc + test_perc != 1:
            raise Exception(
                "\n[EXCEPTION] Dataset splitting cannot e performed, proportions are not correct. Exiting program "
                "with status 1\n")
    except Exception as exc:
        print(exc.args[0])
        exit(1)

    # first split into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(SamplesMatrix, LabelsMatrix, test_size=test_perc, random_state=365)

    if (val_perc == 0.0):
        # return only train and test sets
        return X_train, X_test, Y_train, Y_test
    else:
        # split again the train set into train and validation, return train, validation and test sets
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=(val_perc/(val_perc+train_perc)), random_state=365)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test


def createHyperparametersLists(c_exp_min, c_exp_max, gamma_exp_min, gamma_exp_max):

    # our range of values for C
    C_list = []
    for i in range(c_exp_min, c_exp_max):
        C_list.append(10 ** i)
    print("C values: ", C_list, "\n")

    # our range of values for Gamma (used in RBF svm)
    Gamma_list = []
    for i in range(gamma_exp_min, gamma_exp_max):
        Gamma_list.append(10 ** i)
    print("Gamma values: ", Gamma_list, "\n\n\n")

    return C_list, Gamma_list


def calcPercentageWrongPredictions(Y_known, Y_predicted):
    
    Y_predicted.reshape(Y_predicted.shape[0], 1)  # reshape to have same shape as Y_known
    i = 0
    num_mislabeled_points = 0

    while i != Y_known.shape[0]:
        if Y_predicted[i] != Y_known[i]:
            num_mislabeled_points += 1
        i += 1

    return (100 * num_mislabeled_points) / Y_known.shape[
        0]  # calc percentage, divide by number of samples (= rows of Y_known)



def applySVM_C_Gamma(kernel, C_list, Gamma_list, X_train, Y_train, X_test, Y_test):

    models = []  # list of all the different SVM that we had trained
    highest_accuracy = -1  # the highest accuracy found
    hyperparameters = np.empty(
        shape=(len(C_list), len(Gamma_list)))  # matrix of accuracy for each combination of hyperparameters

    for c in C_list:
        for gamma in Gamma_list:

            print("Performing fit and predict for RBF SVM with C =", c, "and Gamma =", gamma)

            linearSVM = svm.SVC(C=c, gamma=gamma, kernel=kernel)
            linearSVM.fit(X_train, Y_train.ravel())  # fit on training data
            Y_predicted = linearSVM.predict(X_test)  # predict evaluation data
            accuracy = linearSVM.score(X_test, Y_test)  # get accuracy level
            models.append(linearSVM)  # save model so that we can plot later
            print(" -> Percentage of mislabeled points: %.1f%%" % (
            calcPercentageWrongPredictions(Y_test, Y_predicted)))
            print(" -> Gives a mean accuracy of: %.3f" % accuracy, "\n")

            if (accuracy > highest_accuracy):
                highest_accuracy = accuracy

            hyperparameters[C_list.index(c), Gamma_list.index(
                gamma)] = accuracy  # save accuracy for this combination of hyperparameters

    return models, hyperparameters, highest_accuracy


# stolen from scikit examples: https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def make_meshgrid(x, y, h=.02):
    
    # calc range max and min values
    x_min, x_max = min(x) - 1, max(x) + 1
    y_min, y_max = min(y) - 1, max(y) + 1

    # Return coordinate matrices from coordinate vectors.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return xx, yy


# stolen from scikit examples: https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def plot_contours(axes, model, xx, yy, **params):
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = axes.contourf(xx, yy, Z, **params)
    return out


def plotModel(fignum, X, Y, model, title, pca):
    
    # create a figure and a set of subplots, return a figure and an array of Axes objects
    fig, axes = plt.subplots(num=fignum, figsize=(10, 10), nrows=1, ncols=1)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)

    # obtain a coordinate matrices to use as grid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # assemble plot
    plot_contours(axes, model, xx, yy, cmap=plt.cm.bone, alpha=0.8)
    axes.scatter(X0, X1, c=Y, cmap=plt.cm.bone, s=20, edgecolors='k')
    axes.set_xlim(xx.min(), xx.max())
    axes.set_ylim(yy.min(), yy.max())
    if pca:
        axes.set_xlabel('PCA 1st')
        axes.set_ylabel('{PCA 2nd}')
    else:
        axes.set_xlabel('Alcohol')
        axes.set_ylabel('Malic acid')

    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_title(title)


def plotModels(fignum, X, Y, models, titles, n):
    
    # create a figure and a set of subplots, return a figure and an array of Axes objects
    fig, sub = plt.subplots(figsize=(12, 12), num=fignum, nrows=n, ncols=n)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # obtain a coordinate matrices to use as grid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # use zip to obtain an iterable collection
    for model, title, axes in zip(models, titles, sub.flatten()):
        plot_contours(axes, model, xx, yy, cmap=plt.cm.bone, alpha=0.8)
        axes.scatter(X0, X1, c=Y, cmap=plt.cm.bone, s=20, edgecolors='k')
        axes.set_xlim(xx.min(), xx.max())
        axes.set_ylim(yy.min(), yy.max())
        axes.set_xlabel('Alcohol')
        axes.set_ylabel('Malic acid')
        axes.set_xticks(())
        axes.set_yticks(())
        axes.set_title(title)


def plotAccuracyComparison(fignum, the_list, accuracy_list, x_label, title, type):

    # plot C values and the respective accuracy
    plt.figure(fignum)
    plt.plot(the_list, accuracy_list, '-o')
    plt.xticks(np.array(the_list))
    plt.stem(the_list, accuracy_list, use_line_collection=True)
    plt.ylim(0, 1)
    plt.xscale(type)
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.suptitle(title)

    ymax = max(accuracy_list)
    xpos = accuracy_list.index(ymax)
    xmax = the_list[xpos]

    plt.annotate('MAX', xy=(xmax, ymax), xytext=(xmax, ymax + 0.05))



def selectBestHyperparameters(C_list, Gamma_list, hyperparameters, highest_accuracy):

    C_bests_list = []
    Gamma_bests_list = []

    print("\n------------------------------------------------")
    print("Highest accuracy = %.3f" % highest_accuracy, ", obtained for hyperparameters =")

    for c in C_list:
        for gamma in Gamma_list:
            if (hyperparameters[C_list.index(c), Gamma_list.index(gamma)] == highest_accuracy):
                print("-> C=", c, ", Gamma=", gamma)
                C_bests_list.append(c)  # get all the C values that gave maximum accuracy
                Gamma_bests_list.append(gamma)  # and corresponding gamma

    # Select a C value from the best, we take the smallest one, together with the corresponding value of gamma
    C_best = min(C_bests_list)
    Gamma_best = Gamma_bests_list[C_bests_list.index(C_best)]

    print("Selecting as best Hyperparameters: C=", C_best, ", Gamma=", Gamma_best)

    return C_best, Gamma_best


# Utility function to move the midpoint of a colormap to be around
# the values of interest.
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        super().__init__(vmin, vmax, clip)
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))