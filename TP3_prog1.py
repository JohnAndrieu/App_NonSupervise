from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn


def plot_fig(data):
    plt.figure(figsize=(20, 13))

    plt.subplot(3, 1, 1)
    plt.plot(data[0], data[3], 'x-', color='red')
    plt.xlabel('k')
    plt.ylabel('Erreur sur le jeu de données')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(data[0], data[1], 'x-', color='green')
    plt.xlabel('k')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(data[0], data[2], 'x-', color='blue')
    plt.xlabel('k')
    plt.ylabel("Durée de prédiction")
    plt.grid(True)

    plt.show()


def main():
    mnist = fetch_openml(name='mnist_784')
    echantillon = np.random.randint(70000, size=5000)
    data = mnist.data[echantillon]
    target = mnist.target[echantillon]

    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(xtrain, ytrain)
    error = 1 - classifier.score(xtest, ytest)
    print(f"Score SVM linéaire : {error}")

    kernels = []
    print("Modification du kernel : ")
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        classifier = svm.SVC(kernel=kernel)

        start_training = time.time()
        classifier.fit(xtrain, ytrain)
        final_training = time.time() - start_training

        start_prediction = time.time()
        ypred = classifier.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        kernels.append((kernel, final_training, final_prediction, error))
        print(f"\t {kernels[-1]}")

    kernels_liste = list(zip(*kernels))

    plot_fig(kernels_liste)

    tol = []
    print("Evolution de la tolérance : ")
    for tolerance in np.linspace(0.1, 1.0, num=5):
        svm = svm.SVC(C=tolerance)

        start_training = time.time()
        svm.fit(xtrain, ytrain)
        final_training = time.time() - start_training

        start_prediction = time.time()
        ypred = svm.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        error_training = svm.score(xtrain, ytrain)
        tol.append((tolerance, final_training, final_prediction, error, error_training))
        print(f"\t {tol[-1]}")

    tol_list = list(zip(*tol))

    plot_fig(tol_list)

    plt.figure(figsize=(19, 9))
    plt.plot(tol_list[0], tol_list[3], 'x-', color='blue') # erreur de test
    plt.plot(tol_list[0], tol_list[-1], 'x-', color='orange') # erreur d'entrainement
    plt.grid(True)
    plt.show()

    best_kernel = 'rbf'
    best_tolerance = 1.0
    best_svm = svm.SVC(kernel=best_kernel, C=best_tolerance)

    start_training = time.time()
    best_svm.fit(xtrain, ytrain)
    best_final_entrainement = time.time() - start_training

    start_prediction = time.time()
    ypred = best_svm.predict(xtest)
    best_final_prediction = time.time() - start_prediction

    cross_val = model_selection.cross_val_score(best_svm, data, target, cv=10)
    meilleure_erreur = 1 - np.mean(cross_val)

    print(f"Durée de l'entraînement : {best_final_entrainement}")
    print(f"Durée de la prédiction : {best_final_prediction}")
    print(f"Erreur : {meilleure_erreur}")

    cm = confusion_matrix(ytest, ypred)
    df_cm = pd.DataFrame(cm, columns=np.unique(ytest), index=np.unique(ytest))
    df_cm.index.name = 'Valeur réelle'
    df_cm.columns.name = 'Valeur prédite'
    plt.figure(figsize=(16, 9))
    sn.heatmap(df_cm, cmap="Blues", annot=True)
    plt.show()


if __name__ == "__main__":
    main()
