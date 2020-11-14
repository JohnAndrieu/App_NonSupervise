from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn


def main():
    mnist = fetch_openml(name='mnist_784')
    echantillon = np.random.randint(70000, size=5000)
    data = mnist.data[echantillon]
    target = mnist.target[echantillon]

    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

    classifier = svm.SVC(kernel='linear')
    classifier.fit(xtrain, ytrain)
    Y_pred = classifier.predict(xtest)
    error = 1 - classifier.score(xtest, ytest)
    print(f"Score SVM linéaire : {error}")

    kernels = []
    print("Variations du kernel : ")
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        classifier = svm.SVC(kernel=kernel)

        debut_entrainement = time.time()
        classifier.fit(xtrain, ytrain)
        duree_entrainement = time.time() - debut_entrainement

        debut_prediction = time.time()
        ypred = classifier.predict(xtest)
        duree_prediction = time.time() - debut_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        kernels.append((kernel, duree_entrainement, duree_prediction, error))
        print(f"\t {kernels[-1]}")

    kernels_liste = list(zip(*kernels))

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(kernels_liste[0], kernels_liste[3], 'x-', color='red')
    plt.xlabel('kernel')
    plt.ylabel('Erreur')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(kernels_liste[0], kernels_liste[1], 'x-', color='green')
    plt.xlabel('kernel')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(kernels_liste[0], kernels_liste[2], 'x-', color='blue')
    plt.xlabel('kernel')
    plt.ylabel('Durée de prédiction')
    plt.grid(True)

    plt.show()

    tolerances = []
    print("Variations de la tolérance : ")
    for tolerance in np.linspace(0.1, 1.0, num=5):
        clsvm = svm.SVC(C=tolerance)

        debut_entrainement = time.time()
        clsvm.fit(xtrain, ytrain)
        duree_entrainement = time.time() - debut_entrainement

        debut_prediction = time.time()
        ypred = clsvm.predict(xtest)
        duree_prediction = time.time() - debut_prediction

        erreur = metrics.zero_one_loss(ytest, ypred)
        erreur_entrainement = clsvm.score(xtrain, ytrain)
        tolerances.append((tolerance, duree_entrainement, duree_prediction, erreur, erreur_entrainement))
        print(f"\t {tolerances[-1]}")

    tolerances_liste = list(zip(*tolerances))

    fig = plt.figure(figsize=(19, 9))
    plt.subplot(3, 1, 1)
    plt.plot(tolerances_liste[0], tolerances_liste[3], 'x-', color='red')
    plt.xlabel('tolérance')
    plt.ylabel('Erreur')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(tolerances_liste[0], tolerances_liste[1], 'x-', color='green')
    plt.xlabel('tolérance')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(tolerances_liste[0], tolerances_liste[2], 'x-', color='blue')
    plt.xlabel('tolérance')
    plt.ylabel('Durée de prédiction')
    plt.grid(True)

    plt.show()

    fig = plt.figure(figsize=(19, 9))
    # erreur de test
    plt.plot(tolerances_liste[0], tolerances_liste[3], 'x-', color='blue')
    # erreur d'entrainement
    plt.plot(tolerances_liste[0], tolerances_liste[-1], 'x-', color='orange')
    plt.grid(True)
    plt.show()

    print(tolerances_liste[3])
    print(tolerances_liste[-1])

    cm = confusion_matrix(ytest, Y_pred)
    df_cm = pd.DataFrame(cm, columns=np.unique(ytest), index=np.unique(ytest))
    df_cm.index.name = 'Valeur réelle'
    df_cm.columns.name = 'Valeur prédite'
    plt.figure(figsize=(16, 9))
    sn.heatmap(df_cm, cmap="Blues", annot=True)
    plt.show()


if __name__ == "__main__":
    main()
