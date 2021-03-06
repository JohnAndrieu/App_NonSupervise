from sklearn import neighbors, model_selection, metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import time
import matplotlib.pyplot as plt
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

    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)
    clf = neighbors.KNeighborsClassifier(n_neighbors=10)
    clf.fit(xtrain, ytrain)

    # Classe de l’image 4 et sa classe prédite.
    print(mnist.target[4])
    print(clf.predict(mnist.data[4].reshape((1, -1)))) #Reshape de notre jeu de donnée en 2D

    # Score échantillon de test et d'apprentissage
    print(f"Score sur le jeu de test : {str(clf.score(xtest, ytest))}")
    print(f"Score sur le jeu d'entraînement : {str(clf.score(xtrain, ytrain))}")

    # Taux d'erreur sur les données de test et d'apprentissage
    print(f"Erreur sur le jeu de test : {str(1 - clf.score(xtest, ytest))}")
    print(f"Erreur sur le jeu d'entraînement : {str(1 - clf.score(xtrain, ytrain))}")

    ks = []
    print("Variations de k :")
    for i in range(2, 16):
        cl = neighbors.KNeighborsClassifier(i)

        start_train = time.time()
        cl.fit(xtrain, ytrain)
        final_train = time.time() - start_train

        start_prediction = time.time()
        ypred = cl.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        ks.append((i, error, final_train, final_prediction))
        print(f"\t {ks[-1]}")

    ks_liste = list(zip(*ks))

    plot_fig(ks_liste)

    samples = []
    print("Variations du % train/test : ")
    for i in np.arange(0.7, 0.99, 0.01):
        xtrain_echantillons, xtest_echantillons, ytrain_echantillons, ytest_echantillons = model_selection.train_test_split(
            data, target, train_size=i)
        cl = neighbors.KNeighborsClassifier(3)

        start_train = time.time()
        cl.fit(xtrain_echantillons, ytrain_echantillons)
        final_train = time.time() - start_train

        start_prediction = time.time()
        ypred = cl.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        samples.append((i, final_train, final_prediction, error))
        print(f"\t {samples[-1]}")

    sample_list = list(zip(*samples))

    plot_fig(sample_list)

    dt = []
    print("Evolution de la distance dt : ")
    for i in range(1, 3):
        #on fixe le nombre de voisins à 3 car c'est l'optimal
        cl = neighbors.KNeighborsClassifier(3, p=i)

        start_train = time.time()
        cl.fit(xtrain, ytrain)
        final_train = time.time() - start_train

        start_prediction = time.time()
        ypred = cl.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        dt.append((i, final_train, final_prediction, error))
        print(f"\t {dt[-1]}")

    dt_list = list(zip(*dt))

    plot_fig(dt_list)

    training = []
    for i in range(100, 4000, 100):
        sample_train = np.random.randint(4000, size=i)
        xtrain_var = xtrain[sample_train]
        ytrain_var = ytrain[sample_train]
        cl = neighbors.KNeighborsClassifier(3)

        start_train = time.time()
        cl.fit(xtrain_var, ytrain_var)
        final_train = time.time() - start_train

        start_prediction = time.time()
        ypred = cl.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        training.append((i, final_train, final_prediction, error))

        print(f"\t {training[-1]}")

    training_list = list(zip(*training))

    plot_fig(training_list)

    best_k = 3
    best_p = 2
    best_clf = neighbors.KNeighborsClassifier(best_k, p=best_p)

    start_train = time.time()
    best_clf.fit(xtrain, ytrain)
    best_final_train = time.time() - start_train

    start_prediction = time.time()
    ypred = best_clf.predict(xtest)
    best_final_prediction = time.time() - start_prediction

    cross_val = model_selection.cross_val_score(best_clf, data, target, cv=10)
    best_error = 1 - np.mean(cross_val)

    print(f"Durée de l'entraînement : {best_final_train}")
    print(f"Durée de la prédiction : {best_final_prediction}")
    print(f"Erreur : {best_error}")

    cm = confusion_matrix(ytest, ypred)
    df_cm = pd.DataFrame(cm, columns=np.unique(ytest), index=np.unique(ytest))
    df_cm.index.name = 'Valeur réelle'
    df_cm.columns.name = 'Valeur prédite'
    plt.figure(figsize=(16, 9))
    sn.heatmap(df_cm, cmap="Blues", annot=True)
    plt.show()

if __name__ == "__main__":
    main()
