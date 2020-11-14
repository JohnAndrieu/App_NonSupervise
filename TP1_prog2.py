from sklearn import neighbors, model_selection
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt


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
        debut_entrainement = time.time()
        cl.fit(xtrain, ytrain)
        duree_entrainement = time.time() - debut_entrainement
        debut_prediction = time.time()
        ypred = cl.predict(xtest)
        duree_prediction = time.time() - debut_prediction
        erreur = metrics.zero_one_loss(ytest, ypred)
        ks.append((i, erreur, duree_entrainement, duree_prediction))
        print(f"\t {ks[-1]}")

    ks_liste = list(zip(*ks))

    plt.figure(figsize=(20, 13))

    plt.subplot(3, 1, 1)
    plt.plot(ks_liste[0], ks_liste[3], 'x-', color='red')
    plt.xlabel('k')
    plt.ylabel('Erreur sur le jeu de données')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(ks_liste[0], ks_liste[1], 'x-', color='green')
    plt.xlabel('k')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(ks_liste[0], ks_liste[2], 'x-', color='blue')
    plt.xlabel('k')
    plt.ylabel("Durée de prédiction")
    plt.grid(True)

    plt.savefig('variation_k')
    plt.show()

    echantillons = []
    print("Variations du % train/test : ")
    for i in np.arange(0.7, 0.99, 0.01):
        xtrain_echantillons, xtest_echantillons, ytrain_echantillons, ytest_echantillons = model_selection.train_test_split(
            data, target, train_size=i)
        cl = neighbors.KNeighborsClassifier(3)
        debut_entrainement = time.time()
        cl.fit(xtrain_echantillons, ytrain_echantillons)
        duree_entrainement = time.time() - debut_entrainement
        debut_prediction = time.time()
        ypred = cl.predict(xtest)
        duree_prediction = time.time() - debut_prediction
        erreur = metrics.zero_one_loss(ytest, ypred)
        echantillons.append((i, duree_entrainement, duree_prediction, erreur))
        print(f"\t {echantillons[-1]}")

    sample_list = list(zip(*echantillons))

    plt.figure(figsize=(16, 9))

    plt.subplot(3, 1, 1)
    plt.plot(sample_list[0], sample_list[3], 'x-', color='red')
    plt.xlabel('% train / test')
    plt.ylabel('Erreur sur le jeu de données')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(sample_list[0], sample_list[1], 'x-', color='green')
    plt.xlabel('% train / test')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(sample_list[0], sample_list[2], 'x-', color='blue')
    plt.xlabel('% train / test')
    plt.ylabel("Durée de prédiction")
    plt.grid(True)

    plt.show()

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

    plt.figure(figsize=(16, 9))

    plt.subplot(3, 1, 1)
    plt.plot(dt_list[0], dt_list[3], 'x-', color='red')
    plt.xlabel('distance')
    plt.ylabel('Erreur sur le jeu de données')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(dt_list[0], dt_list[1], 'x-', color='green')
    plt.xlabel('distance')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(dt_list[0], dt_list[2], 'x-', color='blue')
    plt.xlabel('distance')
    plt.ylabel("Durée de prédiction")
    plt.grid(True)

    plt.show()

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

    plt.figure(figsize=(16, 9))

    plt.subplot(3, 1, 1)
    plt.plot(training_list[0], training_list[3], 'x-', color='red')
    plt.xlabel("taille du jeu d'entraînement")
    plt.ylabel('Erreur sur le jeu de données')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(training_list[0], training_list[1], 'x-', color='green')
    plt.xlabel("taille du jeu d'entraînement")
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(training_list[0], training_list[2], 'x-', color='blue')
    plt.xlabel("taille du jeu d'entraînement")
    plt.ylabel("Durée de prédiction")
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
