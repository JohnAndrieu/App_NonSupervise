from sklearn import neural_network
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    mnist = fetch_openml(name='mnist_784')
    echantillon = np.random.randint(70000, size=5000)
    data = mnist.data[echantillon]
    target = mnist.target[echantillon]

    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(50))
    mlp.fit(xtrain, ytrain)
    score = mlp.score(xtest, ytest)
    print(f"Score avec .score : {score}")

    # Classe de l’image 4 et sa classe prédite.
    print(mnist.target[4])
    print(mlp.predict(mnist.data[4].reshape((1, -1))))  # Reshape de notre jeu de donnée en 2D

    # Calcul de précision avec la package metrics.precision_score
    ypredTest = mlp.predict(xtest)
    precision = metrics.precision_score(ytest, ypredTest, average='micro')
    print(f"Score avec precision_score : {precision}")

    """
    # Varier le nombre de couches de 1 entre (2 et 100) couches
    _50neuron_layer = []
    print("Variation du nombre de couches de 2 à 100 : ")
    for nb_layer in range(2, 101):
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=tuple([50 for i in range(nb_layer)]))

        start_training = time.time()
        mlp.fit(xtrain, ytrain)
        final_training = time.time() - start_training

        start_prediction = time.time()
        ypred = mlp.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        _50neuron_layer.append((nb_layer, final_training, final_prediction, error))
        print(f"\t {_50neuron_layer[-1]}")

    _50neuron_layer_list = list(zip(*_50neuron_layer))

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(_50neuron_layer_list[0], _50neuron_layer_list[3], 'x-', color='red')
    plt.xlabel('nb layer')
    plt.ylabel("Erreur sur le jeu de données")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(_50neuron_layer_list[0], _50neuron_layer_list[1], 'x-', color='green')
    plt.xlabel('nb layer')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(_50neuron_layer_list[0], _50neuron_layer_list[2], 'x-', color='blue')
    plt.xlabel('nb layer')
    plt.ylabel("Durée de prédiction")
    plt.grid(True)

    plt.show()
    """

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=tuple(range(60, 10, -1)))
    start_training = time.time()
    mlp.fit(xtrain, ytrain)
    final_training = time.time() - start_training
    start_prediction = time.time()
    ypred = mlp.predict(xtest)
    final_prediction = time.time() - start_prediction
    erreur = metrics.zero_one_loss(ytest, ypred)
    print(f"Erreur d'un MLP de 50 couches de 60 à 11 nerones: {erreur}")
    print(f"Temps d'apprentissage d'un MLP de 50 couches de 60 à 11 nerones: {final_training}")
    print(f"Temps de prédiction MLP de 50 couches de 60 à 11 nerones: {final_prediction}")

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=tuple(list(range(60, 32, -3)) + list(range(31, 12, -2))))
    start_training = time.time()
    mlp.fit(xtrain, ytrain)
    final_training = time.time() - start_training
    start_prediction = time.time()
    ypred = mlp.predict(xtest)
    final_prediction = time.time() - start_prediction
    erreur = metrics.zero_one_loss(ytest, ypred)
    print(f"Erreur d'un MLP de 50 couches -3 puis -2 nerones: {erreur}")
    print(f"Temps d'apprentissage d'un MLP de 50 couches -3 puis -2 nerones: {final_training}")
    print(f"Temps de prédiction MLP de 50 couches -3 puis -2 nerones: {final_prediction}")

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(14, 36, 64))
    start_training = time.time()
    mlp.fit(xtrain, ytrain)
    final_training = time.time() - start_training
    start_prediction = time.time()
    ypred = mlp.predict(xtest)
    final_prediction = time.time() - start_prediction
    erreur = metrics.zero_one_loss(ytest, ypred)
    print(f"Erreur d'un MLP de 3 couches 14 36 64 nerones: {erreur}")
    print(f"Temps d'apprentissage d'un MLP de 3 couches 14 36 64 nerones: {final_training}")
    print(f"Temps de prédiction MLP de 3 couches 14 36 64 nerones: {final_prediction}")

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(14, 36, 64, 112, 176, 204, 226, 283))
    start_training = time.time()
    mlp.fit(xtrain, ytrain)
    final_training = time.time() - start_training
    start_prediction = time.time()
    ypred = mlp.predict(xtest)
    final_prediction = time.time() - start_prediction
    erreur = metrics.zero_one_loss(ytest, ypred)
    print(f"Erreur d'un MLP de 8 couches 14, 36, 64, 112, 176, 204, 226, 283 nerones: {erreur}")
    print(f"Temps d'apprentissage d'un MLP de 8 couches 14, 36, 64, 112, 176, 204, 226, 283 nerones: {final_training}")
    print(f"Temps de prédiction MLP de 8 couches 14, 36, 64, 112, 176, 204, 226, 2832 nerones: {final_prediction}")

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 92, 117, 208, 117, 92, 64))
    start_training = time.time()
    mlp.fit(xtrain, ytrain)
    final_training = time.time() - start_training
    start_prediction = time.time()
    ypred = mlp.predict(xtest)
    final_prediction = time.time() - start_prediction
    erreur = metrics.zero_one_loss(ytest, ypred)
    print(f"Erreur d'un MLP de 7 couches 64, 92, 117, 208, 117, 92, 64 nerones: {erreur}")
    print(f"Temps d'apprentissage d'un MLP de 7 couches 64, 92, 117, 208, 117, 92, 64 nerones: {final_training}")
    print(f"Temps de prédiction MLP de 7 couches 64, 92, 117, 208, 117, 92, 64 nerones: {final_prediction}")

    solving = []
    print("Variations du solver : ")
    for solver in ['lbfgs', 'sgd', 'adam']:
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 92, 117, 208, 117, 92, 64), solver=solver)
        debut_entrainement = time.time()
        mlp.fit(xtrain, ytrain)
        duree_entrainement = time.time() - debut_entrainement
        debut_prediction = time.time()
        ypred = mlp.predict(xtest)
        duree_prediction = time.time() - debut_prediction
        erreur = metrics.zero_one_loss(ytest, ypred)
        solving.append((solver, duree_entrainement, duree_prediction, erreur))
        print(f"\t {solving[-1]}")

    solving_list = list(zip(*solving))

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(solving_list[0], solving_list[3], 'x-', color='red')
    plt.xlabel('solver')
    plt.ylabel('Erreur sur les jeux de données')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(solving_list[0], solving_list[1], 'x-', color='green')
    plt.xlabel('solver')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(solving_list[0], solving_list[2], 'x-', color='blue')
    plt.xlabel('solver')
    plt.ylabel('Durée de prédiction')
    plt.grid(True)

    plt.show()

    activations = []
    print("Variations de l'activation : ")
    for activation in ['identity', 'logistic', 'tanh', 'relu']:
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 92, 117, 208, 117, 92, 64), activation=activation)
        debut_entrainement = time.time()
        mlp.fit(xtrain, ytrain)
        duree_entrainement = time.time() - debut_entrainement
        debut_prediction = time.time()
        ypred = mlp.predict(xtest)
        duree_prediction = time.time() - debut_prediction
        erreur = metrics.zero_one_loss(ytest, ypred)
        activations.append((activation, duree_entrainement, duree_prediction, erreur))
        print(f"\t {activations[-1]}")

    activations_liste = list(zip(*activations))

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(activations_liste[0], activations_liste[3], 'x-', color='red')
    plt.xlabel('Fonction d\'activation')
    plt.ylabel('Erreur sur les jeux de données')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(activations_liste[0], activations_liste[1], 'x-', color="green")
    plt.xlabel('fct activation')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(activations_liste[0], activations_liste[2], 'x-', color="blue")
    plt.xlabel('fct activation')
    plt.ylabel('Durée de prédiction')
    plt.grid(True)

    plt.show()

    regularisations = []
    print("Variations de la régularisation : ")
    for regularisation in np.arange(0.0001, 0.01, 0.001):
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 92, 117, 208, 117, 92, 64), alpha=regularisation)
        debut_entrainement = time.time()
        mlp.fit(xtrain, ytrain)
        duree_entrainement = time.time() - debut_entrainement
        debut_prediction = time.time()
        ypred = mlp.predict(xtest)
        duree_prediction = time.time() - debut_prediction
        erreur = metrics.zero_one_loss(ytest, ypred)
        regularisations.append((regularisation, duree_entrainement, duree_prediction, erreur))
        print(f"\t {regularisations[-1]}")

    regularisations_liste = list(zip(*regularisations))

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(regularisations_liste[0], regularisations_liste[3], 'x-', color='red')
    plt.xlabel('Régularisation')
    plt.ylabel('erreur')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(regularisations_liste[0], regularisations_liste[1], 'x-', color="green")
    plt.xlabel('Régularisation')
    plt.ylabel("Durée d'entraînement")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(regularisations_liste[0], regularisations_liste[2], 'x-', color='blue')
    plt.xlabel('Régularisation')
    plt.ylabel('Durée de prédiction')
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
