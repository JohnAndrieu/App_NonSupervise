from sklearn import neural_network, model_selection
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn import metrics
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


def create_neural_network(layers, xtrain, ytrain, xtest, ytest, label):
    mlp = neural_network.MLPClassifier(hidden_layer_sizes=layers)
    start_training = time.time()
    mlp.fit(xtrain, ytrain)
    final_training = time.time() - start_training
    start_prediction = time.time()
    ypred = mlp.predict(xtest)
    final_prediction = time.time() - start_prediction
    erreur = metrics.zero_one_loss(ytest, ypred)
    print(f"Erreur d'un MLP de {label}: {erreur}")
    print(f"Temps d'apprentissage d'un MLP de {label}: {final_training}")
    print(f"Temps de prédiction MLP de {label}: {final_prediction}")


def main():
    mnist = fetch_openml(name='mnist_784')
    echantillon = np.random.randint(70000, size=5000)
    data = mnist.data[echantillon]
    target = mnist.target[echantillon]

    xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(50))
    mlp.fit(xtrain, ytrain)
    score = mlp.score(xtest, ytest)
    print(f"Score avec mlp.score : {score}")

    # Classe de l’image 4 et sa classe prédite.
    print(mnist.target[4])
    print(mlp.predict(mnist.data[4].reshape((1, -1))))  # Reshape de notre jeu de donnée en 2D

    # Calcul de précision avec la package metrics.precision_score
    ypredTest = mlp.predict(xtest)
    precision = metrics.precision_score(ytest, ypredTest, average='micro')
    print(f"Score avec la fonction precision_score : {precision}")

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

    plot_fig(_50neuron_layer_list)

    create_neural_network(tuple(range(60, 10, -1)), xtrain, ytrain, xtest, ytest, "50 couches de 60 à 11 nerones")
    create_neural_network(tuple(list(range(60, 32, -3)) + list(range(31, 12, -2))), xtrain, ytrain, xtest, ytest,
                          "50 couches -3 puis -2 nerones")
    create_neural_network((14, 36, 64), xtrain, ytrain, xtest, ytest, "3 couches 14 36 64 nerones")
    create_neural_network((14, 36, 64, 112, 176, 204, 226, 283), xtrain, ytrain, xtest, ytest,
                          "8 couches 14, 36, 64, 112, 176, 204, 226, 283 nerones")
    create_neural_network((64, 92, 117, 208, 117, 92, 64), xtrain, ytrain, xtest, ytest,
                          "7 couches 64, 92, 117, 208, 117, 92, 64 nerones")

    solving = []
    print("Modification du solver : ")
    for solver in ['lbfgs', 'sgd', 'adam']:
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 92, 117, 208, 117, 92, 64), solver=solver)

        start_training = time.time()
        mlp.fit(xtrain, ytrain)
        final_training = time.time() - start_training

        start_prediction = time.time()
        ypred = mlp.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        solving.append((solver, final_training, final_prediction, error))
        print(f"\t {solving[-1]}")

    solving_list = list(zip(*solving))

    plot_fig(solving_list)

    activ = []
    print("Variations de l'activation : ")
    for activation in ['identity', 'logistic', 'tanh', 'relu']:
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 92, 117, 208, 117, 92, 64), activation=activation)

        start_training = time.time()
        mlp.fit(xtrain, ytrain)
        final_training = time.time() - start_training

        start_prediction = time.time()
        ypred = mlp.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        activ.append((activation, final_training, final_prediction, error))
        print(f"\t {activ[-1]}")

    activ_liste = list(zip(*activ))

    plot_fig(activ_liste)

    regul = []
    print("Evolution de la régularisation : ")
    for regularisation in np.arange(0.0001, 0.01, 0.001):
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(64, 92, 117, 208, 117, 92, 64), alpha=regularisation)

        start_training = time.time()
        mlp.fit(xtrain, ytrain)
        final_training = time.time() - start_training

        start_prediction = time.time()
        ypred = mlp.predict(xtest)
        final_prediction = time.time() - start_prediction

        error = metrics.zero_one_loss(ytest, ypred)
        regul.append((regularisation, final_training, final_prediction, error))
        print(f"\t {regul[-1]}")

    regul_liste = list(zip(*regul))

    plot_fig(regul_liste)

    best_layer = (64, 92, 117, 208, 117, 92, 64)
    best_solver = "adam"
    best_activation = "relu"
    best_regularisation = 0.008

    best_mlp = neural_network.MLPClassifier(hidden_layer_sizes=best_layer, solver=best_solver,
                                            activation=best_activation, alpha=best_regularisation)
    start_training = time.time()
    best_mlp.fit(xtrain, ytrain)
    best_final_entrainement = time.time() - start_training

    start_prediction = time.time()
    ypred = best_mlp.predict(xtest)
    best_final_prediction = time.time() - start_prediction

    cross_val = model_selection.cross_val_score(best_mlp, data, target, cv=10)
    best_error = 1 - np.mean(cross_val)

    print(f"Durée de l'entraînement : {best_final_entrainement}")
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
