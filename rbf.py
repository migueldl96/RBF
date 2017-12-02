#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:37:04 2016

@author: pagutierrez
"""

import numpy as np
import pandas as pd
import click
import random
import pdb
import time

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

@click.command()
@click.option('--train_file', '-t', default=None, required=True,
              help=u'Fichero con los datos de entrenamiento.')
@click.option('--test_file', '-T', default=None, required=False,
              help=u'Fichero con los datos de test. Si no se especifica, se usa el fichero de train.')
@click.option('--classification', '-c', is_flag=True,
              help=u'Utilizar si se trata de un problema de clasificación. Por defecto, se asume regresión.')
@click.option('--ratio_rbf', '-r', default=0.1, required=False,
              help=u'Porcentaje de RBFs con respecto a los patrones de entrenamiento. Por defecto, 0.1')
@click.option('--l2', '-l', is_flag=True,
              help=u'Utilizar regularización l2. Por defecto se utiliza l1.')
@click.option('--eta', '-e', default=0.01, required=False,
              help=u'Tasa de aprendizaje. Por defecto, 0.01')
@click.option('--outs', '-o', default=1, required=True,
              help=u'Numero de salidas del problema. Por defecto, 1.')
def entrenar_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outs):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Ejecución de 5 semillas.
    """

    train_mses = np.empty(5)
    train_ccrs = np.empty(5)
    test_mses = np.empty(5)
    test_ccrs = np.empty(5)
    times = np.empty(5)
    coefs = np.empty(5)

    for s in range(10,60,10):   
        print("-----------")
        print("Semilla: %d" % s)
        print("-----------")     
        np.random.seed(s)

        start = time.time()
        train_mses[s//10-1], test_mses[s//10-1], train_ccrs[s//10-1], test_ccrs[s//10-1], cm, coefs[s//10-1] = \
            entrenar_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outs)
        end   = time.time()
        times[s//10-1] = end-start
        print("Tiempo de ejecución: %f" % times[s//10-1])
        print("MSE de entrenamiento: %f" % train_mses[s//10-1])
        print("MSE de test: %f" % test_mses[s//10-1])
        if classification:
            print("CCR de entrenamiento: %.2f%%" % train_ccrs[s//10-1])
            print("CCR de test: %.2f%%" % test_ccrs[s//10-1])
            print("Matriz de confusión:")
            print(cm)
            print("Numero de coeficientes de la regresión logística: %.2f" % coefs[s//10-1])
    
    print("*********************")
    print("Resumen de resultados")
    print("*********************")
    print("Tiempo medio: %f" % np.mean(times))
    print("MSE de entrenamiento: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
    print("MSE de test: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
    #if classification:
    print("CCR de entrenamiento: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
    print("CCR de test: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))
    print("Numero de coeficientes medio: %.2f" % np.mean(coefs))


def entrenar_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outs):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Una única ejecución.
        Recibe los siguientes parámetros:
            - train_file: nombre del fichero de entrenamiento.
            - test_file: nombre del fichero de test.
            - classification: True si el problema es de clasificacion.
            - ratio_rbf: Ratio (en tanto por uno) de neuronas RBF con 
              respecto al total de patrones.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1 (para regresión logística).
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
        Devuelve:
            - train_mse: Error de tipo Mean Squared Error en entrenamiento. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - test_mse: Error de tipo Mean Squared Error en test. 
              En el caso de clasificación, calcularemos el MSE de las 
              probabilidades predichas frente a las objetivo.
            - train_ccr: Error de clasificación en entrenamiento. 
              En el caso de regresión, devolvemos un cero.
            - test_ccr: Error de clasificación en test. 
              En el caso de regresión, devolvemos un cero.
    """
    train_inputs, train_outputs, test_inputs, test_outputs = lectura_datos(train_file, 
                                                                           test_file, outs)

    #Numero de patrones
    num_patrones_train = train_inputs.shape[0]
    num_patrones_test  = test_inputs.shape[0]

    #Número de RBFs
    num_rbf = int(ratio_rbf * num_patrones_train)
    print("Número de RBFs utilizadas: %d" %(num_rbf))

    kmedias, distancias, centros = clustering(classification, train_inputs,
                                              train_outputs, num_rbf)
    
    radios = calcular_radios(centros, num_rbf)
    
    matriz_r = calcular_matriz_r(distancias, radios)

    if not classification:
        coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
    else:
        logreg = logreg_clasificacion(matriz_r, train_outputs, eta, l2)

    # Distancia de patrones de test a centroides y matriz R
    distancias_centroides_test = kmedias.transform(test_inputs)
    matriz_r_test = calcular_matriz_r(distancias_centroides_test, radios)

    if not classification:
        # Predicciones del modelo
        train_predictions = np.dot(matriz_r, coeficientes)
        test_predictions  = np.dot(matriz_r_test, coeficientes)

        # MSE de train y test
        train_mse = mean_squared_error(train_outputs, train_predictions)
        test_mse  = mean_squared_error(test_outputs, test_predictions)

        # Clasificación desde regresión
        predicted_train_classes = np.round(train_predictions)
        predicted_test_classes = np.round(test_predictions)
        train_ccr = (np.sum(predicted_train_classes == train_outputs) / np.float(num_patrones_train)) * 100
        test_ccr = (np.sum(predicted_test_classes == test_outputs) / np.float(num_patrones_test)) * 100

        cm = 0
        coefs = 0
        #pdb.set_trace()

    else:
        # CCR en train y test
        train_ccr = logreg.score(matriz_r, train_outputs) * 100
        test_ccr  = logreg.score(matriz_r_test, test_outputs) * 100

        # Patrones mal clasificados
        # predicted_test_classes = logreg.predict(matriz_r_test)
        # classification_mask = predicted_test_classes == test_outputs
        # missclassified_indexes = np.where(classification_mask == False)
        # print(zip(missclassified_indexes[0], predicted_test_classes[missclassified_indexes]))

        # MSE en train y test
        clases = logreg.classes_
        train_probs = logreg.predict_proba(matriz_r)
        train_outputs_binarized = label_binarize(train_outputs, clases)

        test_probs  = logreg.predict_proba(matriz_r_test)
        test_outputs_binarized  = label_binarize(test_outputs, clases)

        train_mse = mean_squared_error(train_outputs_binarized, train_probs)
        test_mse  = mean_squared_error(test_outputs_binarized, test_probs)

        # Matriz de confusión
        real_classes = test_outputs
        predicted_classes = logreg.predict(matriz_r_test)
        classes = logreg.classes_
        cm = confusion_matrix(real_classes, predicted_classes, labels=classes)
        coefs = len(logreg.coef_[np.where(np.absolute(logreg.coef_) > 1e-5)])

    return train_mse, test_mse, train_ccr, test_ccr, cm, coefs

    
def lectura_datos(fichero_train, fichero_test, outs):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
        Devuelve:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - test_inputs: matriz con las variables de entrada de 
              test.
            - test_outputs: matriz con las variables de salida de 
              test.
    """

    # Leemos los CSV
    train_dataset = pd.read_csv(fichero_train)

    # Si no se especifica fichero de test, se utiliza el de train
    if not fichero_test:
      test_dataset  = train_dataset
    else:
      test_dataset = pd.read_csv(fichero_test)

    # Separamos las entradas y salidas de entrenamiento y test
    train_inputs  = train_dataset.values[:,0:-outs]
    train_outputs = train_dataset.values[:,-outs]

    test_inputs  = test_dataset.values[:,0:-outs]
    test_outputs = test_dataset.values[:,-outs]

    return train_inputs, train_outputs, test_inputs, test_outputs

def inicializar_centroides_clas(train_inputs, train_outputs, num_rbf):
    """ Inicializa los centroides para el caso de clasificación.
        Debe elegir, aprox., num_rbf/num_clases
        patrones por cada clase. Recibe los siguientes parámetros:
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - centroides: matriz con todos los centroides iniciales
                          (num_rbf x num_entradas).
    """
    
    # Particiones estratificadas de num_rbf/clases elementos
    stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=num_rbf, test_size=None)
    splits = stratified_split.split(train_inputs, train_outputs)
    indices = list(splits)[0][0]
    centroides = train_inputs[indices]

    return centroides

def clustering(clasificacion, train_inputs, train_outputs, num_rbf):
    """ Realiza el proceso de clustering. En el caso de la clasificación, se
        deben escoger los centroides usando inicializar_centroides_clas()
        En el caso de la regresión, se escogen aleatoriamente.
        Recibe los siguientes parámetros:
            - clasificacion: True si el problema es de clasificacion.
            - train_inputs: matriz con las variables de entrada de 
              entrenamiento.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - kmedias: objeto de tipo sklearn.cluster.KMeans ya entrenado.
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - centros: matriz (num_rbf x num_entradas) con los centroides 
              obtenidos tras el proceso de clustering.
    """

    # Centroides de las RBFs
    if clasificacion:
      centros_iniciales = inicializar_centroides_clas(train_inputs, train_outputs, num_rbf)
      kmedias = KMeans(n_clusters=num_rbf, init=centros_iniciales, n_init=1, max_iter=500, n_jobs=-1)
    else:
      kmedias = KMeans(n_clusters=num_rbf, init='random', n_init=1, max_iter=500, n_jobs=-1)

    # Matriz de distancias
    distancias = kmedias.fit_transform(train_inputs)

    # Centros
    centros = kmedias.cluster_centers_

    return kmedias, distancias, centros

def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """

    # Matriz de distancias
    distancias = pairwise_distances(centros, Y=None, metric="euclidean", n_jobs=-1)

    # Radios = suma de filas / 2 * num_rbf-1
    radios = distancias.sum(axis=1)/(2 * (num_rbf-1))

    return radios

def calcular_matriz_r(distancias, radios):
    """ Devuelve el valor de activación de cada neurona para cada patrón 
        (matriz R en la presentación)
        Recibe los siguientes parámetros:
            - distancias: matriz (num_patrones x num_rbf) con la distancia 
              desde cada patrón hasta cada rbf.
            - radios: array (num_rbf) con el radio de cada RBF.
        Devuelve:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
    """

    # Salida = exp(-distancia^2/2*radio^2) - [0,1]
    sesgo = np.ones(distancias.shape[0])
    matriz_r = np.exp(-np.square(distancias)/(np.square(radios)*2))
    matriz_r = np.column_stack((matriz_r, sesgo))

    return matriz_r

def invertir_matriz_regresion(matriz_r, train_outputs):
    """ Devuelve el vector de coeficientes obtenidos para el caso de la 
        regresión (matriz beta en las diapositivas)
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
        Devuelve:
            - coeficientes: vector (num_rbf+1) con el valor del sesgo y del 
              coeficiente de salida para cada rbf.
    """

    # Pseudo-inversa
    pseudoinv_matriz_r = np.linalg.pinv(matriz_r)
    coeficientes = np.dot(pseudoinv_matriz_r, train_outputs)

    return coeficientes

def logreg_clasificacion(matriz_r, train_outputs, eta, l2):
    """ Devuelve el objeto de tipo regresión logística obtenido a partir de la
        matriz R.
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de 
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los 
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de 
              entrenamiento.
            - eta: valor del parámetro de regularización para la Regresión 
              Logística.
            - l2: True si queremos utilizar L2 para la Regresión Logística. 
              False si queremos usar L1.
        Devuelve:
            - logreg: objeto de tipo sklearn.linear_model.LogisticRegression ya
              entrenado.
    """

    # Regularización
    if l2:
      regularizacion = 'l2'
    else:
      regularizacion = 'l1'

    # Parámetro C
    c = 1/eta

    lr = LogisticRegression(penalty=regularizacion, C=c, fit_intercept=False)
    logreg = lr.fit(matriz_r, train_outputs)
    return logreg


if __name__ == "__main__":
    entrenar_rbf_total()
