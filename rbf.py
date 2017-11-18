#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:37:04 2016

@author: pagutierrez
"""

import numpy as np
import pandas as pd
import click


@click.command()
@click.option('--train_file', '-t', default=None, required=True,
              help=u'Fichero con los datos de entrenamiento.')
@click.option('--test_file', '-T', default=None, required=False,
              help=u'Fichero con los datos de test. Si no se especifica, se usa el fichero de train.')
@click.option('--classification', '-c', default=False, required=False,
              help=u'Utilizar si se trata de un problema de clasificación. Por defecto, se asume regresión.')
@click.option('--ratio_rbf', '-r', default=0.1, required=False,
              help=u'Porcentaje de RBFs con respecto a los patrones de entrenamiento. Por defecto, 0.1')
@click.option('--l2', '-l', default=False, required=False,
              help=u'Utilizar regularización l2. Por defecto se utiliza l1.')
@click.option('--eta', '-e', default=0.01, required=False,
              help=u'Tasa de aprendizaje. Por defecto, 0.01')
def entrenar_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta):
    """ Modelo de aprendizaje supervisado mediante red neuronal de tipo RBF.
        Ejecución de 5 semillas.
    """
    train_mses = np.empty(5)
    train_ccrs = np.empty(5)
    test_mses = np.empty(5)
    test_ccrs = np.empty(5)

    for s in range(10,60,10):   
        print("-----------")
        print("Semilla: %d" % s)
        print("-----------")     
        np.random.seed(s)
        train_mses[s//10-1], test_mses[s//10-1], train_ccrs[s//10-1], test_ccrs[s//10-1] = \
            entrenar_rbf(train_file, test_file, classification, ratio_rbf, l2, eta)
        print("MSE de entrenamiento: %f" % train_mses[s//10-1])
        print("MSE de test: %f" % test_mses[s//10-1])
        if classification:
            print("CCR de entrenamiento: %.2f%%" % train_ccrs[s//10-1])
            print("CCR de test: %.2f%%" % test_ccrs[s//10-1])
    
    print("*********************")
    print("Resumen de resultados")
    print("*********************")
    print("MSE de entrenamiento: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
    print("MSE de test: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
    if classification:
        print("CCR de entrenamiento: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("CCR de test: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))


def entrenar_rbf(train_file, test_file, classification, ratio_rbf, l2, eta):
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
                                                                           test_file)

    #TODO: Obtener num_rbf a partir de ratio_rbf
    print("Número de RBFs utilizadas: %d" %(num_rbf))
    kmedias, distancias, centros = clustering(classification, train_inputs, 
                                              train_outputs, num_rbf)
    
    radios = calcular_radios(centros, num_rbf)
    
    matriz_r = calcular_matriz_r(distancias, radios)

    if not classification:
        coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
    else:
        logreg = logreg_clasificacion(matriz_r, train_outputs, eta, l2)

    """
    TODO: Calcular las distancias de los centroides a los patrones de test
          y la matriz R de test
    """

    if not clasificacion:
        """
        TODO: Obtener las predicciones de entrenamiento y de test y calcular
              el MSE
        """
    else:
        """
        TODO: Obtener las predicciones de entrenamiento y de test y calcular
              el CCR. Calcular también el MSE, comparando las probabilidades 
              obtenidas y las probabilidades objetivo
        """

    return train_mse, test_mse, train_ccr, test_ccr

    
def lectura_datos(fichero_train, fichero_test):
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
    train_inputs  = train_dataset.values[:,0:-1]
    train_outputs = train_dataset.values[:,-1]

    test_inputs  = test_dataset.values[:,0:-1]
    test_outputs = test_dataset.values[:,-1]

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
    
    #TODO: Completar el código de la función
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

    #TODO: Completar el código de la función
    return kmedias, distancias, centros

def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """

    #TODO: Completar el código de la función
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

    #TODO: Completar el código de la función
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

    #TODO: Completar el código de la función
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

    #TODO: Completar el código de la función
    return logreg


if __name__ == "__main__":
    entrenar_rbf_total()
