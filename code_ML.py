# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:15:48 2021

@author: tbeaudelain
"""

"""Module importation"""

from prettytable import PrettyTable
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, ridge_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


dataa = pd.read_csv('dataset_classification_simple_ok.csv', delimiter = ';') #charge les données du dataset sous un DataFrame pandas
#dataa = dataa.drop([306, 410, 64, 522, 850, 492, 590, 797], axis = 0) #retire les outliers du dataset
dataa = dataa.to_numpy() #converti le DataFrame pandas en tableau Numpy
scaler = StandardScaler()

# dataa[:,5] = dataa[:,0]*dataa[:,5]
# dataa[:,6] = dataa[:,0]*dataa[:,6]





#def train_ML_without_PCA_correction(data): 
print('Entrainement pour la correction ...') #affiche un message d'attente
x = np.delete(dataa,6, axis = 1)

#                     STANDARDISATION
# scaler = StandardScaler()
# x = scaler.fit_transform(x)

#                     PCA SANS LE RATIO MODELE METEO
# x = np.delete(x,5, axis = 1)
# pca = PCA(n_components=5)
# x = pca.fit_transform(x)
# z = dataa[:,5]
# z = z.reshape(-1, 1)
# x = np.concatenate((x, z), axis = 1)

y = dataa[:,6]
# erreurs_absolue = abs(dataa[:,5] - y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False, random_state=0)
ratio = X_test[:,5]
random_forest = RandomForestRegressor(max_features = 5) #Affecte à RandomForest le modèle( en précisant les hypers paramètres)
random_forest.fit(X_train, y_train) #Effectue le randomForest
model_name = 'ML_without_PCA.pkl' # On ..
with open(model_name, 'wb') as file: # .. convertie ..
    pickle.dump(random_forest, file) # .. le randomForest en fichier binaire qu'on enregistre
ratio = X_test[:,5]
ratio_ML = random_forest.predict(X_test)  #on affecte a 'ratio_ML' la prédiction que va faire le randomForest de Y en fonction de ce qu'il a appris
stats_results_ML(ratio,ratio_ML,y_test) #on appel la fonction 'stats_results_ML'






def stats_results_ML(ratio, ratio_ML, y_test): 
    # ERREUR ABSOLUE - MODELE / ML
    erreurs_absolue_modele = abs(ratio - y_test) 
    erreurs_absolue_ML = abs(ratio_ML - y_test)
    # ERREUR MOYENNE ABSOLUE - MODELE / ML
    mean_erreurs_modele= round(np.mean(erreurs_absolue_modele), 3) 
    mean_erreurs_ML= round(np.mean(erreurs_absolue_ML), 3)
    # ERREUR - MODELE / ML
    erreur_modele = ratio - y_test 
    erreur_ML = ratio_ML - y_test
    # ECART-TYPE - MODELE / ML
    std_modele = round(np.std(erreur_modele), 3) 
    std_ML = round(np.std(erreur_ML), 3)
    # ERREUR QUADRATIQUE - MODELE / ML
    erreursQ_modele = (ratio - y_test)**2
    erreursQ_ML = (ratio_ML - y_test)**2
    # ERREUR QUADRATIQUE MOYENNE - MODELE / ML
    mean_erreursQ_modele= round(np.mean(erreursQ_modele), 2) 
    mean_erreursQ_ML= round(np.mean(erreursQ_ML), 2)
    # BIAIS (moyenne des erreurs) - MODELE / ML
    biais_modele= round(np.mean(erreur_modele), 3) 
    biais_ML= round(np.mean(erreur_ML), 3)
    # %AGE ERREUR MOYENNE ABSOLUE - MODELE / ML
    mape_modele = 100 * (erreurs_absolue_modele / y_test)
    mape_ML = 100 * (erreurs_absolue_ML / y_test)
    # SCORE   -->    100 - MOYENNE(%AGE ERREUR MOYENNE ABSOLUE - MODELE / ML)
    accuracy_modele = round(100 - np.mean(mape_modele),2) 
    accuracy_ML = round(100 - np.mean(mape_ML),2)
    # COEFFICIENT DE CORRELATION - MODELE / ML
    correlation_modele = round(np.corrcoef(y_test, ratio)[0,1],2)
    correlation_ML = round(np.corrcoef(y_test, ratio_ML)[0,1],2)
    # AMELIORATION DU MODELE METEO
    amelioration  = round(100 * ((mean_erreurs_modele - mean_erreurs_ML) / mean_erreurs_modele), 2)
    #Table pour affichage
    stats_table = [['ratio_modèle_météo',mean_erreurs_modele, biais_modele, std_modele, accuracy_modele],
                   ['ratio_ML(avec le ratio_modèle_météo)',mean_erreurs_ML, biais_ML, std_ML, accuracy_ML]]    
    table = PrettyTable()
    table.field_names = ['Ratio','MAE', 'Biais', 'Ecart_type', 'SCORE (100 - MAPE)']
    for item in stats_table:
        table.add_row(item)
    print(table)
    print("Les erreurs du ratio du modèle météo sont corrigés à hauteur de",amelioration, "% par le ratio prédit par l'apprentissage automatique" )
    
