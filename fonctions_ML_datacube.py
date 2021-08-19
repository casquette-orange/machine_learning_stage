# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:43:12 2021

@author: tbeaudelain
"""


from prettytable import PrettyTable

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
from scipy.stats import skewnorm
import numpy as np; np.random.seed(1)
from sklearn.metrics import *
from scipy.stats import gaussian_kde
from netCDF4 import Dataset
import xarray as xr
import seaborn as sns







"""     ----------------------------------------------------  1/      FONCTIONS PLOTING          """





# ----------------      FIGURE 1     ----------------  
# GRAPHE 1
def hist_plot_features(random_forest):
    plt.subplot(311)
    names_features = [ 'Vent modèle', 'Temperature air', 'Temperature mer','Humidité relative', 'Direction du vent', 'Flux vertical de chaleur','vent_40m', 'moyenne vent image', 'diff temps', 'temps','longitude', 'latitude']#, 'longitude', 'latitude', 'wind_20m', 'wind_100m'
    plt.bar(range(len(random_forest.feature_importances_)), random_forest.feature_importances_, tick_label = names_features, color ='royalblue', alpha = 0.6,   linewidth = '1', edgecolor = 'blue')
    plt.title('Importances des paramètres pour le RandomForest')
    plt.ylabel('importances des paramètres')
    plt.legend()
    plt.show()

# GRAPHE 2
def hist_plot(y_test, vitesse_ML, vitesse):
    plt.style.use('seaborn-white')
    plt.subplot(312)
    err_hist = vitesse - y_test
    err_hist_ML = vitesse_ML - y_test
    plt.hist(err_hist, bins=100, range =[-5, 5], histtype='stepfilled', alpha=0.2, ec="k", density =True, color ='forestgreen', label='vitesse du modèle')
    plt.hist(err_hist_ML, bins=100, range =[-5, 5], histtype='stepfilled', alpha=0.2, ec="k",density =True, color ='r', label='vitesse du modèle corrigé par ML')
    a, mu, sigma = skewnorm.fit(err_hist)
    a_ML, mu_ML, sigma_ML = skewnorm.fit(err_hist_ML)
    plt.title('Répartition des erreurs')
    #plt.xlabel('Erreur en m/s') # car il gêne 
    plt.ylabel('densité de données')
    #plt.xlabel('Erreurs en m/s')
    plt.legend()
    pdf_err=skewnorm.pdf(np.arange(-5, 5, 0.1), a, mu, sigma)
    plt.plot(np.arange(-5, 5, 0.1), pdf_err, 'g')
    pdf_err_ML=skewnorm.pdf(np.arange(-5, 5, 0.1), a_ML, mu_ML, sigma_ML)
    plt.plot(np.arange(-5, 5, 0.1), pdf_err_ML, 'r')
    plt.show()

# GRAPHE 3
def courbe_plot(y_test, vitesse_ML, vitesse, taille_zone):
    plt.subplot(313)
    plt.plot(np.arange(0, int(round(len(vitesse)/((taille_zone**2)*4)))), vitesse[0:int(round(len(vitesse)/((taille_zone**2)*4)))*(taille_zone**2):(taille_zone**2)], label='vitesse du modèle', markeredgewidth=1, color ='forestgreen')
    plt.plot(np.arange(0, int(round(len(vitesse)/((taille_zone**2)*4)))), vitesse_ML[0:int(round(len(vitesse)/((taille_zone**2)*4)))*(taille_zone**2):(taille_zone**2)], label='vitesse du modèle corrigé par ML', markeredgewidth=1, color ='r')
    plt.plot(np.arange(0, int(round(len(vitesse)/((taille_zone**2)*4)))), y_test[0:int(round(len(vitesse)/((taille_zone**2)*4)))*(taille_zone**2):(taille_zone**2)], label='vitesse cible du SAR', markeredgewidth=1, color ='black')
    plt.title('Vitesse du vent (zoomé sur un quart des données)')
    plt.ylabel('vitesse du vent en m/s')
    plt.legend()
    plt.show()
    

    
# ----------------      FIGURE 2     ---------------- 
def plot_carte(y_test, vitesse_ML, vitesse, vitesse_train, y_train, nb_pixel):
    
    y_test_mean = np.zeros((nb_pixel, nb_pixel))
    y_train_mean = np.zeros((nb_pixel, nb_pixel))
    a = 0
    for i in range(1, int(len(y_test)/(nb_pixel**2))):
        a = a + nb_pixel*nb_pixel        
        y_testt = y_test[0+a:nb_pixel*nb_pixel+a]
        y_trainn = y_train[0+a:nb_pixel*nb_pixel+a]
        y_testt = y_testt.reshape((nb_pixel, nb_pixel))
        y_trainn = y_trainn.reshape((nb_pixel, nb_pixel))        
        y_test_mean = y_test_mean + y_testt
        y_train_mean = y_train_mean + y_trainn
    vitesse_mean = np.zeros((nb_pixel, nb_pixel))
    vitesse_train_mean = np.zeros((nb_pixel, nb_pixel))
    vitesse_ML_mean = np.zeros((nb_pixel, nb_pixel))
    
    
    
    a = 0
    for i in range(1, int(len(y_test)/(nb_pixel**2))):
        a = a + nb_pixel*nb_pixel       
        vitessee = vitesse[0+a:nb_pixel*nb_pixel+a]
        vitessee_train = vitesse_train[0+a:nb_pixel*nb_pixel+a]
        vitessee_ML = vitesse_ML[0+a:nb_pixel*nb_pixel+a]
        vitessee = vitessee.reshape((nb_pixel, nb_pixel))
        vitessee_train = vitessee_train.reshape((nb_pixel, nb_pixel))
        vitessee_ML = vitessee_ML.reshape((nb_pixel, nb_pixel))      
        vitesse_mean = vitesse_mean + vitessee
        vitesse_train_mean = vitesse_train_mean + vitessee_train
        vitesse_ML_mean = vitesse_ML_mean + vitessee_ML
        
    erreur = np.flipud((vitesse_mean - y_test_mean)) /i
    erreur_train = np.flipud((vitesse_train_mean - y_train_mean)) /i
    erreur_ML = np.flipud((vitesse_ML_mean - y_test_mean))/i
    
    
        
    fig, (ax1, ax2, ax3, cax) = plt.subplots(ncols=4,figsize=(5.5,3), 
              gridspec_kw={"width_ratios":[1,1,1, 0.05]})
    fig.subplots_adjust(wspace=0.3)
    im1 = ax1.imshow(erreur_train, cmap = 'seismic', vmin = -1, vmax  = 1)
    im2 = ax2.imshow(erreur, cmap = 'seismic', vmin=-1, vmax=1 )
    im3  = ax3.imshow(erreur_ML, cmap ='seismic', vmin = -1, vmax  = 1)
    fig.colorbar(im1, cax=cax)
    ax1.title.set_text('Biais train')
    ax2.title.set_text('Biais test')
    ax3.title.set_text('Biais ML')
    fig.suptitle('Biais sur la zone étudiée', fontsize=16)
    plt.show()



# ----------------      FIGURE 3     ---------------- 
def plot_x_y(y_test, vitesse_ML, vitesse):
    plt.figure(3)
    plt.rcParams['figure.figsize'] = (8, 8)
    # xy = np.vstack([y_test, vitesse_ML])
    # z = gaussian_kde(xy)(xy)
    # xy2 = np.vstack([y_test, vitesse])
    # z2 = gaussian_kde(x2y)(xy2)
    plt.scatter(y_test, vitesse_ML, label='vitesse du modèle corrigé par ML', s=6, alpha = 0.3, c = 'b')
    plt.scatter(y_test, vitesse, label='vitesse du modèle', s= 6, alpha = 0.3, c = 'r')
    plt.plot([0.0, 20.0], [0.0, 20.0], '#4b0082', lw=2) # Red straight line
    plt.xlabel('Vitesse cible du vent SAR (m/s)', fontsize=20)
    plt.ylabel('Vitesses (modèle / ML) (m/s)', fontsize=20)
    plt.legend()
    plt.show()




"""    -------------------------------------  2/     FONCTIONS POUR LE TRAITEMENT DES DONNEES ET POUR L'APPRENTISSAGE DU MODELE        """




def correction_vent_SAR(DS_new):
    input_ML_corrected = np.load('Correction_SAR_code/input_corrected_ML.npy')
    wspd_corrected=input_ML_corrected[:,:,:,0]
    var_xr = xr.DataArray(wspd_corrected, dims=['time','x','y'])
    DS_new['owiWindSpeed'] = var_xr
    return DS_new
    
    

# PCA
def pca(x, features):
    features = [ 'wind_speed', 'temperature', 'water_temperature_surface','relative_humidity', 'wind_dir', 'sensible_heat_net_flux_surface', 'wind_40m']
    # PCA
    # z = x[:,0]
    # z = z.reshape(-1, 1)
    # x = np.delete(x,0, axis = 1)
    # STANDARDISATION
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    pca = PCA(n_components=7)
    x = pca.fit_transform(x)
    # x = np.concatenate((z, x), axis = 1)
    
    # evr = pca.explained_variance_ratio_
    # cvr = np.cumsum(pca.explained_variance_ratio_)
    # pca_df = pd.DataFrame()
    # pca_df['Cumulative Variance Ratio'] = cvr
    # pca_df['Explained Variance Ratio'] = evr
    # pca_dims = []
    # for x in range(1, len(pca_df) + 1):
    #     pca_dims.append('PCA Component {}'.format(x))
    # pca=  pd.DataFrame(pca.components_, columns=features, index=pca_dims)
    # print(pca.head().T)
    
    return x


# ENTRAINEMENT DU MODELE EN RANDOM FOREST
def train_ML_correction(x, y):
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=False, random_state=0)
    vitesse_train = X_train[:,0]
    vitesse = X_test[:,0]
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)
    demande_modele = input('Voulez-vous utiliser un modèle de RandomForest existant? [y]/[n] : ')    
    while demande_modele != 'y' and demande_modele !='n':
        demande_modele = input('Voulez-vous utiliser un modèle de RandomForest existant? [y]/[n] : ')
    if demande_modele == 'y':
        nom_modele = input("veuillez entrer le nom du modele avec l'extension : ")
        with open(nom_modele, 'rb') as f:
            random_forest = pickle.load(f)          
    else:
        demande_nom_modele = input("Veuillez entrer un nom pour le modele : (par défaut : 'ML_manche.pkl' pour écraser le dernier modèle') : ")
        random_forest = RandomForestRegressor(n_estimators = 50) #Affecte à RandomForest le modèle( en précisant les hypers paramètres)
        print('Entrainement pour la correction ...')
        random_forest.fit(X_train, y_train) #Effectue le randomForest
        with open(demande_nom_modele, 'wb') as file: # .. convertie ..
            pickle.dump(random_forest, file) # .. le randomForest en fichier binaire qu'on enregistre         
    vitesse_ML = random_forest.predict(X_test)  #on affecte a 'vitesse_ML' la prédiction que va faire le randomForest de Y en fonction de ce qu'il a appris
    #np.savez('fichier.npz', variable1, variable2)
    return vitesse, vitesse_ML, y_test, random_forest, vitesse_train, y_train





# AFFICHER LES RESULTATS SCORING DANS UNE TABLE
def stats_results_ML(vitesse, vitesse_ML, y_test,latitude_debut, latitude_fin): 
        # ERREUR ABSOLUE - MODELE / ML
    erreurs_absolue_modele = mean_absolute_error(vitesse , y_test)
    erreurs_absolue_ML = mean_absolute_error(vitesse_ML , y_test)
        # ERREUR MOYENNE ABSOLUE - MODELE / ML
    mean_erreurs_modele= round(np.mean(erreurs_absolue_modele), 3) 
    mean_erreurs_ML= round(np.mean(erreurs_absolue_ML), 3)
        # ERREUR - MODELE / ML
    erreur_modele = vitesse - y_test 
    erreur_ML = vitesse_ML - y_test
        # ECART-TYPE - MODELE / ML
    std_modele = round(np.std(erreur_modele), 3) 
    std_ML = round(np.std(erreur_ML), 3)
        # ERREUR QUADRATIQUE - MODELE / ML
    erreursQ_modele = (vitesse - y_test)**2
    erreursQ_ML = (vitesse_ML - y_test)**2
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
    correlation_modele = round(np.corrcoef(y_test, vitesse)[0,1],2)
    correlation_ML = round(np.corrcoef(y_test, vitesse_ML)[0,1],2)
        # AMELIORATION DU MODELE METEO
    amelioration  = round(100 * ((mean_erreurs_modele - mean_erreurs_ML) / mean_erreurs_modele), 2)
        #Table pour affichage
    stats_table = [['vitesse_modèle_météo',mean_erreurs_modele, biais_modele, std_modele, accuracy_modele],
                   ['vitesse_ML(avec la vitesse_modèle_météo)',mean_erreurs_ML, biais_ML, std_ML, accuracy_ML]]    
    table = PrettyTable()
    table.field_names = ['vitesse','MAE', 'Biais', 'Ecart_type', 'SCORE (100 - MAPE)']
    for item in stats_table:
        table.add_row(item)
    print('\n', "Tableau bilan de l'apprentissage sur la zone : ", '\n', '\n',
          "Latitude : ", latitude_debut ,"x", latitude_fin, '      ///      ',"Longitude : ", latitude_debut ,"x", latitude_fin, '      ///      ',
          "Zone de", (latitude_fin -latitude_debut + 1)**2 ,"km² " )
    print(table)
    print("Les erreurs de la vitesse du modèle météo sont corrigés à hauteur de",amelioration, '% par le vitesse prédit par le machine learning' )
    
    
    
    
    
    
    """ FONCTIONS POUR ENREGISTRER PUIS CHARGER LE MODELE DE RANDOMFOREST """
# model_name = 'ML_zone_manche_900km2_10metre.pkl' # On ..
# with open(model_name, 'wb') as file: # .. convertie ..
#     pickle.dump(random_forest, file) # .. le randomForest en fichier binaire qu'on enregistre

# with open('ML_zone_manche_900km2_10metre.pkl', 'rb') as f:
#     RandomF = pickle.load(f)