# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:06:58 2019

@author: Carlos Rana
"""

import numpy as np
from sklearn.manifold import MDS
import time
from math import floor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
import random

np.random.seed(22)
#random.seed(22)

datos = pd.read_csv('mujeres_fecundidad_INE_2018.csv')
for col in datos:
   datos[col].fillna(datos[col].mean(), inplace=True)
  

i = 1
# =============================================================================
#                   PRIMER CASO
# =============================================================================
#seleccionar casos

#primer caso
subset1 = datos.loc[(datos['CONVI15ABU'] == 1) & (datos['NHERM'] > 0)]
#segundo caso
subset2 = datos.loc[(datos['TDPAREJA'] <= 50) & (datos['INGRESOSPAR'] < 4)&(datos['EMPPAREJA'] < 3)]
#tercer caso
subset3 = datos.loc[(datos['SATISFACENINOS'] >= 7) & (datos['NRESI'] > 3) ]


#seleccionar variables de interés para clustering
usadas1 = ['DIFICULTAD', 'NHIJOSPAR', 'SITSENTI', 'TIPOHIJO', 'NDESEOHIJO']
#segundas variables
usadas2 = ['SEXOPAR','TRABAJAACT','V_HIJOSM','V_PRIORIDADM','V_TARDOM']
#terceras variables
usadas3 = ['DIFICULTAD','PAGOVI','SITLABPAR','ESTUDIOSPAR','SATISTARDOM']

casos = []
casos.append(np.array([subset1,usadas1]))
casos.append(np.array([subset2,usadas2]))
casos.append(np.array([subset3,usadas3]))

time_inicio = time.time()
for caso in casos:
    print("Caso de uso numero",i)
    subset = caso[0]
    usadas = caso[1]
    X = subset[usadas]
    X_normal = preprocessing.normalize(X, norm='l2')
    
    if len(X) > 10000:
        muestra_silhoutte = 0.2
    else:
        muestra_silhoutte = 1
    
    
    #print(X.hist(),len(X),X.mean())
    
    
    #Numero de clusters
    #n = 50
    n = 3
    tamanio = [2,3,5,10,50]
    
    # =============================================================================
    # EJECUTAMOS ALGORITMOS DE CLUSTERING.
    # =============================================================================
    
    bandwidth = estimate_bandwidth(X_normal, quantile=0.2, n_samples=500)
    mean_shift =  MeanShift(bandwidth=bandwidth, bin_seeding=True)
    db = DBSCAN(algorithm='auto', eps=0.3, leaf_size=20, metric='euclidean',metric_params=None, min_samples=200, n_jobs=None, p=None)
    
    k_means = KMeans(init='k-means++', n_clusters=n, n_init=5)
    agglomerative = AgglomerativeClustering(n_clusters=n)
    birch = Birch(branching_factor=50, n_clusters=n, threshold=0.1,compute_labels=True)
        
    clustering = (("K Means",k_means),("Agglomerative",agglomerative),("Birch",birch),("Mean Shift",mean_shift),("DBSCAN",db))
    clustering_def = (("K Means",k_means),("Agglomerative",agglomerative),("Birch",birch))
    clustering_no_def = (("Mean Shift",mean_shift),("DBSCAN",db))
    predicciones = {}
    resultados = {}
    
    cluster_predict = k_means.fit_predict(X_normal)
    # =============================================================================
    # PREDECIMOS
    # =============================================================================
    bandwidth = estimate_bandwidth(X_normal, quantile=0.2, n_samples=500)
    mean_shift =  MeanShift(bandwidth=bandwidth, bin_seeding=True)
    db = DBSCAN(algorithm='auto', eps=0.3, leaf_size=20, metric='euclidean',metric_params=None, min_samples=200, n_jobs=None, p=None)
    
    #Ejecutamos los algoritmos en los que no se predefine el numero de clusteres
    for name,alg in clustering_no_def:
        t = time.time()
        cluster_predict = alg.fit_predict(X_normal) 
        tamanio.append(len(cluster_predict))
        tiempo = time.time() - t
        print(name," ha tardado {:.2f} segundos.".format(tiempo))
        predicciones[name] = cluster_predict
        metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
        metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=22)
        resultados[name] = np.array([round((metric_CH),4),round((metric_SC),4)])    
    
    #Ahora ejecutamos los algoritmos restantes en un doble bucle para quedarnos
    #con la ejecucion que mejor metricas devuelve; cambiando el parametro del 
    #numero de clusteres.
    
    n_mejor = 0
    
    for n in np.unique(tamanio):
        k_means = KMeans(init='k-means++', n_clusters=n, n_init=5)
        agglomerative = AgglomerativeClustering(n_clusters=n)
        birch = Birch(branching_factor=50, n_clusters=n, threshold=0.1,compute_labels=True)
        
        for name,alg in clustering_def:    
            t = time.time()
            cluster_predict = alg.fit_predict(X_normal) 
            tiempo = time.time() - t
            print(name," ha tardado {:.2f} segundos.".format(tiempo))
            metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
            metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=22)
    
            try:
                metricas = resultados[name]
                            
                if metricas[0] > metric_CH and metricas[1] > metric_SC:
                    resultados[name] = np.array([round((metric_CH),4),round((metric_SC),4)])
                    n_mejor = n
                    predicciones[name] = cluster_predict
                elif metricas[0] - metric_CH >= 200:
                    resultados[name] = np.array([round((metric_CH),4),round((metric_SC),4)])
                    n_mejor = n
                    predicciones[name] = cluster_predict
            
            except:
                resultados[name] = np.array([round((metric_CH),4),round((metric_SC),4)])
                n_mejor = n
                predicciones[name] = cluster_predict
               
    #tenemos en resultados las mejores metricas y en predicciones las predicciones que han generado
    #las mejores metricas
    
    # =============================================================================
    #     MOSTRAMOS LOS RESULTADOS DE LAS MEJORES METRICAS
    # =============================================================================
    print("El mejor numero de clusteres es:",n_mejor)
    for name,alg in clustering:
        print("Metricas de",name+":")
        print("\tCH:",resultados[name][0])
        print("\tS:",resultados[name][1])
                
    # =============================================================================
    # #REPRESENTACION APROXIMADA DE LOS CLUSTERES
    # =============================================================================
    
    embedding = MDS(n_components=2)
    X_transformed = embedding.fit_transform(X_normal)
    
    plt.scatter(X_transformed[:,0],X_transformed[:,1])
    plt.title("Datos originales")
    plt.show()
    
    for name,alg in clustering: 
        cluster_predict = predicciones[name]
        metricas = resultados[name]
        for i in np.unique(cluster_predict):
            aber = X_transformed[i==cluster_predict]
            aber_d = pd.DataFrame(aber)
            cadena = "cluster "+str(i)
            plt.scatter(aber[:,0],aber[:,1],label=cadena)
            plt.legend()
            plt.title("Clasificacion de "+name)
        plt.show()
    
        
    
    # =============================================================================
    # FILTRAMOS OUTLIERS
    # =============================================================================
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    X_cluster = pd.concat([X, clusters], axis=1)
    
    k = len(set(cluster_predict))
    min_size = 7
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
    k_filtrado = len(set(X_filtrado['cluster']))
    print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
    print("\n")
    n1 = k_filtrado
    X_filtrado = X_filtrado.drop('cluster', 1)
    X_filtrado = preprocessing.normalize(X_filtrado, norm='l2')
    
    if n1 < n_mejor:
        #EJECUCION DE K MEANS DESPUES DE FILTRAR LOS OUTLIERS
        k_means = KMeans(init='k-means++', n_clusters=n1, n_init=5)
        cluster_predict = k_means.fit_predict(X_filtrado) 
        
        #VISUALIZACION DE RESULTADOS DESPUES DE FILTRAR OUTLIERS
#        for i in list(range(n)):
#            dataframe = pd.DataFrame(X_filtrado[i==cluster_predict])
#            dataframe.hist()
#            plt.show()
    else:
        print("No han aparecido outliers que filtrar.")
    

    # COMPROBAMOS DIFERENCIA DE METRICAS FILTRANDO OUTILERS
    # Repetimos experimento inicial pero con los datos filtrados

    for name, alg in clustering:
        cluster_predict = alg.fit_predict(X_filtrado)
        print("Metricas antes de", name + ":")
        print("\tCH:", resultados[name][0])
        print("\tS:", resultados[name][1])
        metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
        metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean',
                                             sample_size=floor(muestra_silhoutte * len(X)), random_state=22)
        print("Metricas despues de", name + ":")
        print("\tCH:", metric_CH)
        print("\tS:", metric_SC)
    # =============================================================================
    # DENDOGRAMA Y HEATMAP
    # =============================================================================
    X_filtrado = X
    X_filtrado_normal = X_normal
    ##Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
    from scipy.cluster import hierarchy
    linkage_array = hierarchy.ward(X_filtrado_normal)
    print(linkage_array)
    plt.figure(1)
    plt.clf()
    dendro = hierarchy.dendrogram(linkage_array,orientation='left') #lo pongo en horizontal para compararlo con el generado por seaborn
    #puedo usar, por ejemplo, "p=10,truncate_mode='lastp'" para cortar el dendrograma en 10 hojas
    
    #Ahora lo saco usando seaborn (que a su vez usa scipy) para incluir un heatmap
    import seaborn as sns
    X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)
    sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
    
    
    # =============================================================================
    # SCATTERMATRIX DE LA CORRELACION ENTRE LAS VARIABLES
    # =============================================================================
    
    print("---------- Preparando el scatter matrix...")
    #Nos quedamos con el algoritmo que mejor metrica haya obtenido
    cadena = str("scatter_matrix"+str(i)+".png")
    i += 1
    mejor1 = 0
    mejor2 = 0
    for name,alg in clustering: 
        metricas = resultados[name]
        if metricas[0] > mejor1 and metricas[1] > mejor2:
            mejor1 = metricas[0]
            mejor2 = metricas[1]
            cluster_predict = predicciones[name]
            nombre = name
        elif metricas[0] - mejor1 >= 200:
            mejor1 = metricas[0]
            mejor2 = metricas[1]
            cluster_predict = predicciones[name]
            nombre = name
            
    print("El algoritmo escogido es:",nombre)
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    #se añade la asignación de clusters como columna a X
    X_kmeans = pd.concat([X, clusters], axis=1)
    sns.set()
    variables = list(X_kmeans)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig(cadena)
    print("")


print("Tiempo ejecucion total:",time.time()-time_inicio,"segundos.")