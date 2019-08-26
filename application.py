#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:16:04 2017
@author: gag 

Script that applies the models (MLR, MLP and MARS), previously trained, to a dataset and obtains
the Taylor diagram for the comparison of the behavior of each one.

"""

import matplotlib.pyplot as plt
import numpy as np
import statistics
import sklearn
import lectura

from taylorDiagram import TaylorDiagram


def taylorGraph(y0, y1, y2, y3):
    """ Generate taylor diagram

    Parameters:
    -----------
    y0 : reference dataset
    y1...y3 : aproximated dataset by models

    Returns: 
    --------
    taylor Graph plot

    """

    data = y0 #CONAE
    #print data
    refstd = data.std(ddof=1) # Reference standard deviation

    # Models
    m1 = y1 # MLR
    m2 = y2 # MLP
    m3 = y3 # MARS

    # Compute stddev and correlation coefficient of models
    samples = np.array([ [m.std(ddof=1), np.corrcoef(data, m)[0,1]]
                         for m in (m1, m2)])
    #samples = np.array([ [sklearn.metrics.r2_score(data, m)**2, np.corrcoef(data, m)[0,1]]
                         #for m in (m1,m2)])
    #samples = np.array([ [statistics.RMSE(data,m), np.corrcoef(data, m)[0,1]]
                         #for m in (m1,m2)])

    #print "AQUI!!" + str(samples)
    fig = plt.figure(10, facecolor="white")

    #ax1 = fig.add_subplot(1,1,1, xlabel='X', ylabel='Y')
    # Taylor diagram
    dia = TaylorDiagram(refstd, fig=fig, rect=111, label="Conae")

    colors = plt.matplotlib.cm.jet(np.linspace(0,1,len(samples)))

    model = [ "MLR", "MLP", "MARS"]
    color = ["green", "blue", "red" ]
    markers = ["s", "o", "-"]


    # Add samples to Taylor diagram
    for i,(stddev,corrcoef) in enumerate(samples):
        dia.add_sample(stddev, corrcoef, marker=markers[i],markersize=7, ls='', c=color[i],
                       label= str(model[i]))

    # Add RMS contours, and label them
    contours = dia.add_contours(colors='0.5')
    plt.clabel(contours, inline=1, fontsize=10)

    # Add a figure legend
    fig.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size='small'), loc='upper right')

    plt.show()



def application(nameFile, MLRmodel, MLPmodel, MARSmodel):
    """ Function that aplly models to data set

    Parameters:
    -----------
    nameFile : String instances that contain file path
    MLRmodel, MLPmodel, MARSmodel : trained models

    Returns: 
    --------
    taylor Graph plot

    """

    if (type == "etapa1"):
        data = lectura.lecturaAplicacion(nameFile)
        data2 = lectura.lecturaAplicacionMLP(nameFile)
        del data2["RSOILMOIST"]
        del data2["SMAP"]
    if (type == "etapa2"):
        data = lectura.lecturaAplicacion(nameFile)
        data2 = lectura.lecturaAplicacionMLP(nameFile)
        del data2["RSOILMOIST"]
        del data2["SMAP"]
    #del data2["SM10Km_PCA"]
    #del data2["NDVI_30m_B"]
    #print data2
    ySmap = np.array(data["SMAP"])
    del data["SMAP"]
    pred = MLRmodel.predict(data)
    yMLR = (10**(np.array(pred)))
    yConae = (np.array(data["RSOILMOIST"]))


    yMLP = MLPmodel.predict(data2)

    taylorGraph(yConae, yMLR, yMLP, MARS )

    plt.show()


