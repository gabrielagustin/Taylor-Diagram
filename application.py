#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:16:04 2017
@author: gag 



"""

import matplotlib.pyplot as plt
import numpy as np
import statistics
import sklearn
import lectura

from taylorDiagram import TaylorDiagram

#v4
def taylorGraph(v1, v2, v3):
    # Reference dataset
    data = v1 #SMAP
    #print data
    refstd = data.std(ddof=1)           # Reference standard deviation

    # Models
    m1 = v2 # MLR
    m2 = v3 # MLP

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

    model = [ "MLR", "MLP"]
    color = ["green", "blue"]
    markers = ["s", "o"]

    #model = [ "MLR", "MLP", "SMAP"]
    #color = ["green", "blue", "red"]
    #markers = ["s", "d", "o"]

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



def application(nameFile, MLRmodel, MLPmodel, type):
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
    ySmap = ySmap*100

    taylorGraph(yConae,ySmap, yMLR,yMLP )


    ## se obtiene el error
    rmseC = 0
    rmseC = statistics.RMSE(yConae,yMLR)
    print("RMSE MLR:" + str(rmseC))
    biasConae = statistics.bias(yMLR,yConae)
    print("Bias MLR:" + str(biasConae))
    RR = sklearn.metrics.r2_score(yConae, yMLR)
    print("RR MLR:" + str(RR))


    rmseCS = 0
    rmseCS = statistics.RMSE(yConae, ySmap)
    print("RMSE SMAP:" + str(rmseCS))
    biasConae_SMAP = statistics.bias(ySmap,yConae)
    print("Bias SMAP:" + str(biasConae_SMAP))
    RR = sklearn.metrics.r2_score(yConae, ySmap)
    print("RR smap:" + str(RR))


    rmseCS = 0
    rmseCS = statistics.RMSE(yConae, yMLP)
    print("RMSE MLP:" + str(rmseCS))
    biasConae_SMAP = statistics.bias(yMLP,yConae)
    print("Bias MLP:" + str(biasConae_SMAP))
    RR = sklearn.metrics.r2_score(yConae, yMLP)
    print("RR MLP:" + str(RR))


    xx = np.linspace(0,len(yMLP),len(yMLP))

    v1 = []
    v2 = []
    v3 = []
    v4 = []
    for i in range(len(yConae)):
        v1.append(float(yConae[i]))
        v2.append(float(yMLR[i]))
        v3.append(float(yMLP[i]))
        v4.append(float(ySmap[i]))


    fig = plt.figure(1,facecolor="white")
    fig0 = fig.add_subplot(111)
    z = np.polyfit(v1,v2, 1)
    g = np.poly1d(z)
    fig0.plot(v1,g(v1),'green')

    z = np.polyfit(v1,v3, 1)
    g = np.poly1d(z)
    fig0.plot(v1,g(v1),'blue')

    z = np.polyfit(v1,v4, 1)
    g = np.poly1d(z)
    fig0.plot(v1,g(v1),'red')

    fig0.scatter(yConae, yMLR,color = 'green', s=10, linewidth=3, label='MLR')
    fig0.scatter(yConae, ySmap, color = 'red', s=30, marker = "*", label='SMAP')
    fig0.scatter(yConae, yMLP, color="blue", s=30, marker ="^", label='MLP')
    #fig0.scatter(xx, ySmap*100, color = 'black', s=65, facecolors='none', label='SM_Smap')
    #fig0.text(7.5,38, 'RMSE_in_situ=%5.3f' % rmseC, fontsize=10)
    #fig0.text(7.5, 37, 'RMSE_Smap=%5.3f' % rmseS, fontsize=10)
    #fig0.text(7.5, 36, 'RMSE_in_Situ_Smap=%5.3f' % rmseCS, fontsize=10)
    #fig0.set_title(str(type))
    #fig0.set_xlabel("Training percent[%]",fontsize=12)
    #fig0.set_ylabel(str(type) +" "+"[ad]",fontsize=12)
    fig0.set_xlabel("SM observations",fontsize=12)
    fig0.set_ylabel("MLR-SMAP [% GSM]",fontsize=12)
    #fig0.axis([14.9,45.1, 14.9,45.1])
    #fig0.axis([10,45.1, 10,45.1])
    fig0.plot([15, 40], [15, 40], ls="--", c=".3")
    #fig0.axis([5,45, 5,45])

    plt.grid(True)
    #fig0.axis([-1,12, 10,35])
    #plt.xticks(np.linspace(10, 90, 15, endpoint=True))
    plt.legend(loc=4, fontsize = 'small')

    plt.show()


