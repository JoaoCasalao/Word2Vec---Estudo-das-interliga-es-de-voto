from __future__ import absolute_import, division, print_function
import numpy as np
import gensim.models.word2vec as w2v
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from PIL import Image

def WordSpacePlot(dimLayer, dimSVD, window, nomeData, nomeModelo, pastaDestino, partido, tipo):
    

    for modelo in os.listdir("Trained/" + pastaDestino + "/" + tipo):
        
        for n_component in range(dimLayer,0,-1):
            
            if n_component == dimLayer or n_component <= dimSVD:
                # Open the trained model
                fileName = os.path.join("Trained", pastaDestino, tipo, modelo)
                model = w2v.Word2Vec.load(fileName)
                
                vocab_len = len(model.wv)
                
                
                # Open the trained model matrix
                mtx_name = modelo.replace(nomeModelo + ".w2v", "_Matrix" + str(n_component) + ".npy")
                mtx_name = mtx_name.replace("Dados_", "PlotTSNE")
                mtx_name = os.path.join('Plots' + "/", pastaDestino + "/", tipo + "/", "SVD", mtx_name)
                if "SemPonto_TudoJunto" in pastaDestino:
                    mtx_name = mtx_name.replace("_" + partido, "")
                word_vectors_matrix = np.load(mtx_name)
                
                
                word_list = []
                i = 0
                for word in model.wv.key_to_index:
                    #print('absinto')
                    word_list.append(word)
                    i += 1
                    if i == vocab_len:
                        break
                
                # Word points DataFrame
                points = pd.DataFrame([
                    (word, coords[0], coords[1])
                    for word, coords in [
                        (word, word_vectors_matrix[word_list.index(word)])
                        for word in word_list
                    ]
                ], columns=["Word", "x", "y"])
                
                sns.set_context("poster")
                fig, ax = plt.subplots()
                
                fig.subplots_adjust(right = 0.85, left = 0.15)
                
                # Plot the word points
                
                for i in points.index:
                    if 'PS' in points['Word'][i][-4:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = 'b', markersize=2, alpha = 0.2)
                        
                    if 'PSD' in points['Word'][i]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = '#FFA500', markersize=2, alpha = 0.2)
                        
                    if 'CH' in points['Word'][i][-4:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = '#808080', markersize=2, alpha = 0.2)
                        
                    if 'IL' in points['Word'][i][-4:-2]:
                       ax.plot(points.x[i], points.y[i], 'ro', color = 'm', markersize=2, alpha = 0.2)
                        
                    if 'BE' in points['Word'][i][-4:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = '#964B00', markersize=2, alpha = 0.2)
                        
                    if 'CDU' in points['Word'][i][-5:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = 'r', markersize=2, alpha = 0.2)
                        
                    if 'CDS' in points['Word'][i][-5:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = 'k', markersize=2, alpha = 0.2)
                        
                    if 'PAN' in points['Word'][i][-5:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = 'g', markersize=2, alpha = 0.2)  
                        
                    if 'L' in points['Word'][i][-3:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = 'y', markersize=2, alpha = 0.2)
                        
                    if 'Outro' in points['Word'][i][-7:-2]:
                        ax.plot(points.x[i], points.y[i], 'ro', color = '#8F00FF', markersize=2, alpha = 0.2)
                        
                
                """ax.text(max(points.x)*1.05, (min(points.y)+max(points.y))/2, 'PS - azul \nPSD - laranja \nCH - cinza \nIL - magenta \nBE - castanho \nCDU - vermelho \nCDS - preto \nPAN - verde  \nL - amarelo \nOutro - lilás', fontsize=7, style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.3, 'pad': 1})"""
                           
                # Defining Axes
                offset = 0.0
                ax.set_xlim(min(points.x) - offset, max(points.x) + offset)
                ax.set_ylim(min(points.y) - offset, max(points.y) + offset)
                
                nomeGraph = modelo.replace(nomeModelo + ".w2v", ": Dimensão" + str(dimLayer) + "-> Dimensão" + str(n_component))
                nomeGraph = nomeGraph.replace("Dados_", "")
            
                plt.title(nomeGraph)
                plt.show()
                
                nomePng = modelo.replace(nomeModelo + ".w2v", "_Dim" + str(dimLayer) + "paraDim" + str(n_component))
                nomePng = nomePng.replace("Dados_", "")
                nomePng = nomePng + ".png"
                nomePng = os.path.join("Resultados/", pastaDestino + "/", tipo + "/", nomePng)
                
                plt.savefig(nomePng, bbox_inches='tight')
                
                plt.close()
                
    return None
            
    