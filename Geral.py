from __future__ import absolute_import, division, print_function
import os
from DimRed import DimRed
from ModelTrain import ModelTrain
from WordSpacePlot import WordSpacePlot

def geral(dimLayer, dimSVD, window, nomeData, nomeModelo, pastaDestino, partido, tipo, Train=False, dimRed=False, decomp = False, Plot=False):
    
    if not os.path.exists("Trained/" + pastaDestino + "/" + tipo):
        os.makedirs("Trained/" + pastaDestino + "/" + tipo)
        
    if not os.path.exists("Plots/" + pastaDestino + "/" + tipo):
        os.makedirs("Plots/" + pastaDestino + "/" + tipo)
        
    if not os.path.exists("Plots/" + pastaDestino + "/" + tipo + "/" + "SVD"):
        os.makedirs("Plots/" + pastaDestino + "/" + tipo + "/" + "SVD")
        
    if not os.path.exists("Resultados/" + pastaDestino + "/" + tipo):
        os.makedirs("Resultados/" + pastaDestino + "/" + tipo)
        
    if Train: 
        ModelTrain(dimLayer, window, pastaDestino, nomeData, nomeModelo, tipo)
    
    if dimRed:
        DimRed(dimLayer, dimSVD, pastaDestino, nomeData, nomeModelo, tipo, decomp)
    
    if Plot:
        WordSpacePlot(dimLayer, dimSVD, window, nomeData, nomeModelo, pastaDestino, partido, tipo)
    
    return None
