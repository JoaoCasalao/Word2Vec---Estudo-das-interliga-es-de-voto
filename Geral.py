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
#"PSD", "CDU"
part = ["CDU"]
dims = ["Dim5", "Dim5Wind", "Dim10", "Dim10Wind", "Dim19", "Dim19Wind"]

for i in part:
    
    geral(dimLayer = 5, dimSVD = 5, window = 10, nomeData = "_" + i, nomeModelo = "Model",
    pastaDestino = "SemPonto_TudoJunto/" + i, partido = i, tipo = "Dim5", Train = False, dimRed = True, decomp = False, Plot = True)
    
    geral(dimLayer = 10, dimSVD = 10, window = 10, nomeData = "_" + i, nomeModelo = "Model",
    pastaDestino = "SemPonto_TudoJunto/" + i, partido = i, tipo = "Dim10", Train = False, dimRed = True, decomp = False, Plot = True)

    geral(dimLayer = 19, dimSVD = 10, window = 10, nomeData = "_" + i, nomeModelo = "Model", 
          pastaDestino = "SemPonto_TudoJunto/" + i, partido = i, tipo = "Dim19", Train = False, dimRed = True, decomp = False, Plot = True)
    
    geral(dimLayer = 5, dimSVD = 5, window = 20105, nomeData = "_" + i, nomeModelo = "ModelWind", 
      pastaDestino = "SemPonto_TudoJunto/" + i, partido = i, tipo = "Dim5_Wind", Train = False, dimRed = True, decomp = False, Plot = True)

    geral(dimLayer = 10, dimSVD = 10, window = 20105, nomeData = "_" + i, nomeModelo = "ModelWind",
      pastaDestino = "SemPonto_TudoJunto/" + i, partido = i, tipo = "Dim10_Wind", Train = False, dimRed = True, decomp = False, Plot = True)
    
    geral(dimLayer = 19, dimSVD = 10, window = 20105, nomeData = "_" + i, nomeModelo = "ModelWind",
      pastaDestino = "SemPonto_TudoJunto/" + i, partido = i, tipo = "Dim19_Wind", Train = False, dimRed = True, decomp = False, Plot = True)


#ComPonto
    geral(dimLayer = 5, dimSVD = 5, window = 10, nomeData = "_" + i, nomeModelo = "Model",
          pastaDestino = "ComPonto_Tudo/" + i, partido = i, tipo = "Dim5", Train = False, dimRed = True, decomp = False, Plot = True)

    geral(dimLayer = 10, dimSVD = 10, window = 10, nomeData = "_" + i, nomeModelo = "Model",
          pastaDestino = "ComPonto_Tudo/" + i, partido = i, tipo = "Dim10", Train = False, dimRed = True, decomp = False, Plot = True)

    geral(dimLayer = 19, dimSVD = 10, window = 10, nomeData = "_" + i, nomeModelo = "Model",
          pastaDestino = "ComPonto_Tudo/" + i, partido = i, tipo = "Dim19", Train = False, dimRed = True, decomp = False, Plot = True)
    
    geral(dimLayer = 5, dimSVD = 5, window = 20105, nomeData = "_" + i, nomeModelo = "ModelWind",
          pastaDestino = "ComPonto_Tudo/" + i, partido = i, tipo = "Dim5_Wind", Train = False, dimRed = True, decomp = False, Plot = True)

    geral(dimLayer = 10, dimSVD = 10, window = 20105, nomeData = "_" + i, nomeModelo = "ModelWind",
          pastaDestino = "ComPonto_Tudo/" + i, partido = i, tipo = "Dim10_Wind", Train = False, dimRed = True, decomp = False, Plot = True)

    geral(dimLayer = 19, dimSVD = 10, window = 20105, nomeData = "_" + i, nomeModelo = "ModelWind",
          pastaDestino = "ComPonto_Tudo/" + i, partido = i, tipo = "Dim19_Wind", Train = False, dimRed = True, decomp = False, Plot = True)
    
    
    print(i)

    