import numpy as np
from scipy.linalg import svd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def DimRed(dimLayer, dimSVD, pastaDestino, nomeData, nomeModelo, tipo, decomp):

    for modelo in os.listdir("Plots/" + pastaDestino + "/" + tipo):
        if not "SVD" in modelo: 
            file= os.path.join('Plots', pastaDestino, tipo, modelo)
            data = np.load(file)
        
            if "Mtx_2d" not in modelo:
                
                for n_component in range(dimLayer,0,-1):
                    
                    if n_component == dimLayer or n_component <= dimSVD:
            
                        nameFig = modelo.replace("Mtx_Tudo.npy", "") + "_SVD" + str(n_component) + ".png" 
                        nameFig = nameFig.replace("Dados_", "")
                        
                        nameMatrix = modelo.replace("Mtx_Tudo.npy", "") + "_Matrix" + str(n_component)
                        Mtx_2d_name = nameMatrix.replace("Dados_", "PlotTSNE")
                        
                        Mtx_2d_name = os.path.join('Plots' + "/", pastaDestino + "/", tipo + "/", "SVD",Mtx_2d_name)
                        nameMatrix = os.path.join('Plots' + "/", pastaDestino + "/", tipo + "/", "SVD", nameMatrix)
                        
                        rows, columns = data.shape
                        
                        #Singular-value decomposition
                        U, s, VT = svd(data)
                        
                        if decomp:
                            S = np.zeros((data.shape[0], data.shape[1]))
                            
                            #populate Sigma with n x n diagonal matrix
                            S[:data.shape[1], :data.shape[1]] = np.diag(s)
                            
                            #Recombine matrix S and VT
                            S = S[:, :n_component]
                            VT = VT[:n_component, :]
                            
                            #Remake priginal matrix 
                            A = U.dot(S.dot(VT))
                            
                            tsne = TSNE(n_components = 2, random_state = 0, metric="cosine", square_distances=True)
                            word_vectors_matrix_2dCosine = tsne.fit_transform(A)
                                
                            np.save(Mtx_2d_name, word_vectors_matrix_2dCosine)
                            np.save(nameMatrix, A)
                        
                        
                        if n_component == dimSVD:
                             
                            x_data= list(range(1, len(s) + 1))
                            y_data=s
                            fig, ax = plt.subplots()
                            ax.plot(x_data, y_data, 'go')
                             
                            nomeGraph = modelo.replace("Dados_", "")
                            nomeGraph = nomeGraph.replace("Mtx_Tudo.npy", ": SVD")
                            plt.ylabel("Valores Singulares")
                            plt.xlabel("Posição dos Valores Singulares na diagonal da matriz")
                            plt.title(nomeGraph)
                            

                            
                            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                             
                            plt.show()
                            
                            nameFig = os.path.join("Resultados/", pastaDestino + "/", tipo + "/", nameFig)
                            
                            
                            plt.savefig(nameFig, bbox_inches='tight')
                            plt.close()
        
        print("SVD completed")
        
    return None
                    
                    
                
                    
                    