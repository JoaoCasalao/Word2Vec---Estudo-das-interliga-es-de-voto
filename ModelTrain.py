from __future__ import absolute_import, division, print_function
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
import multiprocessing
import os
import re
import gensim.models.word2vec as w2v
import numpy as np

def ModelTrain(dimLayer, window, pastaDestino, nomeData, nomeModelo, tipo):

    for doc in  os.listdir("Dados/" + pastaDestino):
        
        fileName = os.path.join("Dados",pastaDestino, doc)
        file = open(fileName, encoding='utf-8', errors='ignore')
        ficheiro = file.read()
        file.close()
        
        raw_days = sent_tokenize(ficheiro)
        
        def word_tokenizer(raw):
            clean = re.sub("[^a-zA-Z]"," ", raw)
            words = word_tokenize(clean)
            return words
        
        sentences = []
        
        for raw_day in raw_days:
            if len(raw_day) > 0:
                sentences.append(word_tokenizer(raw_day))
                
        #print("sentences:", sentences)
        
        # Tokens Counter
        token_count = sum([len(sentence) for sentence in sentences])
        
        # Train The Model
        Model = w2v.Word2Vec(
            sg = 1, #Skip-Gram
            workers = multiprocessing.cpu_count(),
            vector_size = dimLayer,
            max_final_vocab = token_count,
            min_count = 1,
            window = window,
            #sample = 1e-4
        )
        
        Model.build_vocab(sentences)
        
        print("model count: ", Model.corpus_count)
        
        
        Model.train(sentences, total_examples=Model.corpus_count,
                               epochs=Model.epochs)
        
        
        # Guardar modelo na pasta
        nameModel = doc.replace(nomeData + ".txt",  nomeModelo + ".w2v")
        Model.save(os.path.join("Trained/" + pastaDestino + "/" + tipo, nameModel))
        
        vocab_len = len(Model.wv)
        print("Vocabulary length is ", vocab_len)
        
        # Define Matix
        word_vectors_matrix = np.ndarray(shape=(vocab_len, dimLayer), 
                                            dtype='float64')
        word_list = []
        i = 0
        
        # Fill the Matix
        for word in Model.wv.key_to_index:
            word_vectors_matrix[i] = Model.wv[word]
            word_list.append(word)
            i += 1
            if i == vocab_len:
                break
    
        # Guardar matrix
        name = nameModel.replace(nomeModelo + ".w2v", "Mtx_Tudo")
        
        Mtx_name = os.path.join("Plots/" , pastaDestino + "/", tipo + "/", name)
        np.save(Mtx_name, word_vectors_matrix)
        
    print("Models transformed into vectors")
        
    print("Models Trained")
    
    return None
