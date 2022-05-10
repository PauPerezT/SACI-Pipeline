
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries
from torch.utils.data import Dataset
import torch
import pandas as pd
import csv

import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv
import os
import gc
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertModel,logging

from tqdm import tqdm
import sys
sys.path.append('./fts')
from utils import noPunctuation, StopWordsRemoval,noPunctuationExtra, removeNumbers, removeURL,removeNonAlphabet
import copy

from transformers import AutoTokenizer, AutoModel
from scipy.stats import kurtosis, skew



#%%


if torch.cuda.is_available():
    
    torch.cuda.empty_cache()
    torch.cuda.get_device_name(0)
    n_gpu = torch.cuda.device_count()



                
#%%
class WEBERT:
    """
    WEBERT computes BETO to get static or dynamic embeddings. 
    BETO is a pretrained BERT model from spanish corpus (https://github.com/dccuchile/beto).
    BETO uses Transformers (https://github.com/huggingface/transformers). 
    It can be computed using only spanish model.
    Also considers cased or uncased options, and stopword removal.
    
    :param inputs: input data
    :param file: name of the document.
    :param stopwords: boolean variable for removing stopwords (By defalut: False).
    :param model: base or large model (By defalut: base).
    :param cased: boolean variable to compute cased or lower-case model (By defalut: False).
    :param cuda: boolean value for using cuda to compute the embeddings, True for using it. (By defalut: False).
    :returns: WEBERT object
    """    
    
    def __init__(self, stopwords=False, model='base', cased=False, cuda=False):   
        

        
        
        self.data=''
        self.words=[]
        self.word_counting=[] 

        
        self.stopwords=stopwords

        self.neurons=768
        if model=='large':
            self.neurons=1024
        cased_str='cased'
        self.cased=cased
        if cased:
            cased_str='cased'
        
        self.model='dccuchile/bert-'+model+'-spanish-wwm'+'-'+cased_str
        if cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device='cpu'
        
        
        
    def preprocessing(self,inputs):
        
        """
        Text Pre-processing
        
        :param inputs: input data
        :returns: proprocessed text
        """
        data=inputs
        
        docs=[]
        for j in range (len(data)):
            
            text =data[j]
            
    
            text_aux=copy.copy(text)
            text_aux=removeURL(text_aux)
            text_aux=noPunctuationExtra(text_aux)
            text_aux=removeNumbers(text_aux)
            
            text_aux=text_aux.replace('. '," [SEP] " )
            if text_aux[-5:]=="[SEP]":
                text_aux=text_aux[0:-5]

            
            text_aux=text_aux.replace('.',' ')
            text_org=noPunctuationExtra(text.replace('.',' '))
            text_org=removeURL(text_org)
            text_org=noPunctuation(text_org)
            text_org=removeNumbers(text_org)
            
            
            if self.stopwords:
                text=StopWordsRemoval(text_aux,self.language)
            self.words.append(text_org.split())
            docs.append(text_aux)
        return docs
            
    
    def __data_preparation(self):
        
        """
        Data preparation and adaptation for BETO to work properly

        """
        
        # add special tokens for BERT to work properly
        data=self.preprocessing(self.data)
        
        sentences = ["[CLS] " + query + " [SEP]" for query in data]


        
        # Tokenize with BERT tokenizer
        
        tokenizer = BertTokenizer.from_pretrained(self.model, do_lower_case=self.cased)
       

        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        self.word_counting= [len(words)-1 for words in tokenized_texts]

        self.tokenized_texts=tokenized_texts
        

        # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
        self.indexed_tokens = [np.array(tokenizer.convert_tokens_to_ids(tk)) for tk in tokenized_texts]

        data_ids = [torch.tensor(tokenizer.convert_tokens_to_ids(x)).unsqueeze(0) for x in tokenized_texts]

        
        # Create an iterator of our data with torch DataLoader 
        
 
        self.data_dataloader = DataLoader(data_ids,  batch_size=1)
        
        
                      
        
    def get_bert_embeddings(self, input_data):
        """
        BETO embeddings computation using Transformes. It store and transforms the texts into BETO embeddings. The embeddings are stored in csv files.
        
        :param path: path to save the embeddings
        :returns: static embeddings 
        
        """  
        self.data=input_data
        self.__data_preparation()
        
        data_stat=[]
        logging.set_verbosity_error()
        bert = BertModel.from_pretrained(self.model).embeddings
        bert=bert.to(self.device)



        for idx_batch, sequence in enumerate(self.data_dataloader,1):
            sequence=sequence.to(self.device)

            ids_tokens=np.where((self.indexed_tokens[idx_batch-1]!=3) &(self.indexed_tokens[idx_batch-1]!=5) & (self.indexed_tokens[idx_batch-1]!=4))[0]
            tokens=np.array(self.tokenized_texts[idx_batch-1])[ids_tokens]
            index=[]
            index_num=[]
            for i in range(len(tokens)):
                if [idx for idx, x in enumerate(tokens[i]) if x=='#'] ==[]:
                    index.append(i)
                else:
                    index_num.append(i)
                
            

            bert_embeddings=bert(sequence)[0][:,ids_tokens].cpu().detach()

            embeddings=torch.tensor(np.zeros((bert_embeddings.shape[1]-len(index_num),bert_embeddings.shape[2])))
            count=0
            if index_num!=[]:
                for idx in range (len(ids_tokens)):
                     if np.where(index_num==np.array([idx]))[0].size!=0:
                         nums=bert_embeddings[0][idx]*bert_embeddings[0][idx-1]
                         embeddings[idx-count-1]=nums.cpu().detach()
                         count+=1
                     else:
                         embeddings[idx-count]=bert_embeddings[0][idx].cpu().detach()
            else:
                
                embeddings=bert_embeddings[0]
            
            
            for emb in embeddings:
                data_stat.append(emb)
                    
                

                
        wordar=np.vstack(data_stat)
        del data_stat
        meanBERT=np.mean(wordar, axis=0)
        stdBERT=np.std(wordar, axis=0)
        kurtosisBERT=kurtosis(wordar, axis=0)
        skewnessBERT=skew(wordar, axis=0)
        skewnessBERT=skew(wordar, axis=0)
        minBERT=np.min(wordar, axis=0)
        maxBERT=np.max(wordar, axis=0)
        statisticalMeasures=np.hstack((meanBERT, stdBERT, kurtosisBERT, skewnessBERT,minBERT, maxBERT))
            
            
        gc.collect()       
        return statisticalMeasures
    
    def return_df(self, files_path):
        
        
        files=np.hstack(sorted([f for f in os.listdir(files_path) if f.endswith('.txt')]))
        
        j=0
        
        
        neurons=768
        
        labelstf=[]
        for n in range (neurons):
            labelstf.append('meanBETOnn_'+str(n)) 
        for n in range (neurons):
            labelstf.append('stdBETOnn_'+str(n)) 
        for n in range (neurons):
            labelstf.append('skewBETOnn_'+str(n)) 
        for n in range (neurons):
            labelstf.append('kurtBETOnn_'+str(n)) 
        for n in range (neurons):
            labelstf.append('minBETOnn_'+str(n)) 
        for n in range (neurons):
            labelstf.append('maxBETOnn_'+str(n)) 
        
        embs=[]    
        pbar=tqdm(files)
        files_names=[]
        for file in pbar:
            pbar.set_description("Text Processing %s" % file[:-4])
            
            try:
                
                data = pd.read_csv(os.path.join(files_path,file), sep='\t', header=None, encoding = "ISO-8859-1")
        
                
                data_input=list(data[0])
                
                j+=1
         
                data_stat=self.get_bert_embeddings(data_input)
                embs.append(data_stat)
                files_names.append(file[:-4])
                
                

            except:
                pass
        fts_names=np.hstack(labelstf).reshape(-1,1)
        features=np.hstack((fts_names, np.vstack(embs).T))
        
        return pd.DataFrame(features, columns=[np.hstack(('Features',np.hstack(files_names)))], index=None)    

            
            
                        
            


