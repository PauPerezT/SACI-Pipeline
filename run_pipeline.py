# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:38:13 2022

@author: Paula Perez
"""




import numpy as np
import pandas as pd
from fts.ac_functions import acoustics
from fts.BETO import WEBERT
import pickle
import argparse
from fts.utils import str2bool


if __name__ == '__main__':
    
    
    
    
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('-fa','--files_audio', default='./audios/',help='File folder of the set of documents', action="store")
    parser.add_argument('-ft','--file_txt', default='./txts/',help='Path to save the embeddings', action="store")
    parser.add_argument('-cu','--cuda', type=str2bool, nargs='?', const=True, default=True, help='Boolean value for using cuda to compute the embeddings (True). By defaul False.', choices=(True, False))
    parser.add_argument('-mdl','--model', default='nlp_speech', action="store", choices=('speech', 'nlp', 'nlp_speech'))


    #parser.print_help()
    args = parser.parse_args()
    
    path_audio=args.files_audio
    path_txt=args.file_txt

    cuda=args.cuda
    
    model=args.model
    

    
    
    if model=='nlp_speech':
    
        ac=acoustics(path_audio)
        bt=WEBERT(cuda=cuda)#
        
        ac_fts=ac.get_fts()
        bt_fts=bt.return_df(path_txt)#
        file_names=bt_fts.columns[1:]

        
        with open('./models/ac_nlp.pkl', 'rb') as fin:
            standar,clf = pickle.load(fin)
        
        fts=[np.hstack((ac_fts[f],bt_fts[f])) for f in file_names]
        fts=np.vstack(fts)
        
        fts_stardar=standar.transform(fts)
        X_new_preds = clf.predict(fts_stardar)
        
        pred=np.vstack((np.hstack(file_names),X_new_preds)).T
        
        
        df=pd.DataFrame(pred, columns=['File Name', 'Predictions'])
        df.to_csv('nlp-speech_predictions.csv', columns=['File Name', 'Predictions'], index=None)
        
    elif model=='speech':
        ac=acoustics(path_audio)
        ac_fts=ac.get_fts()
        file_names=ac_fts.columns[1:]
        
        
        ac_fts=np.array(ac_fts.iloc[:,1:]).T
        with open('./models/ac_model.pkl', 'rb') as fin:
            standar,clf = pickle.load(fin)
            
        fts_stardar=standar.transform(ac_fts)
        X_new_preds = clf.predict(fts_stardar)
        
        pred=np.vstack((np.hstack(file_names),X_new_preds)).T
        
        
        df=pd.DataFrame(pred, columns=['File Name', 'Predictions'])
        df.to_csv('speech_predictions.csv', columns=['File Name', 'Predictions'], index=None)
        
        
    else:
        bt=WEBERT(cuda=cuda)
        bt_fts=bt.return_df(path_txt)#
        file_names=bt_fts.columns[1:]
        
        
        bt_fts=np.array(bt_fts.iloc[:,1:]).T
        with open('./models/nlp_model.pkl', 'rb') as fin:
            standar,clf = pickle.load(fin)
        fts_stardar=standar.transform(bt_fts)
        X_new_preds = clf.predict(fts_stardar)
        
        pred=np.vstack((np.hstack(file_names),X_new_preds)).T
        
        
        df=pd.DataFrame(pred, columns=['File Name', 'Predictions'])
        df.to_csv('nlp_predictions.csv', columns=['File Name', 'Predictions'], index=None)
    
   
    
    

    

    
