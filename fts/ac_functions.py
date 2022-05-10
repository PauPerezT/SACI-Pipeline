# -*- coding: utf-8 -*-
"""
Created on Mon May  9 21:48:54 2022

@author: Paula Perez
"""
import numpy as np
import os
from tqdm import tqdm
import scipy.stats as st
import pandas as pd
import parselmouth
from parselmouth.praat import call
import sys
from scipy.io.wavfile import read
import scipy.integrate as integrate
#Acoustic Functions
class acoustics:
    """
    Computes some basic acoustic features. 
    This features were extracted using some libraries such as numpy, parselmouth, stats, and some self implemented features.

    
    :param path: Audio path (It has to be a folder with several audios or only one).
    :returns: acoustics object
    """    
    
    def __init__(self,path): 
        
        self.path=path
        #Predefined parameters
        self.step_size=0.01
        self.window_size=0.025
        self.unit='Hertz'
        self.f0min=70 
        self.f0max=500
        self.fs=16000
        
    
    
    def get_fts(self):
        """
        :returns: a dataframe with the computed features. The features are based on energy, F0, and speech rates
        """
        #Read audio
         
        pbar = tqdm(os.listdir(self.path))
        nns= np.hstack(['W2Vnn_'+str(i) for i in range(1024)])
        features=[]
        file_names=[]
        for file in pbar:
            pbar.set_description("Audio Processing %s" % file[:-4])
            
            subpath=os.path.join(self.path,file)
            snd = parselmouth.Sound(subpath)
            fs, sig = read(subpath)
  
            
            #Pitch based
            f0,stdev,harmonicity, jitter, shimmer, voiced_rt=self.pitchBase_extract(snd, self.f0min,  self.f0max, self.step_size, fs)
    
    
            meanF0=f0  # get mean pitch
            stdF0=stdev  # get standard deviation
            
            #Energy coutour
            
            energy=np.hstack(self.compute_energy(sig,size=self.window_size, step=self.step_size, fs=fs))
            #Energy countour
            en_mn=np.mean(energy)
            en_std=np.std(energy)
            en_skew=st.skew(energy)
            en_kurt=st.kurtosis(energy)
            en_max=np.max(energy)
        
            file_names.append(file[:-4])
            
            features.append(np.hstack(( meanF0, stdF0, harmonicity, jitter, shimmer, voiced_rt, en_mn, en_std, en_skew, en_kurt, en_max)))
        fts_names=np.hstack(('meanF0', 'stdF0', 'HNR', 'Jitter', 'Shimmer', 'VRT','meanEC', 'stdEC', 'skewEC', 'kurtEC', 'maxEC')).reshape(-1,1)
        features=np.hstack((fts_names, np.vstack(features).T))
        file_names=np.hstack(file_names)
        return pd.DataFrame(features, columns=[np.hstack(('Features',file_names ))], index=None)
    


    def __voiced_segments(self,F0, data_audio, step_size):
        pitch_seg = np.where(F0!=0)[0]
    
        dchange = np.diff(pitch_seg)
        change = np.where(dchange>1)[0]
        if len(pitch_seg)==0:
            return []
        init_seg= (pitch_seg[0]*step_size)+step_size
        segment=[]
        for indx in change:
            end_seg = (pitch_seg[indx]*step_size)+step_size
            seg = data_audio[int(init_seg):int(end_seg)]
            init_seg = (pitch_seg[indx+1]*step_size)+step_size
            segment.append(seg)
        return segment     
    
    def __compute_VoicedRate(self,v_seg,data_audio, fs):
        return fs*float(len(v_seg))/len(data_audio)
    
    def pitchBase_extract(self, snd, f0min, f0max,step_size, fs, unit='Hertz'):
        pitch = call(snd, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object
        f0=call(pitch, "Get mean", 0, 0, unit)
        F0=snd.to_pitch(time_step=step_size).selected_array['frequency']
        stdevF0=call(pitch, "Get standard deviation", 0, 0, unit)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.025, 1.0)
        harmonicity=call(harmonicity, "Get mean", 0, 0)
        pointProcess = call(snd, "To PointProcess (periodic, cc)", f0min, f0max)
        jitter=call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer=call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        
        v_seg=self.__voiced_segments(F0, np.hstack(np.array(snd)), int(step_size*fs))
        voiced_rt=self.__compute_VoicedRate(v_seg,np.hstack(np.array(snd)), fs)
        return f0,stdevF0,harmonicity, jitter, shimmer, voiced_rt



    
    def __extract_windows(self, signal, size, step):
        # make sure we have a mono signal
        assert(signal.ndim == 1)
    
       
        n_frames = int((len(signal) - size) / step)
       
        # extract frames
        windows = [signal[i * step : i * step + size]
                   for i in range(n_frames)]
       
        # stack (each row is a window)
        return np.vstack(windows)


    def compute_energy(self, signal,size=0.025, step=0.01, fs=16000):
        size=int(0.025*fs) #25 miliseconds window -> to samples 0.025*fs
        step=int(0.010*fs) #10 miliseconds hop
        
        signal=signal/max(signal)
        frames=self.__extract_windows(signal, size, step)
        energy=[]
    
        for fr in frames:
    
            #Square of the signal
            x2 = fr**2
        
            #t
            
            t=np.arange(0, float(len(fr))/fs, 1.0/fs) # Time vector
            
            #integral
            energy.append(integrate.simps(x2,t))
    
        return np.hstack(energy)
    

    
    
#%%

