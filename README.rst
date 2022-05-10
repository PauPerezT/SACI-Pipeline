==========
SACI-Pipeline
==========

"SACI-Pipeline: Satisfaction Analysis for Customer Interactions"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This repository conatins the proposed pipeline for satisfaction analysis for customer interactions developed for Pratech group.
It uses some pre-trained embedding for NLP and a combination of pre-trained embeddings and basic acoustic features for the speech analysis.
The recording used for training were divided in to segments using a voiced activity detection algorithm and then weighted using some statistics.
Label 0 correspond to a non-satisfied customer and 1 to a satisfied.

The code for this project is available at https://github.com/PauPerezT/SACI-Pipeline

   
From this repository::

    git clone https://github.com/PauPerezT//SACI-Pipeline
    
Install
^^^^^^^

To install the requeriments, please run::

    install.sh


Quickstart
^^^^^^^^^^


Run Example::

    python run_pipeline.py 
    
    
Run it automatically from linux terminal
-----------------------------------------

To compute the predictions automatically



====================  ===================  =====================================================================================
Optional arguments    Optional Values      Description
====================  ===================  =====================================================================================
-fa                                         Path folder of the audio file (Only wav format, 1 channel, sampling rate=16000Hz).
-ft                                         Path folder of the txt file (Only txt format).
                                           
                                            By default './audios'
-cu                    True, False          Boolean value for using cuda to compute the 
                                            
                                           embeddings (True). By default False.        
-mdl                  nlp, speech          Choose between three different classification models.

                      nlp_speech           By default nlp				                                                   
====================  ===================  =====================================================================================





    
Usage Example::

    python run_pipeline.py -fa ./audios/ -ft ./txts/ -cu True -mdl nlp
    

    
Results:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
20 bootstraping with a internal cross-validation were performed for each case.

====================  ===================  ======================  =======================
Features              NLP                  Speech                  NLP+Speech
====================  ===================  ======================  =======================
UA Recall (std)       74,42% (9.26)        62.00% (7.58)           76.29% (8.41)
SC Recall (std)       71.05% (9.10)        64.16% (14.77)          70.99% (6.25)
NSC Recall (std)      77.80% (14.28)       59.84% (12.86)          81.59% (16.33)
====================  ===================  ======================  =======================

SC: Satisfied customer
NSC: non-satified costumer
UA: Unweighted average

