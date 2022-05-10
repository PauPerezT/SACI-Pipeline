==========
SACI-Pipeline
==========

"SACI-Pipeline: Satisfaction Analysis for Customer Interactions"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This repository conatins the proposed pipeline for satisfaction anlysis for customer interactions developed for Pratech group.
It uses some pre-trained embedding for NLP and a combination of pre-trained embeddings and basic acoustic features for the speech analysis.
The recording used for training were divided in to segments using a voiced activity detection algorithm and then weighted using some statistics.


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
-h                                         Show this help message and exit
-f                                         Path folder of the txt documents (Only txt format). 
                                           
                                           By default './texts'
-sv                                        Path to save the embeddings. 

                                           By default './bert_embeddings'
-bm                   Bert, Beto, SciBert  Choose between three different BERT models.

                                           By default BERT				             
-d                    True, False          Boolean value to get dynamic features= True.

                                           By default True.                                         
-st                   True, False          Boolean value to get static features= True from the

                                           embeddings such as mean, standard deviation, kurtosis,
                                           
                                           skeweness, min and max. By default False.                       
-l                    english, spanish     Chosen language (only available for BERT model).

                                           By default english.                               
-sw                   True, False          Boolean value, set True if you want to remove

                                           stopwords. By default False.                                         
-m                    base, large          Bert models, two options base and large.
 
                                           By default base.                                   
-ca                    True, False         Boolean value for cased= True o lower-cased= False

                                           models. No avalaible for SciBert. By default False.
-cu                    True, False         Boolean value for using cuda to compute the 
                                            
                                           embeddings (True). By default False.                                                   
====================  ===================  =====================================================================================





    
Usage Example::

    python run_pipeline.py -f ./texts/ -sv ./bert_embs -bm Bert -d True -st True -l english -sw True -m base -ca True -cu True
    

    
Results:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

====================  ===================  ======================  =======================
Features              NLP                  Speech                  NLP+Speech
====================  ===================  ======================  =======================
UAR(std)              74,42%               62.00%                  76.29%
SC Recall(std)        71.05%               64.16%                  70.99%
NSC Recall(std)       77.80%               59.84%                  81.59%
====================  ===================  ======================  =======================




