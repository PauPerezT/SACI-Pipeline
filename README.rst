==========
SACI-Pipeline
==========

"SACI-Pipeline: Satisfaction Analysis for Customer Interactions"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WEBERT is a python toolkit designed to help students to compute dynamic and static Bidirectional Encoder Representations from Transformers (BERT) embeddings (https://github.com/huggingface/transformers). WEBERT is available for english and spanish (multilingual) models, as well as for base and large models, and  cased and lower-cased options. BETO and SciBERT are also available here. BETO is a pretrained BERT model from a spanish corpus (https://github.com/dccuchile/beto). SciBERT is a pre-trained model on english scientific text (https://github.com/allenai/scibert). The static features are computed per each neuron based on the mean, standard deviation, kurtosis, skewness, min and max. The project is currently ongoing.
It was test on linux.

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

====================  ===================  ===================  =====================  
Features              Unweighted Average   Satisfied Customer   No satisfied customer
                      Recall (mean/std)    Recall (mean/std)    Recall (mean/std) 
====================  ===================  ==================   =====================
NLP                          
Speech                                   
NLP+Speech                                                          
====================  ===================  ===================  ===================== 
