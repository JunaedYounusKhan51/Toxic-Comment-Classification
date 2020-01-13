Project: Toxic Comment Classification
========


Depnedencies:
============ 
      1. pandas
      2. keras
      3. sklearn
      4. pickle
      5. csv
      6. nltk



Manual for Traditional Machine Learning:
======================================
1. Go to "Traditinal_ML_Approaches" floder in the submitted codes.
2. Goto "Dataset_for_Traditional_ML" folder from the given link.
3. Copy each and every csv file from "Dataset_for_Traditional_ML" to "Traditinal_ML_Approaches".
4. Now notice that there are 5 python files in the "Traditinal_ML_Approaches" folder which are as below:
	a. emapth.py (KNN with empath generated features)
	b. lexical.py (LR and DecisionTree with lexical features)
	c. n_gram.py (Naive Bayes with bigram features)
	d. nb_lr.py (hybrid of NB and LR with n_gram features)
	e. one_vs_rest__svc.py (with n_gram feautres)
5. Run any of the 5 python files for evaluation.




Manual for Deep Learning:
=========================
Dataset: "Dataset_for_Deep_Learing/Pickles" folder contains the pre-processed datasets. Put this folder inside the project directory.



CNN directory: 1.CNN_Create_Model.py contains the CNN Model.
	       2.CNN_Balanced_Binary_1.py - RUN THIS FILE for binary toxic classification with CNN.

LSTM directory: 1.LSTM_Create_Model.py contains the LSTM Model.
	        2.LSTM_Balanced_Binary_1.py - RUN THIS FILE for binary toxic classification with LSTM.
  		3.LSTM_Balanced_Binary_Reversed_1.py - RUN THIS FILE for binary toxic classification with LSTM in reverse direction.

Bi-LSTM directory: 1.Bi_LSTM_Create_Model.py contains the Bi-LSTM Model.
	           2.Bi_LSTM_Balanced_Binary_1.py - RUN THIS FILE for binary toxic classification with Bi-LSTM.


DataPrep directory: Codes for dataset pre-processing.

Graph directory: Save graphs for train and validation accuracy, loss.

Models directory: Save the trained models as .h5 file.

Prediction directory: Save the predicted values.