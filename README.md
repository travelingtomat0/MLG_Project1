# Predicting Gene Expression Based on Histone Modifications

This repository was generated over the course of the ETH course "Machine Learning for Genomics". Goal of this repository is to reconstruct / predict gene expression of cells based on their histone modification marks.

While different methods were tested (eg. CNN's), we found that if evaluating the correlation of predicted and actual results, a simple (Lasso) Regression gets the job done.

If changing the evaluation metric to include for instance the MSE, a deep learning approach will most likely be the go-to method.  
