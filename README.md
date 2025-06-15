# Attention-Based Deep Learning Architecture for Comprehensive Nucleic Acid-Binding Proteins Characterization

In this study, we developed a novel hierarchical and multi-class deep learning approach designed to effectively classify subtypes of NABPs (non-NABPs, RBPs, DSBs, and SSBs) with high precision

# Files in /models:

non-NABP_NABP.h5: Pre-trained model for predicting Non-NABPs/NABPs.<br>
DBP_RBP.h5: Pre-trained model for predicting DBPs/RBPs.<br>
SSB_DSB.h5: Pre-trained model for predicting SSBs/DSBs.<br>
Multi-class.h5: Pre-trained model for predicting Non-NABPs/RBPs/DSBs/SSBs.<br>

# Dependencies:
Python/3.12.7<br>
TensorFlow/2.18.0<br>

# Please use the .fasta file format to input sequences for prediction with our pre-trained models. These models will accurately classify the proteins and provide confidence scores for each prediction.
