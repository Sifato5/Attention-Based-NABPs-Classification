# AttentionNABP: Attention-Based Deep Learning Architecture for Nucleic Acid-Binding Proteins Characterization

In this study, we developed a hierarchical and multi-class deep learning approach designed to effectively classify subtypes of NABPs with high precision.

# Files in /datasets:

/non-NABP_NABP: All test datasets of five independent run for non-NABP/NABP classification. <br>
/DBP_RBP: All test datasets of five independent run for DBP/RBP classification. <br>
/SSB_DSB: All test datasets of five independent run for SSB/DSB classification. <br>
/Multi-class: All test datasets of five independent run for Multi-class classification. <br>
# Files in /models:

/non-NABP_NABP: All pre-trained models of five independent run for predicting non-NABPs/NABPs.<br>
/DBP_RBP: All pre-trained models of five independent run for predicting DBPs/RBPs.<br>
/SSB_DSB: All pre-trained models of five independent run for predicting SSBs/DSBs.<br>
/Multi-class: All pre-trained models of five independent run for predicting non-NABPs/RBPs/DSBs/SSBs.<br>

# Dependencies:
python/3.9.18<br>
tensorflow/2.18.0<br>
numpy/1.26.4<br>
biopython/1.85<br>
scikit-learn/1.6.1<br>
transformers/4.57.3<br>
pytorch/2.5.1<br>

# Download Pre-trained ESM2 Model
To use the pre-trained ESM2 model in your setup, please download it from Hugging Face:<br>
[Download ESM2 (t33, 650M parameters, UR50D)](https://huggingface.co/facebook/esm2_t33_650M_UR50D)<br>
After downloading, run our pre-trained models using run_model.py<br>


# N.B.: Please use the .fasta file format to input sequences for prediction with our pre-trained models. These models will accurately classify the proteins and provide confidence score for each prediction.
