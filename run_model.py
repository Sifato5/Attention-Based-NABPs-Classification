import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
import os

class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attention_output = self.multi_head_attention(inputs, inputs)
        attention_output = self.layer_norm(inputs + attention_output)
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config

custom_objects = {
    'SelfAttentionBlock': SelfAttentionBlock
}
class HierarchicalNABPPredictor:
    def __init__(self, model_paths, max_lengths):
        """
        Initialize hierarchical predictor with:
        - model_paths: List of paths to three .h5 models [nonNABP/NABP, RBP/DBP, DSB/SSB]
        - max_lengths: List of max sequence lengths for each model
        """
        with custom_object_scope(custom_objects):
            self.models = {
                'level1': load_model(model_paths[0]),  
                'level2': load_model(model_paths[1]),  
                'level3': load_model(model_paths[2])  
            }
        
        self.max_lengths = max_lengths
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_int = {aa: i for i, aa in enumerate(self.amino_acids)}
        
       
        self.label_encoders = {
            'level1': LabelEncoder().fit(['non-NABP', 'NABP']),
            'level2': LabelEncoder().fit(['RBP', 'DBP']),
            'level3': LabelEncoder().fit(['DSB', 'SSB'])
        }

    def one_hot_encode(self, sequence, max_length):
        """One-hot encode a single sequence"""
        one_hot = np.zeros((max_length, len(self.aa_to_int)))
        for i, aa in enumerate(sequence):
            if i >= max_length:
                break
            if aa in self.aa_to_int:
                one_hot[i, self.aa_to_int[aa]] = 1
        return one_hot

    def predict_sequence(self, sequence):
        """Make hierarchical prediction for a single sequence"""
        
        x_level1 = self.one_hot_encode(sequence, self.max_lengths[0])
        l1_probs = self.models['level1'].predict(np.array([x_level1]))[0]
        l1_pred = self.label_encoders['level1'].inverse_transform([np.argmax(l1_probs)])[0]
        
        if l1_pred == 'non-NABP':
            return {
                'prediction': 'non-NABP',
                'confidence': float(np.max(l1_probs)),
                'probabilities': {
                    'non-NABP': float(l1_probs[0]),
                    'NABP': float(l1_probs[1])
                },
                'level': 1
            }
        
        
        x_level2 = self.one_hot_encode(sequence, self.max_lengths[1])
        l2_probs = self.models['level2'].predict(np.array([x_level2]))[0]
        l2_pred = self.label_encoders['level2'].inverse_transform([np.argmax(l2_probs)])[0]
        
        if l2_pred == 'RBP':
            return {
                'prediction': 'RBP',
                'confidence': float(np.max(l2_probs)),
                'probabilities': {
                    'non-NABP': float(l1_probs[0]),
                    'NABP': float(l1_probs[1]),
                    'RBP': float(l2_probs[0]),
                    'DBP': float(l2_probs[1])
                },
                'level': 2
            }
        
       
        x_level3 = self.one_hot_encode(sequence, self.max_lengths[2])
        l3_probs = self.models['level3'].predict(np.array([x_level3]))[0]
        l3_pred = self.label_encoders['level3'].inverse_transform([np.argmax(l3_probs)])[0]
        
        return {
            'prediction': f'DBP-{l3_pred}',
            'confidence': float(np.max(l2_probs) * np.max(l3_probs)),
            'probabilities': {
                'non-NABP': float(l1_probs[0]),
                'NABP': float(l1_probs[1]),
                'RBP': float(l2_probs[0]),
                'DBP': float(l2_probs[1]),
                'DSB': float(l3_probs[0]),
                'SSB': float(l3_probs[1])
            },
            'level': 3
        }

    def predict_fasta(self, fasta_path):
        """Predict classes for all sequences in a FASTA file"""
        results = {}
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq)
            results[record.id] = self.predict_sequence(seq)
        return results

if __name__ == "__main__":
    
    MODEL_PATHS = [
        "non-NABP_NABP.h5", 
        "DBP_RBP.h5", 
        "SSB_DSB.h5"  
    ]
    
   
    MAX_LENGTHS = [
        1499, 
        1499,   
        1499    
    ]
    
    predictor = HierarchicalNABPPredictor(MODEL_PATHS, MAX_LENGTHS)
    predictions = predictor.predict_fasta("rcsb_pdb_6RVV.fasta") # Please input your desired sequences 
    for seq_id, pred in predictions.items():
        print(f"\nSequence: {seq_id}")
        print(f"Prediction: {pred['prediction']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        
       