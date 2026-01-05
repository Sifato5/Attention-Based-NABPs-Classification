import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import custom_object_scope
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import os

class SelfAttentionBlock(tf.keras.layers.Layer):
    
    def __init__(self, num_heads, key_dim, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
       
        attention_output = self.multi_head_attention(inputs, inputs)
        attention_output = self.layer_norm(inputs + attention_output)
        return attention_output

    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config

class LogitAdjustedLoss(tf.keras.losses.Loss):

    def __init__(self, class_priors, tau=1.0, reduction='sum_over_batch_size', name="logit_adjusted_loss"):
        super().__init__(reduction=reduction, name=name)
        self.class_priors = tf.constant(class_priors, dtype=tf.float32)
        self.tau = tau
        
        self.logit_adjustments = self.tau * tf.math.log(self.class_priors + 1e-8)
    
    def call(self, y_true, y_pred):
 
        adjustments = tf.reduce_sum(y_true * self.logit_adjustments, axis=-1, keepdims=True)
        adjusted_logits = y_pred - adjustments
        
        y_true = tf.cast(y_true, dtype=adjusted_logits.dtype)
        loss = tf.nn.softmax_cross_entropy_with_logits(y_true, adjusted_logits)
        
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'class_priors': self.class_priors.numpy().tolist(),
            'tau': self.tau
        })
        return config

def logit_adjusted_loss(class_priors, tau=1.0):
  
    class_priors = tf.constant(class_priors, dtype=tf.float32)
    adjustments = tau * tf.math.log(class_priors + 1e-8)
    
    def loss_fn(y_true, y_pred):
        
        batch_adjustments = tf.reduce_sum(y_true * adjustments, axis=-1, keepdims=True)
        adjusted_logits = y_pred - batch_adjustments
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true, adjusted_logits))
    
    return loss_fn

class ESM2Embedding:
   
    def __init__(self, model_path="/home/esm2_t33_650M_UR50D"):
        print(f"Loading ESM2 model from: {model_path}")
        
        try:
          
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.eval()
            
            print(f"ESM2 model loaded successfully")
            
        except Exception as e:
            print(f"Error loading ESM2 model: {e}")
            raise
    
    def get_single_embedding(self, sequence, pooling="mean", max_length=1024):
        """Get embedding for a single sequence"""
        try:
            
            inputs = self.tokenizer(
                sequence, 
                return_tensors="pt", 
                max_length=max_length, 
                truncation=True, 
                padding='max_length'
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                
                if pooling == "mean":
                    embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
                elif pooling == "max":
                    embedding = last_hidden_state.max(dim=1).values.squeeze().numpy()
                elif pooling == "cls":
                    embedding = last_hidden_state[:, 0, :].squeeze().numpy()
                else:
                    embedding = last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return embedding.reshape(1, -1)
            
        except Exception as e:
            print(f"Error getting embedding: {e}")
            
            return np.zeros((1, 1280))


class HierarchicalNABPPredictor:
    def __init__(self, model_paths, esm2_model_path):
       
        custom_objects = {
            'SelfAttentionBlock': SelfAttentionBlock,
            'LogitAdjustedLoss': LogitAdjustedLoss,
            'logit_adjusted_loss': logit_adjusted_loss,
        }
        
        self.models = {}
        for i, path in enumerate(model_paths):
            level = f'level{i+1}'
            print(f"\nLoading {level} model from: {path}")
            
            try:
                with custom_object_scope(custom_objects):
                    self.models[level] = load_model(path, compile=False)
                print(f" {level} model loaded successfully")
                
            except Exception as e:
                print(f"Error loading {level}: {e}")
                raise
        
        print("\nInitializing ESM2 embedder...")
        self.esm2_embedder = ESM2Embedding(esm2_model_path)
        
        self.label_encoders = {
            'level1': LabelEncoder().fit(['non-NABP', 'NABP']),
            'level2': LabelEncoder().fit(['RBP', 'DBP']),
            'level3': LabelEncoder().fit(['DSB', 'SSB'])
        }
        
        print("\n" + "=" * 70)
        print("PREDICTOR INITIALIZATION COMPLETE")
        print("=" * 70)
    
    def get_esm2_embedding(self, sequence):
        """Get ESM2 embedding for a single sequence"""
        return self.esm2_embedder.get_single_embedding(sequence, pooling="mean")
    
    def predict_sequence(self, sequence):
        """Make hierarchical prediction for a single sequence"""
        
        x_level1 = self.get_esm2_embedding(sequence)
        
        l1_logits = self.models['level1'].predict(x_level1, verbose=0)[0]
        l1_probs = tf.nn.softmax(l1_logits).numpy()
        l1_pred = self.label_encoders['level1'].inverse_transform([np.argmax(l1_probs)])[0]
        
        if l1_pred == 'non-NABP':
            return {
                'prediction': 'non-NABP',
                'confidence': float(np.max(l1_probs)),
                'probabilities': {
                    'non-NABP': float(l1_probs[0]),
                    'NABP': float(l1_probs[1])
                },
                'level': 1,
                'logits': l1_logits.tolist()
            }
        
        x_level2 = self.get_esm2_embedding(sequence)
        
        l2_logits = self.models['level2'].predict(x_level2, verbose=0)[0]
        l2_probs = tf.nn.softmax(l2_logits).numpy()
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
                'level': 2,
                'logits': {
                    'level1': l1_logits.tolist(),
                    'level2': l2_logits.tolist()
                }
            }
        
        x_level3 = self.get_esm2_embedding(sequence)
        
        l3_logits = self.models['level3'].predict(x_level3, verbose=0)[0]
        l3_probs = tf.nn.softmax(l3_logits).numpy()
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
            'level': 3,
            'logits': {
                'level1': l1_logits.tolist(),
                'level2': l2_logits.tolist(),
                'level3': l3_logits.tolist()
            }
        }
    
    def predict_fasta(self, fasta_path, verbose=True):
        """Predict classes for all sequences in a FASTA file"""
        results = {}
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = record.id
            sequence = str(record.seq)
            
            if verbose:
                print(f"\nPredicting sequence: {seq_id}")
                print(f"Sequence length: {len(sequence)} amino acids")
            
            try:
                result = self.predict_sequence(sequence)
                results[seq_id] = result
                
                if verbose:
                    print(f"  Prediction: {result['prediction']}")
                    print(f"  Confidence: {result['confidence']:.4f}")
                    print(f"  Level: {result['level']}")
                    
            except Exception as e:
                print(f" Error predicting {seq_id}: {e}")
                results[seq_id] = {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def predict_single(self, sequence, sequence_id="sequence"):
        """Predict class for a single sequence string"""
        print(f"\nPredicting sequence: {sequence_id}")
        print(f"Sequence length: {len(sequence)} amino acids")
        
        try:
            result = self.predict_sequence(sequence)
            
            print(f"\n{'='*60}")
            print(f"PREDICTION RESULTS")
            print(f"{'='*60}")
            print(f"Sequence ID: {sequence_id}")
            print(f"Length: {len(sequence)} amino acids")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Hierarchy Level: {result['level']}")
            
            print(f"\nProbabilities:")
            for class_name, prob in result['probabilities'].items():
                if prob > 0.001:  
                    print(f"  {class_name}: {prob:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Error predicting sequence: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    

if __name__ == "__main__":
    # Use actual model path
    MODEL_PATHS = [
        "/home/model_run_4.h5",  # Level 1: non-NABP vs NABP
        "/home/model_run_3.h5",   # Level 2: DBP vs RBP  
        "/home/model_run_1.h5"    # Level 3: SSB vs DSB
    ]
    
    ESM2_MODEL_PATH = "/home/esm2_t33_650M_UR50D"   # Use your actual ESM2 model path
    
    print("\nInitializing predictor...")
    predictor = HierarchicalNABPPredictor(MODEL_PATHS, ESM2_MODEL_PATH)
    
    fasta_path = "/home/rcsb_pdb_9C5T.fasta"  # Use your desired FASTA file and actual FASTA file path
    
    if os.path.exists(fasta_path):
        print(f"\nAnalyzing FASTA file: {fasta_path}")
        results = predictor.predict_fasta(fasta_path, verbose=True)
        
        print(f"\n{'='*80}")
        print("PREDICTION SUMMARY")
        print(f"{'='*80}")
        
        prediction_counts = {}
        for seq_id, result in results.items():
            pred = result['prediction']
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        print(f"Total sequences analyzed: {len(results)}")
        print("\nPrediction distribution:")
        for pred, count in prediction_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {pred}: {count} sequences ({percentage:.1f}%)")
    
    else:
        print(f"FASTA file not found: {fasta_path}")
        print("\nTo analyze your own FASTA file:")
        print(f"1. Create a FASTA file with your sequences")
        print(f"2. Update the 'fasta_path' variable in the code")
        print(f"3. Run the script again")
    
    print("\n" + "=" * 80)
    print("PREDICTION COMPLETE!")
    print("=" * 80)