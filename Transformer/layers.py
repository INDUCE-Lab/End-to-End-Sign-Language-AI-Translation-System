'''
Title:        End-to-End-Sign-Language-AI-Translation-System
Description:  End-to-end system for sign language translation using AI models across Edge and Cloud.
Licence:      GNU GENERAL PUBLIC LICENSE

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots included in this project, please cite
the following paper:

https://doi.org/10.3390/math13233759">Nada Shahin and Leila Ismail, "Towards Trustworthy Sign Language Translation System: 
A Privacy-Preserving Edge–Cloud–Blockchain Approach",
Mathematics 2025

'''
import tensorflow as tf
from tensorflow.keras.layers import Lambda


class PositionalEmbedding(tf.keras.layers.Layer):
    """Token + learned positional embeddings."""

    def __init__(self, max_gloss_length, gloss_vocab_size, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.token_embeddings = tf.keras.layers.Embedding(input_dim=gloss_vocab_size, output_dim=embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=max_gloss_length, output_dim=embed_dim)
        self.max_gloss_length = max_gloss_length
        self.gloss_vocab_size = gloss_vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions


