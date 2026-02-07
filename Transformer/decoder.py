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


class TransformerDecoder(tf.keras.layers.Layer):
    """
    Standard Transformer decoder block:
    - Causal self-attention
    - Cross-attention to encoder outputs
    - Feed-forward network
    """

    def __init__(self, embed_dim, latent_dim, num_heads, rate, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential([
            tf.keras.layers.Dense(latent_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.supports_masking = True

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, encoder_outputs, training=False, mask=None):
        # inputs: (batch, tgt_len, embed_dim)
        # encoder_outputs: (batch, src_len, embed_dim)

        # Self-attention (on gloss or text)
        causal_mask = self.get_causal_attention_mask(inputs)

        # Ensure consistent dimensions between query, key, and value tensors
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask, training=training
        )
        attention_output_1 = self.dropout1(attention_output_1, training=training)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        # Cross-attention (between gloss/text and encoder outputs)
        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs, training=training
        )
        attention_output_2 = self.dropout2(attention_output_2, training=training)
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        # Feed-forward network
        proj_output = self.dense_proj(out_2, training=training)
        proj_output = self.dropout3(proj_output, training=training)
        proj_output = self.layernorm_3(out_2 + proj_output)

        return proj_output

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        seq_len = input_shape[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, 1, seq_len, seq_len))
        return mask

