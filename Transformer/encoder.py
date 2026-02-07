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

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Standard Transformer encoder layer:
    - Multi-head self-attention (full attention)
    - Residual + LayerNorm
    - Position-wise feed-forward network (FFN)
    - Residual + LayerNorm
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Full multi-head self-attention (standard Transformer)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=rate,
        )

        # Position-wise feed-forward network
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(ff_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training=False, attention_mask=None):
        """
        inputs: (batch, seq_len, embed_dim)
        attention_mask (optional): broadcastable to (batch, seq_len, seq_len)
        """

        # Self-attention: Q=K=V=inputs
        attn_output = self.mha(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=attention_mask,
            training=training,
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # FFN
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


def build_transformer_encoder(num_layers, embed_dim, num_heads, ff_dim, dropout_rate):
    """Return a function that applies N standard Transformer encoder layers in sequence."""
    encoder_layers = [
        TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
        for _ in range(num_layers)
    ]

    def apply(x, training=False, attention_mask=None):
        for layer in encoder_layers:
            x = layer(x, training=training, attention_mask=attention_mask)
        return x

    return apply
