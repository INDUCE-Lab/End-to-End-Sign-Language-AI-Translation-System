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
from layers import LogSparseSelfAttention


class ADATEncoder(tf.keras.layers.Layer):
    """
    ADAT encoder block:
    - Split channels into two halves
    - First half -> Conv1D for local patterns
    - Second half -> log-sparse self-attention + GAP
    - Gate between attention output and GAP
    - Residual + layernorm
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate):
        super(ADATEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Conv for half the input
        self.conv_half = tf.keras.layers.Conv1D(embed_dim // 2, kernel_size=3, padding="same")

        # Define query and key dense layers
        self.query_dense = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.key_dense = tf.keras.layers.Dense(embed_dim, use_bias=False)

        # Custom self-attention layer
        self.att = LogSparseSelfAttention(embed_dim=embed_dim // 2, num_heads=num_heads)

        # Feed-forward network (FFN)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.attn_dense = tf.keras.layers.Dense(embed_dim // 2)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        # Gate network
        self.gating_dense = tf.keras.layers.Dense(1, activation='sigmoid')
        self.output_dense = tf.keras.layers.Dense(embed_dim)
        self.projection_layer = tf.keras.layers.Dense(embed_dim)

        # Global Average Pooling for gap_output
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling1D()

        self.dilated_conv1 = tf.keras.layers.Conv1D(embed_dim, kernel_size=3, dilation_rate=4,
                                                    padding="causal")

        self.dilated_conv2 = tf.keras.layers.Conv1D(embed_dim, kernel_size=3, dilation_rate=2,
                                                    padding="causal")

    def build(self, input_shape):
        super(ADATEncoder, self).build(input_shape)

    def call(self, inputs, training=False):
        # inputs: (batch, seq_len, embed_dim)

        # Split the input into two halves along the last dimension
        first_half, second_half = tf.split(inputs, 2, axis=-1)

        # First half through convolution
        # The convolution extracts local patterns from the input
        # The output has the same length as the input but half the embedding dimension
        conv_half_output = self.conv_half(first_half)

        q = self.query_dense(second_half)
        k = self.key_dense(second_half)
        v = second_half

        # Self-attention
        attn_output = self.att(q,k)
        attn_output = self.attn_dense(attn_output)

        # Global average pooling
        gap_output = self.global_avg_pool(v)  # Shape: (batch_size, embed_dim)

        # Expand gap_output along the second axis to match the rank of attention_output
        gap_output = gap_output[:, tf.newaxis, :]

        # Fix shape mismatch by tiling gap_output to match attention output along the sequence dimension
        gap_output = tf.tile(gap_output, [1, tf.shape(attn_output)[1], 1])
        gap_output = tf.cast(gap_output, dtype=tf.float32)

        # Gating mechanism to decide whether to multiply V by LSSA or GAP
        # The output of LSSA and Gap are concatenated along the last axis
        # The concatenated result is passed through the gating mechanism
        gate_value = self.gating_dense(tf.concat([attn_output, tf.cast(gap_output, dtype=tf.float32)], axis=-1))

        # Apply gating mechanism by multiplying LSSA output by the gate value
        # And multiply the GAP output by the inverse of the gate value
        # These two are added together to produce the final output for the second half of the CSPBlock
        # This is based on the gating mechanism's decision
        gated_output = tf.cast(gate_value, tf.float32) * tf.cast(attn_output, tf.float32) + tf.cast(1 - gate_value,tf.float32) * tf.cast(gap_output, tf.float32)

        # CSP block output by concatenating both halves along the last dimension
        # This will result in dimension 1024 which is double of our embed_dim
        csp_output = tf.concat([conv_half_output, gated_output], axis=-1)

        # Reduce dimensionality from 1024 back to 512
        csp_output = self.output_dense(csp_output)

        # First residual connection
        out1 = self.layernorm1(inputs + csp_output)

        return out1

def build_adat_encoder(num_layers, embed_dim, num_heads, ff_dim, dropout_rate):
    """Return a function that applies N encoder layers in sequence."""
    encoder_layers = [
        ADATEncoder(embed_dim, num_heads, ff_dim, dropout_rate)
        for _ in range(num_layers)
    ]

    def apply(x, training=False):
        for layer in encoder_layers:
            x = layer(x, training=training)
        return x

    return apply


