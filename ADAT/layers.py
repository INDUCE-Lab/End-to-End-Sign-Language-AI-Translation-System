'''
Title:        ADAT
Description:  ADAT (Adaptive Transformer) Toolkit for Sign Language Machine Translation
Licence:      GPL - http://www.gnu.org/copyleft/gpl.html

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots, included in ADAT directory please cite
the following paper:

https://doi.org/TBD">Nada Shahin and Leila Ismail, "ADAT: Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation",
Scientific Reports 2026

'''
import tensorflow as tf
from tensorflow.keras.layers import Lambda


class LogSparseSelfAttention(tf.keras.layers.Layer):
    """Log-sparse self-attention with indices {p, p-1, p-2, p-4, ...}."""

    def __init__(self, num_heads, embed_dim):
        # Call the constructor of the parent class, allowing the custom layer to behave like a Keras layer
        super(LogSparseSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # Define query and key dense layers
        self.query_dense = tf.keras.layers.Dense(embed_dim, use_bias=False)
        self.key_dense = tf.keras.layers.Dense(embed_dim, use_bias=False)

        # Define a multi-head attention layer that will be used to compute attention scores for the subset of patches we select
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    # Since Dense layers already handle weights, we just call the super build method.
    def build(self, input_shape):
        super(LogSparseSelfAttention, self).build(input_shape)

    # Method to generate the indices of the input that the current patch p can attend to, based on a logarithmic step size.
    # This method selects patches with exponentially increasing distances.
    # This method essentially generates a logarithmic subset of indices, such as {p, p-1, p-2, p-4, p-8, ..., 0}.
    def get_logarithmic_indices(self, p, seq_len):
        def condition(step_size, counter, indices):
            return tf.less_equal(step_size, p)

        def body(step_size, counter, indices):
            # Add the patch at the calculated distance (p - step_size) to the list of indices
            indices = indices.write(counter, p - step_size)
            # Double the step size (i.e., moves exponentially further back in time).
            return step_size * 2, counter + 1, indices

        # Set the initial step size and counter to 1
        step_size = 1
        counter = 1

        # Calculate the logarithmic subset of previous patches that the current patch `p` can attend to
        indices = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        # While loop continues as long as there are previous patches available
        step_size, counter, indices = tf.while_loop(condition, body, [step_size, counter, indices])

        # Write the current index p as the first element in the TensorArray
        indices = indices.write(0, p)

        # Convert the indices array to a TensorFlow tensor
        indices_tensor = indices.stack()

        return indices_tensor

    # The core method that performs the forward pass of the layer.
    # It computes the logarithmic self-attention using q and k
    def call(self, q, k):
        """
        inputs: (batch, seq_len, embed_dim)
        returns: (batch, seq_len, embed_dim/2) or embed_dim (depending on your design)
        """

        # Get the sequence length from q
        seq_len = tf.shape(q)[1]

        # Initialization to store the attention output for each patch
        attention_outputs = tf.TensorArray(tf.float16, size=seq_len)

        # Calculate the scaling factor √(d/2) where d is the embedding dimension.
        scaling_factor = tf.cast(tf.math.sqrt(tf.cast(self.embed_dim // 2, tf.float16)), dtype=tf.float16)

        def body(p, attention_outputs):
            log_indices = self.get_logarithmic_indices(p, seq_len)

            # Select the corresponding keys for the logarithmic indices
            k_subset = tf.gather(k, log_indices, axis=1)
            k_subset = tf.cast(k_subset, dtype=tf.float16)

            # Ensure that q and k_subset are the same data type
            q_p = q[:, p:p + 1, :]
            q_p = tf.cast(q_p, dtype=tf.float16)

            # Compute attention scores for the current patch using the selected k subset
            attention_scores = tf.matmul(q_p, k_subset, transpose_b=True) / scaling_factor

            # Softmax over the logarithmic subset of attention scores
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)

            # Apply the attention weights to the corresponding key subset to get the attention output
            attention_output = tf.matmul(attention_weights, k_subset)

            # Append the attention output for patch p to the list of outputs
            attention_outputs = attention_outputs.write(p, attention_output)

            return p + 1, attention_outputs

        # Use tf.while_loop to iterate over the patches
        _, attention_outputs = tf.while_loop(
            cond=lambda p, *_: p < seq_len,
            body=body,
            loop_vars=[0, attention_outputs]
        )

        # Stack and transpose to return the final output
        final_attention_output = attention_outputs.stack()
        final_attention_output = tf.squeeze(final_attention_output, axis=2)  # Remove the extra dimension
        final_attention_output = tf.transpose(final_attention_output, [1, 0, 2])

        return final_attention_output

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


