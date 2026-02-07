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

from encoder import build_transformer_encoder
from decoder import TransformerDecoder
from layers import PositionalEmbedding



def create_transformer(config):
    """
    Build a sign-to-gloss-to-text model using the Transformer encoder.
    Config is expected to have:
        max_video_length, embed_dim, encoder_layers, decoder_layers,
        hidden_units, num_heads, max_gloss_length, max_text_length
    """
    with tf.device('/GPU:0'):
        # Encoder: sign video input
        encoder_inputs = tf.keras.Input(
            shape=(config.max_video_length, 52, 65, 3),
            dtype="float16",
            name="encoder_inputs")

        # Process video input through per-frame Conv2D + pooling + projection
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        )(encoder_inputs)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.MaxPooling2D((2, 2))
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten()
        )(x)
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(config.embed_dim, activation='relu')
        )(x) # (batch, T, embed_dim)

        encoder = build_transformer_encoder(
            num_layers=config.encoder_layers,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.hidden_units,
            dropout_rate=getattr(config, "dropout_rate", 0.0),
        )
        encoder_outputs = encoder(x)

        # Gloss head
        pooled = tf.keras.layers.GlobalAveragePooling1D()(encoder_outputs)
        pooled = tf.keras.layers.Dense(config.embed_dim, activation='relu')(pooled)
        encoder_outputs_gloss = tf.keras.layers.RepeatVector(config.max_gloss_length)(pooled)

        # Final output layer for gloss generation
        gloss_output = tf.keras.layers.Dense(
            config.gloss_vocab_size, activation="softmax", dtype=tf.float32, name="gloss_output"
        )(encoder_outputs_gloss)

        # Predicted gloss tokens from the gloss_output (to be passed into cross-attention in the decoder)
        predicted_gloss_tokens = tf.keras.layers.Lambda(
            lambda x: tf.argmax(x, axis=-1), name="predicted_gloss_tokens"
        )(gloss_output)

        gloss_embedding = PositionalEmbedding(
            config.max_gloss_length, config.gloss_vocab_size, config.embed_dim
        )(predicted_gloss_tokens)

        # Text decoder inputs
        decoder_inputs_text = tf.keras.Input(
            shape=(config.max_text_length,), dtype="int64", name="decoder_inputs_text"
        )
        text_embedding = PositionalEmbedding(
            config.max_text_length, config.text_vocab_size, config.embed_dim
        )(decoder_inputs_text)

        # Stack of decoder blocks
        y = text_embedding
        for i in range(config.decoder_layers):
            y = TransformerDecoder(
                embed_dim=config.embed_dim,
                latent_dim=config.hidden_units,
                num_heads=config.num_heads,
                rate=config.dropout_rate,
                name=f"text_decoder_block_{i}",
            )(y, gloss_embedding)

        # Final output layer for text generation
        text_output = tf.keras.layers.Dense(
            config.text_vocab_size, activation="softmax", dtype=tf.float32,name="text_output"
        )(y)

        # Create the full model
        transformer = tf.keras.Model(
            [encoder_inputs, decoder_inputs_text],
            [gloss_output, text_output],
            name="Transformer")

    return transformer
