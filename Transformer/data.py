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
import numpy as np

def build_dataset(config, num_samples=16):
    """
    Create a tiny synthetic dataset for demos.
    Video: random (batch, T, 52, 65, 3)
    Gloss/Text: random integer sequences.

    real expected format for preprocessed sign-language datasets:

    - video_data: (N, T, H, W, C) or (N, T, F) features
    - gloss_indices: (N, Lg) integer ids with pad token (0)
    - text_indices: (N, Lt) integer ids with pad token (0)
    """
    video = np.random.randn(num_samples, config.max_video_length, 52, 65, 3).astype("float32")
    gloss = np.random.randint(
        low=1, high=config.gloss_vocab_size,
        size=(num_samples, config.max_gloss_length),
        dtype="int32",
    )
    text = np.random.randint(
        low=1, high=config.text_vocab_size,
        size=(num_samples, config.max_text_length),
        dtype="int32",
    )

    def generator():
        for v, g, t in zip(video, gloss, text):
            yield {
                "encoder_inputs": v,
                "decoder_inputs_text": t,
            }, {
                "gloss_output": g,
                "text_output": t,
            }

    output_signature = (
        {
            "encoder_inputs": tf.TensorSpec(
                shape=(config.max_video_length, 52, 65, 3), dtype=tf.float32
            ),
            "decoder_inputs_text": tf.TensorSpec(
                shape=(config.max_text_length,), dtype=tf.int32
            ),
        },
        {
            "gloss_output": tf.TensorSpec(
                shape=(config.max_gloss_length,), dtype=tf.int32
            ),
            "text_output": tf.TensorSpec(
                shape=(config.max_text_length,), dtype=tf.int32
            ),
        },
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset
