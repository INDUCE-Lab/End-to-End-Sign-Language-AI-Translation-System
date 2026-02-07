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


import argparse
import yaml
import tensorflow as tf

from model_builder import create_adat
from data import build_dataset

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main(config_path):
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    config = Config(**cfg_dict)

    model = create_adat(
        config
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            "gloss_output": "sparse_categorical_crossentropy",
            "text_output": "sparse_categorical_crossentropy",
        },
        metrics={"text_output": "accuracy"},
    )

    train_ds = build_dataset(config, num_samples=32).batch(config.batch_size)

    model.fit(
        train_ds,
        epochs=3,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
