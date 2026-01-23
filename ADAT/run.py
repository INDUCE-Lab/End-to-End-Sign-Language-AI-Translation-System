'''
Title:        ADAT
Description:  ADAT (Adaptive Transformer) Toolkit for Sign Language Machine Translation
Licence:      GPL - http://www.gnu.org/copyleft/gpl.html

If you are using any ideas, algorithms, packages, codes, datasets, workload, results, and plots, included in ADAT directory please cite
the following paper:

https://doi.org/TBD">Nada Shahin and Leila Ismail, "ADAT: Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation",
Scientific Reports 2026

'''
import yaml
import numpy as np
import tensorflow as tf

from model_builder import create_adat

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main():
    with open("adat/config.yaml", "r") as f:
        cfg_dict = yaml.safe_load(f)
    config = Config(**cfg_dict)

    model = create_adat(
        config,
        gloss_vocab_size=config.gloss_vocab_size,
        text_vocab_size=config.text_vocab_size,
    )

    dummy_video = np.random.randn(
        2, config.max_video_length, 52, 65, 3
    ).astype("float32")
    dummy_text_in = np.random.randint(
        1, config.text_vocab_size, size=(2, config.max_text_length)
    ).astype("int32")

    gloss_logits, text_logits = model(
        {"encoder_inputs": dummy_video, "decoder_inputs_text": dummy_text_in},
        training=False,
    )

    print("Gloss logits shape:", gloss_logits.shape)
    print("Text logits shape:", text_logits.shape)

if __name__ == "__main__":
    main()
