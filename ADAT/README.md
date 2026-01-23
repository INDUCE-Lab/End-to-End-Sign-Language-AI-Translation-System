# End-to-End-Sign-Language-AI-Translation-System
Computer Vision, Natural Language Processing, Transformer, Sign Language 

# ADAT: Novel Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation

This repository accompanies the paper:

> **Nada Shahin, Leila Ismail**  
> *ADAT: Novel Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation*  
> **Scientific Reports**, 2026.  

## 📬 Contact

Prof. Leila Ismail
Intelligent Distributed Computing and Systems (INDUCE) Lab
College of Information Technology, United Arab Emirates University
leila@uaeu.ac.ae


## 📜 Citation

If you use this work, please cite:
Shahin, Nada, and Leila Ismail. 2026. "ADAT: Novel Time-Series-Aware Adaptive Transformer Architecture for Sign Language Translation" Scientific Reports. https://doi.org/TBD

Overview:

Adaptive Transformer (ADAT) is designed to address core challenges in sign language machine translation (SLMT) by efficiently modeling fine-grained short-range motion and long-range temporal dependencies, while substantially reducing computational overhead.

ADAT consists of:


System Architecture:

Our proposed model, ADAT, is presented in Figure 1. It consists of the following modules:

<img width="1440" height="1040" alt="ADAT Model" src="xxx" />

1. An encoder that integrates convolutional feature extraction, LogSparse Self-Attention, and an adaptive gating mechanism, forming a unified framework for SLMT.
2. A classical Transformer decoder that generates the spoken language sequence

  
## Repository Structure
ADAT/
│
├── __init__.py
├── model_builder.py   # Full ADAT model construction
├── encoder.py         # ADAT encoder
├── decoder.py         # Transformer decoder block
├── layers.py          # LogSparseSelfAttention and PositionalEmbedding
├── data.py            # Synthetic dataset loader + expected real input format
├── train.py           # Minimal training script (using synthetic data)
├── run.py             # Forward-pass demo (sanity check)
├── config.yaml        # Model hyperparameters & training configuration
└── requirements.txt


## Demo & Code

Experimental Setup:

### Requirements
------------
* tensorflow>=2.10
* numpy>=1.20
* pyyaml>=6.0

### Quick start
-----------
1) Clone the repository

2) Install dependencies

pip install -r requirements.txt

3) Train the model

By default, trains ADAT end-to-end using randomly generated video/gloss/text sequences:
python train.py --config config.yaml

4) Forward-pass
   
python run.py

5) Training on real sign language datasets
   
To train on real datasets like RWTH-PHOENIX-Weather-2014T,  replace the synthetic loader inside:
data.py

with the following preprocessed tensors:
		* video_data: (N, T, 52, 65, 3) or (N, T, feature_dim)
		* gloss_indices: (N, max_g)  	# Padded integer sequences
		* text_indices: (N, max_t)  		# Padded integer sequences

Then simply run:

python train.py --config config.yaml

### Reproducibility
---------
This codebase provides everything needed to:
* Build the ADAT architecture
* Train a minimal working model (with synthetic data)
* Run forward inference
* Integrate ADAT into new datasets or pipelines

### Configuration notes
-------------------
* Confidence gate: CONF_THRESH on edge controls when a translation becomes the displayed caption.
* Warm-up: MIN_WINDOWS_BEFORE_DISPLAY skips the first windows for stabilization before translating sign language to text.
* Local inference: toggle with RUN_LOCAL_INFERENCE on edge. If enabled, the edge loads MODEL_PATH and detokenizes with medasl_bpe.model + special_ids.json.

### 📄 License

This project is released under the Creative Commons Attribution (CC BY 4.0) License, consistent with the published article.
