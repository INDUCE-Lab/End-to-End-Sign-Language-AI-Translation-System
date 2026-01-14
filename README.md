# End-to-End-Sign-Language-AI-Translation-System
Computer Vision, Edge Computing, Cloud Computing, Blockchain 

# Towards Trustworthy Sign Language Translation: A Privacy-Preserving Edge–Cloud–Blockchain System

This repository accompanies the paper:

> **Nada Shahin, Leila Ismail**  
> *Towards Trustworthy Sign Language Translation System:  
> A Privacy-Preserving Edge–Cloud–Blockchain Approach*  
> **Mathematics**, 2025, 13, 3759.  
> DOI: 10.3390/math13233759

## 📬 Contact

Prof. Leila Ismail
Intelligent Distributed Computing and Systems (INDUCE) Lab
College of Information Technology, United Arab Emirates University
leila@uaeu.ac.ae


## 📜 Citation

If you use this work, please cite:
hahin, Nada, and Leila Ismail. 2025. "Towards Trustworthy Sign Language Translation System: A Privacy-Preserving Edge–Cloud–Blockchain Approach" Mathematics 13, no. 23: 3759. https://doi.org/10.3390/math13233759

Overview:

This work envisions a new generation of trustworthy sign language translation systems that go beyond translation accuracy to address privacy, accountability, and real-world deployment. In response to the global shortage of certified sign language interpreters and the growing need for inclusive assistive technologies, the paper introduces a privacy-preserving, consent-aware SLMT architecture built on the integration of edge computing, cloud intelligence, and blockchain governance. By operating on abstract keypoint representations rather than raw video and enforcing explicit, auditable user consent, the system enables real-time communication while safeguarding user rights and regulatory compliance 

At its core, the proposed system demonstrates that responsible AI and high performance are not competing goals. Through a comparative evaluation of Transformer-based models on large-scale and medical-domain datasets, the study shows that lightweight adaptive architectures can deliver accurate translations with substantially lower latency and computational cost in distributed environments. By embedding consent management and auditability directly into the AI pipeline, this work establishes a blueprint for ethically grounded, scalable assistive AI, with relevance extending beyond sign language translation to privacy-sensitive applications in healthcare and other biometric domains.

## ✨ Key Contributions

- 🔹 **End-to-end edge–cloud–blockchain architecture** for trustworthy SLMT  
- 🔹 **Consent-aware design** supporting both application-level and system-level privacy  
- 🔹 **Comparative evaluation** of:
  - Encoder–Decoder Transformer  
  - Adaptive Transformer (ADAT)  
- 🔹 **New medical-domain dataset (MedASL)** for sign-to-text translation  
- 🔹 **Comprehensive runtime analysis**, including:
  - Training time  
  - Inference latency  
  - Edge–cloud communication  
  - End-to-end system delay  

System Architecture:

Our proposed end-to-end edge-cloud-blockchain system for SLMT is presented in Figure 1. It consists of the following modules:

<img width="1440" height="1040" alt="End-to-End Model" src="https://github.com/user-attachments/assets/e80b6fc9-0473-42f7-8301-0b27beb889e7" />

1. **Sign Language Recognition Module**

Captures sign videos via camera input for keypoint extraction and processing.

2. **AI-Enabled Translation Application**  
   Acts as a gateway for user interaction and consent management.

3. **Edge Computing Layer**  
   - Keypoint extraction (MediaPipe)
   - Preprocessing and real-time inference
   - Reduced latency and improved privacy

4. **Cloud Computing Layer**  
   - Model training and retraining
   - Dataset storage (with consent)
   - Deployment of updated models

5. **Blockchain Layer**  
   - Immutable logging of:
     - User consent receipts
     - Policy versions
     - System certificates
     - Audit trails
   
   It ensures transparency, integrity, and regulatory compliance.


## 🤖 Models Implemented

### 1. Encoder–Decoder Transformer
- Baseline and widely adopted architecture for SLMT
- Captures long-range spatiotemporal dependencies
- Higher computational cost due to quadratic self-attention

### 2. Adaptive Transformer (ADAT)
- Lightweight and efficient variant
- Key features:
  - LogSparse Self-Attention (O(n log n))
  - Adaptive gating for short- and long-range dependencies
- Demonstrates:
  - Faster convergence
  - Reduced model size
  - Lower inference and communication latency


## 📊 Datasets

### 🔹 RWTH-PHOENIX-Weather-2014T (PHOENIX14T)
- German Sign Language (DGS)
- Weather domain
- Large-scale, multi-signer benchmark dataset

### 🔹 MedASL (New Dataset)
- American Sign Language (ASL)
- Medical and healthcare conversations
- Designed to reflect real-world assistive scenarios

Dataset characteristics, preprocessing pipelines, and splits are fully described in the paper.

## ⚙️ Preprocessing Pipeline

- Keypoint extraction using **MediaPipe**
  - Hands, face, pose, and iris landmarks
- Normalization, rescaling, and padding
- Sliding-window segmentation for inference
- Tokenization and subword modeling for text output

This design improves **privacy**, **efficiency**, and **robustness** compared to raw RGB-based approaches.

## Demo & Code

Sign Language Machine Translation Edge/Cloud Demo with Transformer baseline

End-to-end edge-cloud pipeline for sign language translation including keypoint capture via camera and model inference on edge + communication with cloud.

Experimental Setup:

### Requirements
------------
* Python � 3.10
* Edge: opencv-python, mediapipe, numpy, torch, requests, sentencepiece
* Cloud: flask, numpy

### Quick start
-----------
1) Prepare the cloud service
	1. Install the files under the model directory
		* transformer.pth: best Transformer model checkpoint
		* special_ids.json: token IDs and sizes
		* medasl_bpe.model: SentencePiece-BPE model for text decoding

	2. Install cloud dependencies and run:
		pip install -r Cloud_requirements.txt
		# run server
		python -m Cloud
		# or
		python Cloud/.py

2) Configure the edge client
	1. Open Edge.py and set:
		* CLOUD_UPLOAD_URL = "http://<CLOUD_HOST>:<PORT>/upload_keypoints"
		* CLOUD_MODEL_URL = "http://<CLOUD_HOST>:<PORT>/get_model"
		* Adjust SEQ_LEN, MAX_SAMPLES, and paths if needed.

	2. Install edge dependencies and run:
		pip install -r Edge_requirements.txt
		# run edge
		python -m edge
		# or
		python edge.py

	3. Controls:
		* When running the edge python script, a window will pop up showing landmarks and a caption (�translating�� appears until translation confidence/threshold is met).
		* Press u to force an upload + model fetch cycle.
		* Press q to quit.

### Data flow
---------
1. Edge calls /get_model, /get_specials, /get_spm to refresh assets (periodically).
2. Camera captures frames and the edge extracts keypoints using MediaPipe and concatenates keypoints producing a vector of 1662 floats per frame.
3. Sliding window of length SEQ_LEN forms an array [SEQ_LEN, 1662].
4. When sliding window is full, the window is saved locally (.npy).
5. The edge preprocesses the npy file.
6. Inference is run through the model to produce translations.
7. When the local cache reaches MAX_SAMPLES, the edge packs all windows into samples.npz and transmits them them to the cloud, along with the translations.

### Configuration notes
-------------------
* Confidence gate: CONF_THRESH on edge controls when a translation becomes the displayed caption.
* Warm-up: MIN_WINDOWS_BEFORE_DISPLAY skips the first windows for stabilization before translating sign language to text.
* Local inference: toggle with RUN_LOCAL_INFERENCE on edge. If enabled, the edge loads MODEL_PATH and detokenizes with medasl_bpe.model + special_ids.json.

### Deployment
----------
* Set the edge CLOUD_* URLs to the cloud host/IP and open the cloud port in the firewall.
* Run edge from the repo root:
python -m edge
* Run cloud similarly:
python -m cloud

### 📄 License

This project is released under the Creative Commons Attribution (CC BY 4.0) License, consistent with the published article.
