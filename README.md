# End-to-End-Sign-Language-AI-Translation-System
Computer Vision 


Sign Language Machine Translation Edge�Cloud Demo with Transformer baseline

End-to-end edge-cloud pipeline for sign language translation including keypoint capture via camera and model inference on edge + communication with cloud.


Requirements
------------
* Python � 3.10
* Edge: opencv-python, mediapipe, numpy, torch, requests, sentencepiece
* Cloud: flask, numpy

Quick start
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

Data flow
---------
1. Edge calls /get_model, /get_specials, /get_spm to refresh assets (periodically).
2. Camera captures frames and the edge extracts keypoints using MediaPipe and concatenates keypoints producing a vector of 1662 floats per frame.
3. Sliding window of length SEQ_LEN forms an array [SEQ_LEN, 1662].
4. When sliding window is full, the window is saved locally (.npy).
5. The edge preprocesses the npy file.
6. Inference is run through the model to produce translations.
7. When the local cache reaches MAX_SAMPLES, the edge packs all windows into samples.npz and transmits them them to the cloud, along with the translations.

Configuration notes
-------------------
* Confidence gate: CONF_THRESH on edge controls when a translation becomes the displayed caption.
* Warm-up: MIN_WINDOWS_BEFORE_DISPLAY skips the first windows for stabilization before translating sign language to text.
* Local inference: toggle with RUN_LOCAL_INFERENCE on edge. If enabled, the edge loads MODEL_PATH and detokenizes with medasl_bpe.model + special_ids.json.

Deployment
----------
* Set the edge CLOUD_* URLs to the cloud host/IP and open the cloud port in the firewall.
* Run edge from the repo root:
python -m edge
* Run cloud similarly:
python -m cloud