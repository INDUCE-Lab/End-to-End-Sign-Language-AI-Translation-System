# MedASL2000

MedASL2000 is an ASL corpus that focuses on medical communication, with gloss and text annotations. It is designed to support researchers and industry professionals in advancing Sign Language Machine Translation systems. By incorporating medical terminology and advanced data acquisition, such as the Intel RealSense camera, MedASL2000 enables the development of accurate, context-aware models that reflect real-world healthcare scenarios. The dataset consists of 2000 medical and healthcare-related statements, generated via prompt engineering using ChatGPT and signed by an ASL expert, simulating realistic dialogues between patients and healthcare professionals. <br><br>
The repository contains the MedASL dataset, split into two subfolders: Annotations and Videos.

The Annotations subfolder includes a CSV file listing 2000 medical-related sentences, their corresponding glosses, and the file paths to the related videos signed in American Sign Language (ASL).

The Videos subfolder contains subfolders, each representing a single sentence in the dataset. Each sentence is signed in ASL, and the corresponding video is stored as .npy files, where each .npy file represents one second of the signed video.


## Prompt Engineering Design:

To create MedASL, we design and develop prompts using the following methodology:<br><br>
• High-Level Prompt Structure<br>
We design a high-level prompt to generate realistic medical conversations in the following format:<br>
“Generate a realistic medical interaction between a patient and a [doctor/nurse/pharmacist/technician] in a healthcare setting. The conversation should involve common symptoms, medical advice, and questions about treatments or prescriptions. Ensure the language is clear, professional, and appropriate for real-world scenarios.”<br><br>
• Refinement Process <br>
We refine the high-level prompt by dividing it into low-level prompts. This is to improve the generated sentences’ coherence and relevance using low-level prompt variations such as:<br>
“Generate 10 medical-related statements that a nurse might say when checking a patient’s vitals.”


## Data Recording Process:

We recorded the sign videos using Intel RealSense at a resolution of 1280×800 and stored them in “.npy” format. 


## Data Pre-processing:

For the sign language gloss and spoken language text, we applied the following additional pre-processing steps:<br><br>
• Building Vocabularies: We create unique vocabularies for gloss and text data, including a special token <UNK> to represent unknown words.<br>
• Assigning Unique Indices: We assign a unique index to each word in the gloss and text data for better processing.<br>
• Tokenizing: We tokenize the gloss and text sequences into individual units to enable efficient input representation.<br>
• Padding: We apply zero-padding to the sequences, ensuring uniform lengths for batch processing.<br>
• Adding Special Tokens: We add special tokens such as <sos> (start of sequence) and <eos> (end of sequence) to mark the sequence boundaries.<br>
• Gloss Alignment: We map each gloss annotation to its corresponding spoken language sentence.<br>
• Data serialization: We store the pre-processed gloss, text, and video data in a standardized “.pkl” format for efficient input loading during model training.<br><br>

To prepare the video data for training, we applied the following pre-processing steps: <br><br>
• Video Frames Extraction: We sample the videos at 30 frames per second (fps) to maintain motion fidelity.<br>
• Video Keypoints Extraction: We represent every frame by its corresponding keypoints, extracted using the Mediapipe Python library.<br>
• Frames Concatenation: We concatenate video frames corresponding to each sentence into a continuous sequence to align with gloss annotations.<br>
• Padding: We apply zero-padding to align frame lengths across all videos. 



