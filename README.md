# AeroVision_PBVS2026
# Environment
Install the required dependencies by running pip install -r environment.txt

# Preparation
Download the dataset from CodaLab and put it in the ./datasets folder.

# Inference
During testing, only SAR images are used as input. The main SAR branch generates logits and a confidence score, while the auxiliary SimpleEncoder outputs logits.

**Class Prediction**  
We fuse the logits from the main SAR branch and the auxiliary model using weights based on validation F1 scores to get the final prediction.

**OOD Detection**  
Only the confidence score from the main SAR branch is used for OOD detection to ensure stability.

**Final Output**  
For each test image, output the predicted class ID and the confidence score from the main branch.
