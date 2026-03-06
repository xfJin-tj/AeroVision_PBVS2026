# 1st place solution for 5th Multi-modal Aerial View Imagery Challenge: Classification (MAVIC-C)
# Environment
Install the required dependencies by running `pip install -r environment.txt`.

# Preparation
Download the dataset from the official website and put it in the `./datasets` folder.

# Inference
Download the pretrained weights for the cross_model_2resnet, cross_model_1, and model_complete models, and save them in the appropriate directory. Ensure you update the paths in the code to point to the correct location of these weights.

For the dataset, place your test images in the specified folder. The images should be named with numerical identifiers as the script extracts image_id from the filename using regular expressions, and must be in PNG format.

Run the inference by executing `python infer.py`. This will load the pretrained models, process the test images with respective transforms, fuse predictions using the specified fusion method based on class-wise F1 scores, generate final predictions, and save the results in a `results.csv` file. The output CSV will contain columns for `image_id`, `class_id`, and `score`.

Once the script finishes running, submit the `results.csv` file to the competition platform.

