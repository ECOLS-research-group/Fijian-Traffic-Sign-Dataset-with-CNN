Copyright (c) 2024, ECOLS - All rights reserved.

Paper: Kumar, N. et al., "Fijian Traffic Sign Dataset: A New Collection for Image Recognition and Benchmarking," in 2024 International Conference on Neural Information Processing (ICONIP), 2024.

Programmers and Data Collectors: Nikhil Kumar, Krishan Lal, and Geeta Singh.

Supervisor: Dr. Anuraganand Sharma

Augmnetation.py makes synthetic images from images that the user provides.
Blur.py makes the images blurry depending  on the blur limit set.
Fog.py makes the images foggy depending on the intensity set (max intensity = 1)
Motionblur.py applies a motion effect to the images mimicing speed depending on the blur size (speed amount) and angle (direction of travel)

Finalcode.py is the code with the model in it. It uses the dataset and CSV files provided to train the model, output ROC curves and a CSV file which contains the metrics for the models performance. At the end it will save a trained model. 

Kaggle Link for Dataset: https://www.kaggle.com/datasets/anuraganands/fijian-traffic-sign-dataset-ftsd

This repository is also known as https://github.com/ECOLS-research-group/FTSD-Image-Recognition-and-Benchmarking-using-CNN
