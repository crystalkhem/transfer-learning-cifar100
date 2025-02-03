<h2>ResNet34 CIFAR-10 Classifier</h2>
A deep learning model utilizing ResNet34 for image classification on the CIFAR-10 dataset. This project fine-tunes a pre-trained ResNet34 model and trains it using PyTorch.
<br>
<h3>Features</h3>
<li>Loads CIFAR-10 dataset and applies image transformations.</li>
<li>Fine-tunes ResNet34, replacing the final layer for CIFAR-10 (10 classes).</li><br>
<b>Trains the model with:</b>
<li>Adam optimizer</li>
<li>Cross-Entropy loss</li>
<li>Learning rate scheduler</li>
<li>Evaluates performance using accuracy and a confusion matrix.</li>
