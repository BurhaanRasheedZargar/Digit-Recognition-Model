# Digit-Recognition-Model
This project implements a neural network to classify handwritten digits from the MNIST dataset. The model is built using TensorFlow and Keras, and it achieves high accuracy in identifying digits from 0 to 9.

Model Architecture
The model is a feedforward neural network with the following architecture:

Input Layer:

Accepts input images of size 28x28 pixels.
Images are flattened into a single-dimensional array of 784 pixels.

Hidden Layer:

A fully connected (Dense) layer with 128 neurons.
Activation function: ReLU (Rectified Linear Unit).

Output Layer:

A fully connected (Dense) layer with 10 neurons, representing the 10 digit classes (0–9).
Activation function: Softmax for multi-class probability distribution.
Data Preprocessing
The MNIST dataset is loaded using TensorFlow's tf.keras.datasets.mnist.
Training, validation, and test splits:
Training data: 80% of the dataset.
Validation data: 20% of the training data.
Test data: Provided as a separate set.
Images are normalized by dividing pixel values by 255.0, scaling them to a range of 0 to 1.
Model Training
Loss Function: Sparse Categorical Crossentropy (used for multi-class classification problems with integer labels).
Optimizer: Adam (adaptive learning rate optimization).
Metric: Accuracy.
The model is trained for 10 epochs, and the training process is validated on the validation dataset.

Results
After training, the model is evaluated on the test dataset to measure its performance.
Test Accuracy: The final test accuracy is printed in percentage format.
Model Visualization
The architecture of the model is saved as an image file (model_architecture.png) using Keras's plot_model utility.
Saving the Model
The trained model is saved to a file named model.h5, allowing for reuse and deployment.
Running the Code

To execute the code:

Ensure Python, TensorFlow, and the required dependencies are installed.
Run the script in a Python environment that supports TensorFlow.
The results, including training accuracy, validation accuracy, and test accuracy, will be displayed in the console.
