# White-box-NN
The python files in my repository is my first attempt in understanding and coding a whitebox neural network.
I have created separate files (object oriented way) for each layers and functions used in the final python code xor.py

# ABOUT THE DATASET
My dataset is a simple XOR table

I have chosen this dataset as the points X, Y when plotted in a graph are not linearly separable, hence a neural network can be used to form a complex decision boundary

![Graph](https://user-images.githubusercontent.com/116137959/197009312-993bdf89-24fb-4ca8-8f85-8c35c1ebf3e6.jpg)

# BRIEF EXPLANATION OF THE CODE USED
The base layer just gives the basic of a neural network layer i.e., a layer should have input size, output size, a forward propagation mechanism and a backward propagation mechanism

The dense layer takes its template from the base layer, here the forward propagation mechanism and the backward propagation meachanism including the partial derivatives (dE/dx, dE/dB, dE/dW) are implemented.
where E is the error, x is the input, B is the bias and W is the weight
The weights and bias are also created in this layer

The activation layer also takes its template from the base layer, The activation function I have used is tanh function. The range of tanh function is [-1, 1] and its derivative is given by 1-(tanh)^2
Hence during forward propagation tanh function is used and during backward propagation the derivative of tanh function is used

The mathematics for the activation function is written in Activations.py

Finally all thse separate functions are used in the file xor.py.


