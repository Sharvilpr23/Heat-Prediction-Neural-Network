This program simulates the process of heat diffusion and uses the generated data to train the neural network and predict the temperature of a cell based on the coordinates.

To simulate diffusion of heat, I used extended cellular automata to calculate the temperature of each square on 20 x 20 matrix (square plate).

**Cellular Automata Algorithm:**\
Step 1: Initialize a 20 x 20 matrix with zeros (cold)\
Step 2: Apply heat (assign values $>$ 0) For this assignment, we used the formula : y (20 - y) and applied it along the y - axis.\
![Initial Heat Map](https://github.com/Sharvilpr23/Heat-Prediction-Neural-Network/blob/master/media/initial_heatmap.png)\
Step 3: Now we allow the heat to diffuse. In order to simulate it, temperature of a cell was calculated to be the average of it's Von Neumann neighbourhood.\
![](https://github.com/Sharvilpr23/Heat-Prediction-Neural-Network/blob/master/media/heatmapgeneration.gif)\
Step 4: Stop generating once the plate is stable.\
![Final Stable Heat Map](https://github.com/Sharvilpr23/Heat-Prediction-Neural-Network/blob/master/media/stable_heatmap.png)

I used the final stable heat map to train and test the neural network. A 20 x 20 matrix provides us with 400 2-d coordinates with corresponding temperature values. I decided  to randomly choose 320 data points for training the neural network and 80 data points to test it.

**Constructing the Neural Network**\
To construct the neural network, I decided to use the PyTorch framework as it is relatively easy to understand, feels more natural, is quite native and pythonic.
Input layer will consist of 2 nodes (x and y coordinates) and the output layer will consist of 1 output node (temperature). The number of nodes in the hidden layer was decided on by trial and error method. I feel that 2 hidden layers with 200 nodes each produced the best results. 

**Training the Network**\
The training data is fed to the input layer of the network which uses forward propogation to evaluate it. Each hidden layer accepts the input data, processes it as per the activation function and passes it to the successive layer. I used the sigmoid function as my activation function as it produced better results compared to relu. Once the data passes through all the hidden layers, the final output is compared to the corresponding temperature.\
The error is calculated and is back propogated through the network. The weights of the synapses are adjusted accordingly using the gradient descent method. A running sum of loss is calculated to evaluate how the network improves.\
The entire dataset is passed through the network multiple times, known as epochs. One can specify the number of epochs but for this project I decided to stop training the network once a certain loss threshold is achieved.

**Testing the Network**\
Once the network is trained, we can use our testing data to predict the temperatures. Each of the data points is passed to the network and the output predicted is compared to the actual value. Then we calculate the percentage of correctly predicted temperatures (within acceptable limits).
