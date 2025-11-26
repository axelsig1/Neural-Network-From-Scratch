import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()



# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)



#ReLU activation function
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0



# Softmax activation function
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the softmax function
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)



# Cross-entropy loss
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    

class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)    # prevent log(0)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:  # sparse labels (scalar values)
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        # Probabilities for target values - only if one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        else:
            raise Exception("Invalid shape for y_true")

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples



# Softmax classifier - combined Softmax activation and 
# cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded, turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples



# Stochastic gradient descent optimizer
class Optimizer_SGD:
    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=0.01, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.learning_rate_initial * (1.0 / (1.0 + self.decay * self.iterations))
        self.iterations += 1

    # Update parameters
    def update_params(self, layer):

        # if we use momentum
        if  self.momentum:
            # If we haven't yet created momentum arrays, do so
            if not hasattr(layer, 'weight_momentums'):
                # Initialize momentum arrays
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum
            weight_updates = (self.momentum * layer.weight_momentums) - (self.learning_rate * layer.dweights)
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = (self.momentum * layer.bias_momentums) - (self.learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates

            # Update weights and biases using momentum
            layer.weights += weight_updates
            layer.biases += bias_updates
        else:
            # Vanilla SGD updates
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases





# Stochastic gradient descent optimizer
class Optimizer_Ada_Grad:
    """AdaGrad Optimizer - Adaptive Gradient optimizer that adapts the learning rate
    for each parameter based on historical gradients.
    
    This optimizer maintains a cache of squared gradients for weights and biases,
    scaling down the learning rate for parameters with large gradients and scaling up
    for parameters with small gradients. While AdaGrad can be effective for sparse data,
    it is not as commonly used or as performant as Stochastic Gradient Descent (SGD)
    variants like Adam or RMSprop for most deep learning applications.
    
    Attributes:
        learning_rate (float): Initial learning rate (default: 1.0)
        decay (float): Learning rate decay factor (default: 0.0)
        epsilon (float): Small constant for numerical stability (default: 1e-7)
        iterations (int): Number of update iterations performed
    """

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.learning_rate_initial = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.learning_rate = self.learning_rate_initial * (1.0 / (1.0 + self.decay * self.iterations))
        self.iterations += 1

    # Update parameters
    def update_params(self, layer):

        # If we haven't yet created momentum arrays, do so
        if not hasattr(layer, 'weight_cache'):
            # Initialize momentum arrays
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        

##################################
## Next RMSProp Optimizer p.296 ##
##################################


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2 , 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 64 output values (output values)
dense2 = Layer_Dense(64 , 3)

# more layers can be added similarly
#activation2 = Activation_ReLU()
#dense3 = Layer_Dense(64 , 64)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Create optimizer
#optimizer = Optimizer_SGD(learning_rate=1.0, decay=0.0001, momentum=0.9)
optimizer = Optimizer_Ada_Grad(decay=1e-4)


# Lists to track metrics for plotting
loss_history = []
accuracy_history = []
learning_rate_history = []

# Training loop
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through remaining layers
    dense2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)

    # Calculate accuracy
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Track metrics
    loss_history.append(loss)
    accuracy_history.append(accuracy)
    learning_rate_history.append(optimizer.learning_rate)

    # Print epoch, loss and accuracy
    if not epoch % 100:
        print(f'epoch: {epoch}, \tloss: {loss:.3f}, \taccuracy: {accuracy:.3f}, \tlr: {optimizer.learning_rate:.5f}')

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)


print("Training finished.")
print()


# Plot decision boundary
import matplotlib.pyplot as plt

# ==========================================
# VISUALIZATION CODE
# ==========================================

# Setup the figure layout
fig = plt.figure(figsize=(16, 9))
# 3 Rows, 4 Columns grid
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[0.3, 1, 1])

# 1. Decision Boundary (Left side, spans all rows, first 2 columns)
ax_main = fig.add_subplot(gs[:, :2])

# Generate grid for decision boundary
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
mesh_inputs = np.c_[xx.ravel(), yy.ravel()]
dense1.forward(mesh_inputs)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
# Use the probabilities for smooth color transitions
probs = loss_activation.activation.output if hasattr(loss_activation, 'activation') else loss_activation.output
# We need to run softmax on dense2 output manually if accessing directly
exp_values = np.exp(dense2.output - np.max(dense2.output, axis=1, keepdims=True))
probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Reshape for contour
Z = np.argmax(probs, axis=1)
Z = Z.reshape(xx.shape)

# Plot contours and data
ax_main.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
ax_main.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='black')
ax_main.set_title('Decision Boundary')

# 2. Dense 1 Weights/Biases (Top Right strip)
# Stack weights, gradients, biases for visualization
ax_d1 = fig.add_subplot(gs[0, 2:])
# Combine into a visual block: Weights top, Gradients middle, Biases bottom
# Normalize for display
w1 = dense1.weights
dw1 = dense1.dweights
b1 = dense1.biases
db1 = dense1.dbiases

# Create a visual matrix
# 2 rows of weights, 2 rows of gradients, 1 row biases, 1 row bias grads
viz_d1 = np.vstack([w1, dw1, b1, db1])
ax_d1.imshow(viz_d1, cmap='coolwarm', aspect='auto')
ax_d1.set_title('Dense 1 - Weights, Grads, Biases')
ax_d1.set_yticks([])

# 3. Dense 2 Weights (Middle Right Vertical)
ax_d2 = fig.add_subplot(gs[1:, 2])
# Dense 2 is 64x64, showing weights directly
ax_d2.imshow(dense2.weights, cmap='coolwarm', aspect='auto')
ax_d2.set_title('Dense 2 Weights')
ax_d2.set_xticks([])
ax_d2.set_yticks([])

# 4. Metrics (Loss, Accuracy, Learning Rate) - Rightmost column
sub_gs = gs[1:, 3].subgridspec(3, 1)

# Loss
ax_loss = fig.add_subplot(sub_gs[0])
ax_loss.plot(loss_history, color='brown')
ax_loss.set_title('Loss')
ax_loss.tick_params(labelbottom=False)

# Accuracy
ax_acc = fig.add_subplot(sub_gs[1])
ax_acc.plot(accuracy_history, color='blue')
ax_acc.set_title('Accuracy')
ax_acc.tick_params(labelbottom=False)

# Learning Rate
ax_lr = fig.add_subplot(sub_gs[2])
ax_lr.plot(learning_rate_history, color='green')
ax_lr.set_title('Learning Rate')

plt.tight_layout()
plt.show()