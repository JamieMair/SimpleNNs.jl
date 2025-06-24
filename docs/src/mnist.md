# MNIST Classification

In this basic example, we'll train a convolutional neural network to classify handwritten digits from the MNIST dataset. This example demonstrates both CPU and GPU training, data preprocessing, and model evaluation.

This tutorial will cover:
- Loading and preprocessing MNIST data
- Creating a CNN architecture with SimpleNNs.jl
- Setting up GPU acceleration (optional)
- Implementing a complete training loop
- Evaluating model performance
- Visualising results and analysis

## Dataset Preparation

First, let's load and prepare the MNIST dataset using `MLDatasets.jl`:

```julia
using MLDatasets
using SimpleNNs
using Random

# Load the training dataset
dataset = MNIST(:train)
images, labels = dataset[:]

# Reshape images to add channel dimension: (height, width, channels, batch)
images = reshape(images, size(images, 1), size(images, 2), 1, size(images, 3))

# Convert labels to 1-indexed (MNIST uses 0-9, we need 1-10)
labels .+= 1

println("Dataset shape: ", size(images))
println("Labels range: ", extrema(labels))
```

## GPU Setup (Optional)

If you have an NVIDIA GPU and want to use GPU acceleration, set up the necessary packages:

```julia
# Only run this if you have CUDA capable hardware
using CUDA, cuDNN, NNlib

# Check if CUDA is functional
gpu_available = CUDA.functional()
println("GPU available: ", gpu_available)

# Define device transfer function
use_gpu = gpu_available  # Set to false to force CPU usage
to_device = use_gpu ? gpu : identity

# Move data to chosen device
images = images |> to_device
labels = labels |> to_device

println("Training on: ", use_gpu ? "GPU" : "CPU")
```

## Model Architecture

We'll create a convolutional neural network with the following architecture:
- Convolutional layer: 5×5 kernels, 16 output channels, ReLU activation
- Max pooling: 2×2 pool size
- Convolutional layer: 3×3 kernels, 8 output channels, ReLU activation  
- Max pooling: 4×4 pool size
- Flatten layer
- Dense layer: 32 units, ReLU activation
- Output layer: 10 units (one per digit class)

```julia
# Get image dimensions (excluding batch dimension)
img_size = size(images)[1:end-1]  # (28, 28, 1)

model = chain(
    Static(img_size),
    Conv((5,5), 16; activation_fn=relu),
    MaxPool((2,2)),
    Conv((3,3), 8; activation_fn=relu),
    MaxPool((4,4)),
    Flatten(),
    Dense(32, activation_fn=relu),
    Dense(10, activation_fn=identity)  # 10 classes for digits 0-9
) |> to_device

println("Model created with ", length(parameters(model)), " parameters")
```

## Training Setup

Set up the training infrastructure with preallocated buffers and optimiser:

```julia
# Training hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 1000

# Preallocate forward and backward buffers
forward_buffer = preallocate(model, batch_size)
gradient_buffer = preallocate_grads(model, batch_size)

# Initialise model parameters
Random.seed!(1234)
params = parameters(model)
randn!(params)
params .*= 0.05f0  # Small initial weights

# Setup Adam optimiser using built-in SimpleNNs optimiser
optimiser = AdamOptimiser(gradient_buffer.parameter_gradients; lr=learning_rate)

println("Training setup complete")
```

## Training Loop

Now we'll implement the training loop with mini-batch stochastic gradient descent:

```julia
using ProgressBars

# Track training progress
losses = Float32[]
training_indices = collect(1:size(images, 4))

println("Starting training for $epochs epochs...")

for epoch in ProgressBar(1:epochs)
    # Randomly shuffle and select a batch
    Random.shuffle!(training_indices)
    batch_indices = view(training_indices, 1:batch_size)
    
    # Extract current batch
    batch_images = view(images, :, :, :, batch_indices)
    batch_labels = view(labels, batch_indices)
    
    # Set inputs for forward pass
    set_inputs!(forward_buffer, batch_images)
    
    # Create loss function for this batch
    loss_fn = LogitCrossEntropyLoss(batch_labels, 10)
    
    # Forward pass
    forward!(forward_buffer, model)
    
    # Backward pass and loss computation
    current_loss = backprop!(gradient_buffer, forward_buffer, model, loss_fn)
    push!(losses, current_loss)
    
    # Extract gradients and apply optimiser
    grads = gradients(gradient_buffer)
    update!(params, grads, optimiser)
    
    # Print progress every 100 epochs
    if epoch % 100 == 0
        println("Epoch $epoch: Loss = $(round(current_loss, digits=4))")
    end
end

println("Training completed!")
```

## Visualising Training Progress

You are free to use any package to visualise the training curves.

One popular option in the Julia ecosystem is [`Plots.jl`](https://docs.juliaplots.org/latest/), with the below example:

```julia
using Plots

# Plot training loss
plot(losses, 
     xlabel="Epoch", 
     ylabel="Cross Entropy Loss", 
     title="MNIST Training Progress",
     lw=2, 
     label="Training Loss",
     yscale=:log10)  # Log scale for better visualisation
```

One can also use [`TensorBoardLogger.jl`](https://github.com/JuliaLogging/TensorBoardLogger.jl) to plot and view training updates in real time - but this requires installing `tensorboard` separately in Python.

## Model Evaluation

Evaluate the trained model on the test set:

```julia
# Load test dataset
test_dataset = MNIST(:test)
test_images, test_labels = test_dataset[:]

# Preprocess test data same way as training data
test_images = reshape(test_images, size(test_images, 1), size(test_images, 2), 1, size(test_images, 3))
test_labels .+= 1  # Convert to 1-indexed
test_images = test_images |> to_device
test_labels = test_labels |> to_device

println("Test set size: ", size(test_images))
```

### Inference on Test Set

```julia
# Create forward buffer for entire test set
n_test = size(test_images, 4)
test_forward_buffer = preallocate(model, n_test)

# Run inference
set_inputs!(test_forward_buffer, test_images)
forward!(test_forward_buffer, model)

# Get predictions
logits = get_outputs(test_forward_buffer)
predictions = [argmax(col)[1] for col in eachcol(Array(logits))]

# Convert back to CPU for analysis if needed
cpu_test_labels = Array(test_labels)
cpu_predictions = predictions

# Calculate accuracy
correct_predictions = sum(cpu_predictions .== cpu_test_labels)
accuracy = correct_predictions / length(cpu_test_labels) * 100

println("Test Accuracy: $(round(accuracy, digits=2))%")
println("Correct: $correct_predictions / $(length(cpu_test_labels))")
```

### Detailed Analysis

Create a confusion matrix to analyze model performance:

```julia
# Create confusion matrix
function confusion_matrix(y_true, y_pred, n_classes=10)
    cm = zeros(Int, n_classes, n_classes)
    for (true_label, pred_label) in zip(y_true, y_pred)
        cm[true_label, pred_label] += 1
    end
    return cm
end

cm = confusion_matrix(cpu_test_labels, cpu_predictions)

# Display confusion matrix
println("Confusion Matrix:")
println("Rows: True labels, Columns: Predicted labels")
for i in 1:10
    println("Class $(i-1): ", join(cm[i, :], "\t"))
end

# Per-class accuracy
class_accuracies = [cm[i,i] / sum(cm[i,:]) for i in 1:10]
for (i, acc) in enumerate(class_accuracies)
    println("Digit $(i-1) accuracy: $(round(acc*100, digits=1))%")
end
```

### Visualising Predictions

Let's visualise some test examples with predictions:

```julia
using Plots

# Function to display digit images
function plot_digit_predictions(images, true_labels, predictions, indices)
    n = length(indices)
    plots = []
    
    for i in 1:min(n, 16)  # Show up to 16 examples
        idx = indices[i]
        img = Array(images[:, :, 1, idx])
        true_label = true_labels[idx] - 1  # Convert back to 0-9
        pred_label = predictions[idx] - 1
        
        is_correct = true_label == pred_label
        title_color = is_correct ? :green : :red
        
        p = heatmap(img', 
                   color=:grays, 
                   aspect_ratio=:equal,
                   title="T:$true_label P:$pred_label",
                   titlefontcolor=title_color,
                   showaxis=false,
                   grid=false)
        push!(plots, p)
    end
    
    plot(plots..., layout=(4, 4), size=(800, 800))
end

# Show some random predictions
random_indices = rand(1:n_test, 16)
plot_digit_predictions(test_images, cpu_test_labels, cpu_predictions, random_indices)
```