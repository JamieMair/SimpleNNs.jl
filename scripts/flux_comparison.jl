using CUDA
import Flux
import SimpleNNs
SimpleNNs.enable_gpu()
using Random
using ProgressBars
using BenchmarkTools
using Plots

include("data_utils.jl")
include("flux_utils.jl")
include("simplenn_utils.jl")

device = :gpu;
batch_size = 128;
dataset = load_mnist_data(batch_size, device);

img_size = (28,28);
in_channels = 1;
model_flux = create_flux_mnist_model(img_size, in_channels, device);
model_simple = create_simple_nn_mnist_model(img_size, in_channels, device);
randn!(model_simple.parameters);
model_simple.parameters .*= (1/1000);

input_size = (img_size..., in_channels, batch_size);
forward_cache = SimpleNNs.preallocate(model_simple, batch_size);
batch_features, batch_indices = current_batch(dataset);
SimpleNNs.set_inputs!(forward_cache, reshape(batch_features, input_size));

parameters, re = Flux.destructure(model_flux);
function inference!(parameters, model::SimpleNNs.Model, forward_cache)
    model.parameters .= parameters # copy new parameters
    SimpleNNs.forward!(forward_cache, model)
    nothing
end
function inference!(parameters, re, forward_cache)
    model = re(parameters)
    outputs = model(forward_cache.input)
    nothing
end




## PERFORMANCE BENCHMARKS (RTX 3090) ##
# julia> @benchmark CUDA.@sync inference!($parameters, $model_simple, $forward_cache)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  149.848 μs … 210.264 μs  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     152.472 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   152.761 μs ±   1.483 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%
# 
#                  ▁▄▇█▇▆▅▅▅▄▂▃▂▂                                  
#   ▁▁▁▁▂▂▂▂▃▂▂▃▃▅▆██████████████▇▆▆▆▆▆▅▅▅▅▄▄▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▁▁▂▁▁ ▄
#   150 μs           Histogram: frequency by time          157 μs <
# 
#  Memory estimate: 10.23 KiB, allocs estimate: 191.
# 
# julia> @benchmark CUDA.@sync inference!($parameters, $re, $forward_cache)
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  214.573 μs … 59.864 ms  ┊ GC (min … max): 0.00% … 30.48%
#  Time  (median):     219.822 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   250.831 μs ±  1.310 ms  ┊ GC (mean ± σ):  3.55% ±  0.68%
# 
#             ▃▄▅█▇▅▄▂                                            
#   ▁▁▁▁▁▂▃▄▅█████████▇▆▅▄▃▃▃▃▃▃▄▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁ ▃
#   215 μs          Histogram: frequency by time          234 μs <
# 
#  Memory estimate: 37.09 KiB, allocs estimate: 769.

mutable struct MiniBatch{T, L}
    features::T
    labels::L
    indices::Vector{Int}
    batch_size::Int
    num_samples::Int
    current_batch::Int
    rng::Random.AbstractRNG
    
    function MiniBatch(features, labels, batch_size::Int; rng=Random.default_rng())
        num_samples = size(features, ndims(features))
        indices = collect(1:num_samples)
        new{typeof(features), typeof(labels)}(features, labels, indices, batch_size, num_samples, 1, rng)
    end
end

function reset_epoch!(mb::MiniBatch)
    Random.shuffle!(mb.rng, mb.indices)
    mb.current_batch = 1
    nothing
end

function num_batches(mb::MiniBatch)
    return div(mb.num_samples, mb.batch_size)
end

function has_next_batch(mb::MiniBatch)
    return mb.current_batch <= num_batches(mb)
end

function next_batch!(mb::MiniBatch)
    if !has_next_batch(mb)
        return nothing, nothing
    end
    
    start_idx = (mb.current_batch - 1) * mb.batch_size + 1
    end_idx = mb.current_batch * mb.batch_size
    batch_indices = view(mb.indices, start_idx:end_idx)
    
    # Use views to avoid allocations
    if ndims(mb.features) == 4  # Image data (H, W, C, N)
        batch_features = view(mb.features, :, :, :, batch_indices)
    elseif ndims(mb.features) == 2  # Flattened data (features, N)
        batch_features = view(mb.features, :, batch_indices)
    else
        error("Unsupported feature dimensionality: $(ndims(mb.features))")
    end
    
    # Handle both 1D integer labels and 2D one-hot labels
    if ndims(mb.labels) == 1
        batch_labels = view(mb.labels, batch_indices)
    else  # One-hot labels (classes, samples)
        batch_labels = view(mb.labels, :, batch_indices)
    end
    
    mb.current_batch += 1
    return batch_features, batch_labels
end

function next_batch_flux!(mb::MiniBatch)
    if !has_next_batch(mb)
        return nothing, nothing
    end
    
    start_idx = (mb.current_batch - 1) * mb.batch_size + 1
    end_idx = mb.current_batch * mb.batch_size
    batch_indices = view(mb.indices, start_idx:end_idx)
    
    # For Flux, copy data to avoid GPU scalar indexing issues with views
    if ndims(mb.features) == 4  # Image data (H, W, C, N)
        batch_features = mb.features[:, :, :, batch_indices]  # Copy instead of view
    elseif ndims(mb.features) == 2  # Flattened data (features, N)
        batch_features = mb.features[:, batch_indices]  # Copy instead of view
    else
        error("Unsupported feature dimensionality: $(ndims(mb.features))")
    end
    
    # Handle both 1D integer labels and 2D one-hot labels
    if ndims(mb.labels) == 1
        batch_labels = mb.labels[batch_indices]  # Copy instead of view
    else  # One-hot labels (classes, samples)
        batch_labels = mb.labels[:, batch_indices]  # Copy instead of view
    end
    
    mb.current_batch += 1
    return batch_features, batch_labels
end

function train_simplenn_epoch!(model, minibatch, opt, forward_cache, backward_cache, loss_fn)
    reset_epoch!(minibatch)
    num_batches_processed = 0
    
    while has_next_batch(minibatch)
        batch_features, batch_labels = next_batch!(minibatch)
        if batch_features === nothing
            break
        end
        
        # Forward pass
        if ndims(batch_features) == 3  # Need to add channel dimension
            input_data = reshape(batch_features, (28, 28, 1, minibatch.batch_size))
        else
            input_data = batch_features
        end
        
        SimpleNNs.set_inputs!(forward_cache, input_data)
        SimpleNNs.forward!(forward_cache, model)
        
        # Backward pass
        SimpleNNs.backprop!(backward_cache, forward_cache, model, loss_fn)
        
        # Update parameters
        grads = SimpleNNs.gradients(backward_cache)
        SimpleNNs.update!(model.parameters, grads, opt)
        
        num_batches_processed += 1
    end
    
    return num_batches_processed
end

function train_flux_epoch!(model, minibatch, flux_state)
    reset_epoch!(minibatch)
    total_loss = 0.0f0
    num_batches_processed = 0
    
    while has_next_batch(minibatch)
        batch_features, batch_labels = next_batch_flux!(minibatch)  # Use Flux-specific function
        if batch_features === nothing
            break
        end
        
        # Forward and backward pass with gradient computation
        loss, grads = Flux.withgradient(model) do m
            if ndims(batch_features) == 3  # Need to add channel dimension
                input_data = reshape(batch_features, (28, 28, 1, minibatch.batch_size))
            else
                input_data = batch_features
            end
            predictions = m(input_data)
            Flux.crossentropy(predictions, batch_labels)
        end
        
        total_loss += loss
        
        # Update parameters
        Flux.update!(flux_state, model, grads[1])
        
        num_batches_processed += 1
    end
    
    return total_loss / num_batches_processed
end

function reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
    # Reset SimpleNNs model
    model_simple.parameters .= original_simple_params
    
    # Reset Flux model - use the stored model state directly
    Flux.loadmodel!(model_flux, original_flux_params)
    nothing
end

function benchmark_training_comparison(model_simple, model_flux, dataset; epochs=25)
    println("Benchmarking Training: SimpleNNs vs Flux")
    println("="^50)
    
    # Create minibatch from dataset
    features, raw_labels = dataset  # Extract features and labels from dataset
    
    # Pre-convert labels to one-hot for Flux (on CPU first, then move to GPU)
    onehot_labels = Flux.onehotbatch(Array(raw_labels), 1:10)
    if device == :gpu
        onehot_labels = onehot_labels |> SimpleNNs.gpu
    end
    
    # Create shared RNG for fair comparison
    shared_rng = Random.MersenneTwister(42)
    
    # Create separate minibatch objects for each framework
    minibatch_simple = MiniBatch(features, raw_labels, batch_size; rng=shared_rng)
    minibatch_flux = MiniBatch(features, onehot_labels, batch_size; rng=shared_rng)
    
    # Store original parameters for reset
    flux_params, _ = Flux.destructure(model_simple.parameters)
    model_simple.parameters .= flux_params
    original_simple_params = copy(model_simple.parameters)
    original_flux_params = deepcopy(model_flux)
    
    # Setup optimizers
    lr = 0.001f0
    
    # Preallocate for SimpleNNs
    forward_cache = SimpleNNs.preallocate(model_simple, batch_size)
    backward_cache = SimpleNNs.preallocate_grads(model_simple, batch_size)
    
    # Create loss function for SimpleNNs (assuming crossentropy classification)
    loss_fn = SimpleNNs.LogitCrossEntropyLoss(raw_labels, 10)  # Create loss object once
    
    # Manual benchmark with 3 repeats
    num_repeats = 3
    simple_times = Float64[]
    flux_times = Float64[]
    
    println("Benchmarking SimpleNNs training ($epochs epochs, $num_repeats repeats)...")
    for repeat in 1:num_repeats
        # Reset everything for this repeat
        reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
        Random.seed!(shared_rng, 42)
        
        # Create fresh optimizer
        simple_opt = SimpleNNs.AdamOptimiser(model_simple.parameters; lr=lr)
        
        # Warmup if first repeat
        if repeat == 1
            println("  Warming up...")
            for epoch in 1:min(2, epochs)
                train_simplenn_epoch!(model_simple, minibatch_simple, simple_opt, forward_cache, backward_cache, loss_fn)
            end
            # Reset after warmup
            reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
            Random.seed!(shared_rng, 42)
            simple_opt = SimpleNNs.AdamOptimiser(model_simple.parameters; lr=lr)
        end
        
        # Time only the training loop
        println("  Repeat $repeat...")
        training_time = @elapsed CUDA.@sync begin
            for epoch in 1:epochs
                train_simplenn_epoch!(model_simple, minibatch_simple, simple_opt, forward_cache, backward_cache, loss_fn)
            end
        end
        
        push!(simple_times, training_time)
        
        # Reset model after training
        reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
    end
    
    println("Benchmarking Flux training ($epochs epochs, $num_repeats repeats)...")
    for repeat in 1:num_repeats
        # Reset everything for this repeat
        reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
        Random.seed!(shared_rng, 42)
        
        # Create fresh optimizer
        flux_opt = Flux.Adam(lr)
        flux_state = Flux.setup(flux_opt, model_flux)
        
        # Warmup if first repeat
        if repeat == 1
            println("  Warming up...")
            for epoch in 1:min(2, epochs)
                train_flux_epoch!(model_flux, minibatch_flux, flux_state)
            end
            # Reset after warmup
            reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
            Random.seed!(shared_rng, 42)
            flux_opt = Flux.Adam(lr)
            flux_state = Flux.setup(flux_opt, model_flux)
        end
        
        # Time only the training loop
        println("  Repeat $repeat...")
        training_time = @elapsed CUDA.@sync begin
            for epoch in 1:epochs
                train_flux_epoch!(model_flux, minibatch_flux, flux_state)
            end
        end
        
        push!(flux_times, training_time)
        
        # Reset model after training
        reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
    end
    
    # Calculate statistics
    simple_median = median(simple_times)
    simple_mean = mean(simple_times)
    simple_min = minimum(simple_times)
    simple_max = maximum(simple_times)
    
    flux_median = median(flux_times)
    flux_mean = mean(flux_times)
    flux_min = minimum(flux_times)
    flux_max = maximum(flux_times)
    
    # Print results
    println("\nBenchmark Results:")
    println("="^50)
    println("SimpleNNs ($epochs epochs, $num_repeats repeats):")
    println("  Median time: $(round(simple_median * 1000, digits=3))ms")
    println("  Mean time: $(round(simple_mean * 1000, digits=3))ms")
    println("  Min time: $(round(simple_min * 1000, digits=3))ms")
    println("  Max time: $(round(simple_max * 1000, digits=3))ms")
    println("  Times: $(round.(simple_times .* 1000, digits=3))ms")
    
    println("\nFlux ($epochs epochs, $num_repeats repeats):")
    println("  Median time: $(round(flux_median * 1000, digits=3))ms")
    println("  Mean time: $(round(flux_mean * 1000, digits=3))ms")
    println("  Min time: $(round(flux_min * 1000, digits=3))ms")
    println("  Max time: $(round(flux_max * 1000, digits=3))ms")
    println("  Times: $(round.(flux_times .* 1000, digits=3))ms")
    
    speedup = flux_median / simple_median
    println("\nSpeedup: $(round(speedup, digits=2))x $(speedup > 1 ? "(SimpleNNs faster)" : "(Flux faster)")")
    
    # Final reset to original state
    reset_model_parameters!(model_simple, model_flux, original_simple_params, original_flux_params)
    
    return (simple_times=simple_times, flux_times=flux_times)
end

function train_simplenn_full(model, dataset, test_dataset; epochs=25, lr=0.001f0)
    println("Training SimpleNNs model for $epochs epochs...")
    
    # Extract data
    features, raw_labels = dataset
    test_features, test_raw_labels = test_dataset
    
    # Create minibatch
    shared_rng = Random.MersenneTwister(42)
    minibatch = MiniBatch(features, raw_labels, batch_size; rng=shared_rng)
    
    # Preallocate
    forward_cache = SimpleNNs.preallocate(model, batch_size)
    backward_cache = SimpleNNs.preallocate_grads(model, batch_size)
    test_forward_cache = SimpleNNs.preallocate(model, size(test_features, 4))
    
    # Create loss function and optimizer (using same pattern as benchmark)
    loss_fn = SimpleNNs.LogitCrossEntropyLoss(raw_labels, 10)
    test_loss_fn = SimpleNNs.LogitCrossEntropyLoss(test_raw_labels, 10)
    opt = SimpleNNs.AdamOptimiser(model.parameters; lr=lr)
    
    # Track metrics
    train_losses = Float32[]
    test_losses = Float32[]
    train_accuracies = Float32[]
    test_accuracies = Float32[]
    
    for epoch in 1:epochs
        # Training - use same pattern as train_simplenn_epoch!
        reset_epoch!(minibatch)
        correct_train = 0
        total_train = 0
        batch_count = 0 
        train_loss = 0
        while has_next_batch(minibatch)
            batch_features, batch_labels = next_batch!(minibatch)
            if batch_features === nothing
                break
            end
            batch_count +=1
            # Forward pass - same as benchmark
            input_data = if ndims(batch_features) == 3
                reshape(batch_features, (28, 28, 1, minibatch.batch_size))
            else
                batch_features
            end
            
            SimpleNNs.set_inputs!(forward_cache, input_data)
            SimpleNNs.forward!(forward_cache, model)
            
            # Calculate accuracy before backprop
            predictions = SimpleNNs.get_outputs(forward_cache)
            pred_classes = argmax(Array(predictions), dims=1)[1, :]
            correct_train += sum(pred_classes .== Array(batch_labels))
            total_train += length(batch_labels)
            
            # Backward pass - use same pattern as benchmark
            loss_value = SimpleNNs.backprop!(backward_cache, forward_cache, model, loss_fn)
            train_loss += loss_value
            # Update parameters - same as benchmark
            grads = SimpleNNs.gradients(backward_cache)
            SimpleNNs.update!(model.parameters, grads, opt)
        end
        push!(train_losses, train_loss / batch_count)
        # Calculate train loss using a separate forward pass
        train_forward_cache = SimpleNNs.preallocate(model, min(1000, size(features, 4)))  # Use subset for efficiency
        train_subset_size = min(1000, size(features, 4))
        train_subset_features = features[:, :, :, 1:train_subset_size]
        train_subset_labels = raw_labels[1:train_subset_size]
        
        SimpleNNs.set_inputs!(train_forward_cache, train_subset_features)
        SimpleNNs.forward!(train_forward_cache, model)
        train_predictions = SimpleNNs.get_outputs(train_forward_cache)
        
        # Test evaluation - use forward pass and loss function
        SimpleNNs.set_inputs!(test_forward_cache, test_features)
        SimpleNNs.forward!(test_forward_cache, model)
        test_predictions = SimpleNNs.get_outputs(test_forward_cache)
        
        # Calculate test loss using loss function (same pattern as test.jl)
        temp_test_backward_cache = SimpleNNs.preallocate_grads(model, size(test_features, 4))
        test_loss = SimpleNNs.backprop!(temp_test_backward_cache, test_forward_cache, model, test_loss_fn)
        
        # Test accuracy
        test_pred_classes = argmax(Array(test_predictions), dims=1) .- 1
        correct_test = sum(test_pred_classes .== Array(test_raw_labels))
        test_accuracy = correct_test / length(test_raw_labels)
        
        # Store metrics
        train_accuracy = correct_train / total_train
        
        push!(train_losses, train_loss)
        push!(test_losses, test_loss)
        push!(train_accuracies, train_accuracy)
        push!(test_accuracies, test_accuracy)
        
        if epoch % 5 == 0 || epoch == 1
            println("Epoch $epoch: Train Loss = $(round(train_losses[end], digits=4)), " *
                   "Test Loss = $(round(test_losses[end], digits=4)), " *
                   "Train Acc = $(round(train_accuracies[end]*100, digits=2))%, " *
                   "Test Acc = $(round(test_accuracies[end]*100, digits=2))%")
        end
    end
    
    return (train_losses=train_losses, test_losses=test_losses, 
            train_accuracies=train_accuracies, test_accuracies=test_accuracies)
end

function train_flux_full(model, dataset, test_dataset; epochs=25, lr=0.001f0)
    println("Training Flux model for $epochs epochs...")
    
    # Extract and convert data
    features, raw_labels = dataset
    test_features, test_raw_labels = test_dataset
    
    # Convert labels to one-hot
    onehot_labels = Flux.onehotbatch(Array(raw_labels), 1:10)
    test_onehot_labels = Flux.onehotbatch(Array(test_raw_labels), 1:10)
    if device == :gpu
        onehot_labels = onehot_labels |> SimpleNNs.gpu
        test_onehot_labels = test_onehot_labels |> SimpleNNs.gpu
    end
    
    # Create minibatch
    shared_rng = Random.MersenneTwister(42)
    minibatch = MiniBatch(features, onehot_labels, batch_size; rng=shared_rng)
    
    # Create optimizer
    flux_opt = Flux.Adam(lr)
    flux_state = Flux.setup(flux_opt, model)
    
    # Track metrics
    train_losses = Float32[]
    test_losses = Float32[]
    train_accuracies = Float32[]
    test_accuracies = Float32[]
    
    for epoch in 1:epochs
        # Training
        reset_epoch!(minibatch)
        epoch_loss = 0.0f0
        correct_train = 0
        total_train = 0
        batch_count = 0
        while has_next_batch(minibatch)
            batch_features, batch_labels = next_batch_flux!(minibatch)
            if batch_features === nothing
                break
            end
            batch_count += 1
            # Forward and backward pass
            loss, grads = Flux.withgradient(model) do m
                input_data = if ndims(batch_features) == 3
                    reshape(batch_features, (28, 28, 1, minibatch.batch_size))
                else
                    batch_features
                end
                predictions = m(input_data)
                
                # Calculate accuracy
                pred_classes = argmax(Array(predictions), dims=1)
                true_classes = argmax(Array(batch_labels), dims=1)
                correct_train += sum(pred_classes .== true_classes)
                total_train += length(true_classes)
                
                Flux.crossentropy(predictions, batch_labels)
            end
            
            epoch_loss += loss
            
            # Update parameters
            Flux.update!(flux_state, model, grads[1])
        end
        
        # Test evaluation
        test_predictions = model(test_features)
        test_loss = Flux.crossentropy(test_predictions, test_onehot_labels)
        
        # Test accuracy
        test_pred_classes = argmax(Array(test_predictions), dims=1)
        test_true_classes = argmax(Array(test_onehot_labels), dims=1)
        correct_test = sum(test_pred_classes .== test_true_classes)
        test_accuracy = correct_test / length(test_true_classes)
        
        # Store metrics
        push!(train_losses, epoch_loss / batch_count)
        push!(test_losses, test_loss)
        push!(train_accuracies, correct_train / total_train)
        push!(test_accuracies, test_accuracy)
        
        if epoch % 5 == 0 || epoch == 1
            println("Epoch $epoch: Train Loss = $(round(train_losses[end], digits=4)), " *
                   "Test Loss = $(round(test_losses[end], digits=4)), " *
                   "Train Acc = $(round(train_accuracies[end]*100, digits=2))%, " *
                   "Test Acc = $(round(test_accuracies[end]*100, digits=2))%")
        end
    end
    
    return (train_losses=train_losses, test_losses=test_losses, 
            train_accuracies=train_accuracies, test_accuracies=test_accuracies)
end

function compare_training_performance(model_simple, model_flux, train_dataset, test_dataset; epochs=25)
    println("Comparing Training Performance: SimpleNNs vs Flux")
    println("="^60)
    
    # Synchronize initial parameters
    flux_params, re = Flux.destructure(model_flux)
    model_simple.parameters .= flux_params
    original_params = copy(model_simple.parameters)
    original_flux_model = deepcopy(model_flux)  # Store the actual model state
    
    # Train SimpleNNs
    println("\n" * "="^30 * " SimpleNNs " * "="^30)
    model_simple.parameters .= original_params
    simple_results = train_simplenn_full(model_simple, train_dataset, test_dataset; epochs=epochs)
    
    # Reset and train Flux
    println("\n" * "="^32 * " Flux " * "="^32)
    Flux.loadmodel!(model_flux, original_flux_model)  # Reset using the stored model
    flux_results = train_flux_full(model_flux, train_dataset, test_dataset; epochs=epochs)
    
    # Create comparison plots
    epochs_range = 1:epochs
    
    # Loss comparison plot
    loss_plot = plot(epochs_range, simple_results.train_losses, 
                    label="SimpleNNs Train", linewidth=2, linestyle=:solid)
    plot!(loss_plot, epochs_range, simple_results.test_losses, 
          label="SimpleNNs Test", linewidth=2, linestyle=:dash)
    plot!(loss_plot, epochs_range, flux_results.train_losses, 
          label="Flux Train", linewidth=2, linestyle=:solid)
    plot!(loss_plot, epochs_range, flux_results.test_losses, 
          label="Flux Test", linewidth=2, linestyle=:dash)
    plot!(loss_plot, xlabel="Epoch", ylabel="Loss", title="Training Loss Comparison")
    plot!(loss_plot, legend=:topright, grid=true)
    
    # Accuracy comparison plot
    acc_plot = plot(epochs_range, simple_results.train_accuracies .* 100, 
                   label="SimpleNNs Train", linewidth=2, linestyle=:solid)
    plot!(acc_plot, epochs_range, simple_results.test_accuracies .* 100, 
          label="SimpleNNs Test", linewidth=2, linestyle=:dash)
    plot!(acc_plot, epochs_range, flux_results.train_accuracies .* 100, 
          label="Flux Train", linewidth=2, linestyle=:solid)
    plot!(acc_plot, epochs_range, flux_results.test_accuracies .* 100, 
          label="Flux Test", linewidth=2, linestyle=:dash)
    plot!(acc_plot, xlabel="Epoch", ylabel="Accuracy (%)", title="Training Accuracy Comparison")
    plot!(acc_plot, legend=:bottomright, grid=true)
    
    # Combined plot
    combined_plot = plot(loss_plot, acc_plot, layout=(2,1), size=(800, 600))
    
    # Print final results comparison
    println("\n" * "="^25 * " Final Results " * "="^25)
    println("SimpleNNs - Final Test Loss: $(round(simple_results.test_losses[end], digits=4)), " *
           "Final Test Accuracy: $(round(simple_results.test_accuracies[end]*100, digits=2))%")
    println("Flux      - Final Test Loss: $(round(flux_results.test_losses[end], digits=4)), " *
           "Final Test Accuracy: $(round(flux_results.test_accuracies[end]*100, digits=2))%")
    
    # Reset models to original state
    model_simple.parameters .= original_params
    Flux.loadmodel!(model_flux, original_flux_model)
    
    return (simple_results=simple_results, flux_results=flux_results, 
            loss_plot=loss_plot, acc_plot=acc_plot, combined_plot=combined_plot)
end

# Example benchmark
# benchmark_results = benchmark_training_comparison(model_simple, model_flux, (dataset.features, dataset.labels); epochs=5)
# Example usage (uncommented to run):
# Load test dataset (assuming similar structure to train dataset)
test_dataset = load_mnist_test_data(batch_size, device)  # You'll need to implement this
training_comparison = compare_training_performance(model_simple, model_flux, 
                                                  (dataset.features, dataset.labels), 
                                                  (test_dataset.features, test_dataset.labels); 
                                                  epochs=20)
display(training_comparison.combined_plot)