import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

if torch.cuda.is_available(): # Check if GPU is available
  device = torch.device('cuda')
  print("GPU")
else:
  device = torch.device('cpu')
  print("CPU")

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batchx in tqdm(train_loader):
            images, labels = batchx

            # Move inputs over to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = model(images) # Same thing as model.forward(images)
            
            # Backprop
            loss = loss_fn(outputs, labels)
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad() # Clear gradients before next iteration
            
            outputs = outputs.argmax(axis=1)
            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                #model.eval()?????
                
                # TODO:
                # Compute training loss and accuracy.
                #loss = loss_fn(outputs,labels)
                #print(outputs,labels)
                batchAccuracy = compute_accuracy(outputs, labels)
                print(batchAccuracy)
                
                # Log the results to Tensorboard.

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                #torch.no_grad()??
                #evaluate(val_loader, model, loss_fn)
                #torch.grad()??
                #model.train()??
                print('Epoch:', epoch, 'Loss:', loss.item())

            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
#     print(outputs.shape, labels.shape)
    print(outputs,labels)
    n_correct = (torch.round(outputs) == labels).sum().item()
    
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
#     correct = 0
#     total = 0
#     with torch.no_grad(): # IMPORTANT: turn off gradient computations
#         for batch in test_loader:
#             images, labels = batch
#             images = images.to(device)
#             labels = labels.to(device)

#             images = torch.reshape(images, (-1, 1, 28, 28))
#             outputs = model(images)
#             predictions = torch.argmax(outputs, dim=1)

#     # labels == predictions does an elementwise comparison
#     # e.g.                labels = [1, 2, 3, 4]
#     #                predictions = [1, 4, 3, 3]
#     #      labels == predictions = [1, 0, 1, 0]  (where 1 is true, 0 is false)
#     # So the number of correct predictions is the sum of (labels == predictions)
#             correct += (labels == predictions).int().sum()
#             total += len(predictions)

#     print('Accuracy:', (correct / total).item())
    
    
    
    pass
