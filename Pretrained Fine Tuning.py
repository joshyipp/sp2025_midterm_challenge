import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to finetune.
################################################################################
import torchvision.models as models                                                                                                            
import timm

def get_model(config): #import model
    model = timm.create_model(
        'convnext_small.fb_in22k_ft_in1k', # ConvNeXt image classification model. Pretrained on ImageNet-22k and fine-tuned on ImageNet-1k (https://arxiv.org/abs/2201.03545)
        pretrained=True,
        num_classes=100 # Explicitly sets number of output classes to CIFAR-100 (100 classes).
    )
    return model

def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)
        
        ### TODO - Your code here
        optimizer.zero_grad()  # Zero the gradients

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        running_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) ### TODO -- inference
            loss = criterion(outputs, labels)    ### TODO -- loss calculation

            running_loss += loss.item()   ### SOLUTION -- add loss from this sample
            _, predicted = torch.max(outputs, 1)   ### SOLUTION -- predict the class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():
    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.

    CONFIG = {
        "model": "efficientnet-pretrained",   # Change name when using a different model
        "batch_size": 64, # run batch size finder to find optimal batch size (i don't think the default value is ever used...)
        "learning_rate": 1e-6,
        "epochs": 50,  # Train for longer in a real scenario #10 was fine
        "num_workers": 6, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    # Temporarily define a transform without normalization
    transform_temp = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load dataset without normalization to instantiate dynamic normalization (based on dataset ingested)
    temp_trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True,
                                                download=True, transform=transform_temp)
    temp_loader = torch.utils.data.DataLoader(temp_trainset, batch_size=100, shuffle=False, num_workers=CONFIG["num_workers"])

    # Calculate mean and std
    def compute_mean_std(loader):
        mean = 0.0
        std = 0.0
        total_images = 0

        for images, _ in loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)  
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_images += batch_samples

        mean /= total_images
        std /= total_images
        return mean, std

    mean, std = compute_mean_std(temp_loader)

    #https://github.com/1Konny/gradcam_plus_plus-pytorch/issues/8 explained why we have to upsample to 224 with resnet 18...
    from torchvision.transforms import RandAugment, ColorJitter, GaussianBlur, RandomErasing

    transform_train = transforms.Compose([
        transforms.Resize(224), # needed to adapt 32 x 32 images for particular pretrained models
        transforms.RandomCrop(224, padding=16),
        transforms.RandomHorizontalFlip(),                         # Flip for invariance
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Color/light variation
        transforms.RandomApply([GaussianBlur(kernel_size=3)], p=0.2),  # Occasional blur
        RandAugment(num_ops=2, magnitude=9),                        # Strong augmentations
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),         # Dynamic normalization
        transforms.RandomErasing(p=0.25)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),  # <--- Add this
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),  # <--- Add this
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    ############################################################################
    #       Data Loading
    ############################################################################
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))   ### TODO -- Calculate training set size
    val_size = len(trainset) - train_size     ### TODO -- Calculate validation set size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])  ### TODO -- split into training and validation sets

    trainset.dataset.transform = transform_train
    valset.dataset.transform = transform_val

    ### TODO -- define loaders and test set
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False)

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = get_model(CONFIG)
    model = model.to(CONFIG["device"])   # move it to target device
    print("\nModel summary:")
    print(f"{model}\n")
    
    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    
    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    from torch.optim.lr_scheduler import OneCycleLR
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # Define the loss function with label smoothing to prevent overconfidence in predictions
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ##########################################a##################################
    best_val_acc = 0.0
    early_stopping_patience = 5  # Stop if no improvement for 5 epochs
    early_stopping_counter = 0
    best_val_loss = float("inf")  # or use best_val_acc for accuracy-based stopping

    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"] # Log learning rate
        })

        # Save the best model (based on validation accuracy)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")
        else:
            early_stopping_counter += 1
            print(f"EarlyStopping counter: {early_stopping_counter} / {early_stopping_patience}")
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()

