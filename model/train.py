import torch
import torch.nn as nn
import time
import copy
from tqdm.notebook import tqdm

from model.train_config import TrainConfig, create_train_config

CHECKPOINT_FILE_NAME = "iris_recognition_trained_model.pt"


def train_model(train_config: TrainConfig,
                model: nn.Module,
                criterion,
                optimizer,
                num_epochs: int = 50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    accuracies = {
        "train": [],
        "val": []
    }

    for _ in tqdm(range(num_epochs), desc="Training epoch"):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in train_config.loaders[phase]:
                inputs = inputs.to(train_config.device)
                labels = labels.to(train_config.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / train_config.data_sizes[phase]
            epoch_acc = running_corrects.double() / train_config.data_sizes[
                phase]

            accuracies[phase].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'
                  .format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:}m "
          f"{time_elapsed % 60.2}s")
    print(f"Best val Acc: {best_acc:.4}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, accuracies


def run():
    train_config = create_train_config()
    trained_model, accuracies = train_model(
        train_config=train_config,
        model=train_config.model,
        optimizer=train_config.optimizer,
        criterion=train_config.criterion
    )

    checkpoint_dict = {
        "model_state_dict": trained_model.state_dict(),
        "optimizer_state_dict": train_config.optimizer.state_dict(),
        "train_accuracies": accuracies["train"],
        "validation_accuracies": accuracies["val"],
        "classes": train_config.class_names
    }

    torch.save(checkpoint_dict, CHECKPOINT_FILE_NAME)
