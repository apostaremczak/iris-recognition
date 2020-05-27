import torch
import torch.nn as nn
import time
import copy

from train_config import TrainConfig, create_train_config

CHECKPOINT_FILE_NAME = "iris_recognition_trained_model.pt"


def train_model(train_config: TrainConfig,
                model: nn.Module,
                criterion,
                optimizer,
                scheduler,
                num_epochs: int = 50):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    accuracies = {
        "train": [],
        "val": []
    }

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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
            if phase == 'train':
                scheduler.step()

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
    print('Training complete in {:.0f}m {:.0f}s'
          .format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, accuracies


def run():
    train_config = create_train_config()
    model = train_config.get_model()
    optimizer = train_config.optimizer(model.parameters(),
                                       lr=0.001,
                                       momentum=0.9)
    scheduler = train_config.lr_scheduler(optimizer, step_size=7, gamma=0.1)

    trained_model, accuracies = train_model(
        train_config=train_config,
        model=model,
        optimizer=optimizer,
        criterion=train_config.criterion,
        scheduler=scheduler
    )

    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_accuracies": accuracies["train"],
        "validation_accuracies": accuracies["val"],
        "classes": train_config.class_names
    }, CHECKPOINT_FILE_NAME)
