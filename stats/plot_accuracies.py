import matplotlib.pyplot as plt
import numpy as np
import torch

NUM_EPOCHS = 100

model_dict = torch.load("../model/iris_recognition_trained_model.pt",
                        map_location=torch.device('cpu'))
train_accuracies = model_dict["train_accuracies"]
val_accuracies = model_dict["validation_accuracies"]

xs = np.linspace(0, NUM_EPOCHS, NUM_EPOCHS)

plt.title("Model accuracy during training")
plt.xlim(0, NUM_EPOCHS)
plt.ylim()
plt.plot(xs, train_accuracies, label=f"Train, max: {max(train_accuracies):.2%}")
plt.plot(xs, val_accuracies, label=f"Validation, max: {max(val_accuracies):.2%}")
plt.legend(loc="lower right")
plt.savefig("model_accuracies.png")
