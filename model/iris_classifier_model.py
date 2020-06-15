import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
from typing import List, Tuple

from model.user import User

# Data augmentation for training and validation
DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class IrisClassifier(nn.Module):
    CHECKPOINT_FILE_NAME = "iris_recognition_trained_model.pt"

    def __init__(self,
                 class_names: List[str] = None,
                 num_classes: int = 50,
                 load_from_checkpoint: bool = False,
                 acceptance_threshold: float = 0.98,
                 image_loader=default_loader,
                 checkpoint_file: str = CHECKPOINT_FILE_NAME):
        super().__init__()
        assert class_names is not None or load_from_checkpoint, \
            "Either load a model with predefined classes from a checkpoint, " \
            "or provide them up front"

        if class_names is not None:
            assert len(class_names) == num_classes, \
                "Number of classes must be equal to the length of class " \
                "names provided"
            self.class_names: List[str] = sorted(class_names)
            self.num_classes = len(self.class_names)
        else:
            self.num_classes = num_classes

        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        model = models.resnet50(pretrained=True)

        # Replace the last ResNet layer to match the desired number of classes
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)

        model = model.to(self.device)
        self.model: nn.Module = model

        if load_from_checkpoint:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            state_dict = checkpoint["model_state_dict"]

            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                # Local fix when loading model that was saved with DataParallel
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("model.", "")
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

            self.class_names = checkpoint["classes"]

        self.acceptance_threshold = acceptance_threshold
        self.image_loader = image_loader
        self.transform = DATA_TRANSFORMS["val"]

    def forward(self, x):
        return self.model(x)

    def classify_single_image(self,
                              normalized_image_path: str) -> Tuple[str, float]:
        """
        Predict a class and determine its probability
        """
        self.model.eval()
        image = self.image_loader(normalized_image_path)
        image = self.transform(image).float().to(self.device)
        image = image.unsqueeze_(0)

        with torch.set_grad_enabled(False):
            output = self.model(image)
            index = output.numpy().argmax()

            probabilities = nn.functional.softmax(output, dim=1).squeeze()

            predicted_class_probability = probabilities[index].tolist()

            if predicted_class_probability >= self.acceptance_threshold:
                predicted_class = self.class_names[index]
            else:
                predicted_class = User.UNKNOWN

        return predicted_class, predicted_class_probability
