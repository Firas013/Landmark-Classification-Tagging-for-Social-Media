import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 50, dropout: float = 0.7) -> None:
        super().__init__()
    
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv_bn1 = nn.BatchNorm2d(8)
        
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv_bn2 = nn.BatchNorm2d(16)
        
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv_bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv_bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv_bn5 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adjusted input size for the fully connected layer
        self.fc1 = nn.Linear(128 * 7 * 7, 1024, bias=True)  
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 256, bias=True)
        self.fc4 = nn.Linear(256, 128, bias=True)
        self.fc5 = nn.Linear(128, num_classes)
        
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv_bn1(self.conv1(x))))
        x = self.pool(F.relu(self.conv_bn2(self.conv2(x))))
        x = self.pool(F.relu(self.conv_bn3(self.conv3(x))))
        x = self.pool(F.relu(self.conv_bn4(self.conv4(x))))
        x = self.pool(F.relu(self.conv_bn5(self.conv5(x))))

        # Flatten image into vector, pass to FC layers
        x = x.view(x.size(0), -1)  
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        
        return x






        
        



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
