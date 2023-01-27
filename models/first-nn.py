import nn
import torch

class ImageNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 10),
        )    
        
        self.avg = nn.AdaptiveAvgPool2d((6,6))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.features(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return torch.softmax(x, dim=1)


model = ImageNet()

x = torch.randn(1, 3, 224, 224)
y = model(x)

print(y)
print(y.shape)
