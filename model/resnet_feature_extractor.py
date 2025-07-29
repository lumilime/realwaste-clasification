import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load model ResNet50
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

def extract_features(img_tensor):
    with torch.no_grad():
        features = resnet(img_tensor.unsqueeze(0))
    return features.view(-1).numpy()
