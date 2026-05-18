# In questo file vorrei testare il comportamento di ErfNet e produrre una prima anomaly map, per capire se è
# possibile identificare le anomalie in questo modo Per fare questo, prendo un'immagine di test, la passo attraverso
# il modello e prendo i logits (output prima della softmax) per ogni pixel. Poi, confronto questi logits con quelli
# di un'immagine normale (senza anomalie) per vedere se ci sono differenze significative.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from erfnet import ERFNet

# Load the pre-trained ERFNet model
model = ERFNet(num_classes=20)  # Adjust num_classes as needed


def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
        else:
            own_state[name].copy_(param)
    return model


model = load_my_state_dict(
    model,
    torch.load("../trained_models/erfnet_pretrained.pth",
               map_location=torch.device("cpu"))
)

model.eval()

# Define a transformation for the input image
input_transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_logits(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    input_tensor = input_transform(image).unsqueeze(0)  # Add batch dimension
    # Get the logits from the model
    with torch.no_grad():
        logits = model(input_tensor)
        print(f"Logits shape: {logits.shape}")
    return logits.squeeze(0)  # Remove batch dimension


# Example usage
normal_image_path = "/Users/andrealops/Downloads/Validation_Dataset/fs_static/images/2.jpg"
anomalous_image_path = "/Users/andrealops/Downloads/Validation_Dataset/RoadAnomaly21/images/8.png"
normal_logits = get_logits(normal_image_path)
anomalous_logits = get_logits(anomalous_image_path)

print(anomalous_logits.shape)
print(anomalous_logits.min())
print(anomalous_logits.max())


# Compute the max logit score on the normal and anomalous images    
normal_max_logit = normal_logits.max(dim=0)[0]
anomalous_max_logit = anomalous_logits.max(dim=0)[0]

# Ora creiamo un anomaly map con MSP
anomalous_MSP = torch.softmax(anomalous_logits, dim=0).max(dim=0)[0]  # Max Softmax Probability
anomaly_score = 1 - anomalous_MSP  # Anomaly score is 1 - MSP

# Compute max entropy for anomalous image and normal image
anomalous_entropy = -torch.sum(torch.softmax(anomalous_logits, dim=0) * torch.log(torch.softmax(anomalous_logits, dim=0) + 1e-8), dim=0)
normal_entropy= -torch.sum(torch.softmax(normal_logits, dim=0) * torch.log(torch.softmax(normal_logits, dim=0) + 1e-8), dim=0)

# Visualize the anomaly score map
plt.imshow(anomaly_score.cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title("Anomaly Score Map (1 - MSP)")
plt.show()

# Visualize the max logit maps
plt.imshow(anomalous_max_logit.cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title("Max Logit Map (Anomalous Image)")
plt.show()
plt.imshow(normal_max_logit.cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title("Max Logit Map (Normal Image)")
plt.show()

# Visualize the entropy maps
plt.imshow(anomalous_entropy.cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title("Entropy Map (Anomalous Image)")
plt.show()
plt.imshow(normal_entropy.cpu().numpy(), cmap='hot')
plt.colorbar()
plt.title("Entropy Map (Normal Image)")
plt.show()

