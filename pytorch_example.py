import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn,save,load
import torch.optim as optim
import time


# load data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="data", download=True, 
                               train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# define image classifier model

class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# check for gpu and create instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(6):  # Train for 10 epochs
    train_total = 0
    train_correct = 0
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start_time = time.perf_counter()
    for images, labels in train_loader:
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
    if torch.cuda.is_available(): torch.cuda.synchronize()
    duration = time.perf_counter() - start_time
    throughput = train_total / duration # total images divided by seconds
    epoch_acc = 100 * train_correct / train_total
    print(f"Epoch {epoch} Throughput: {throughput:.2f} images/sec")
