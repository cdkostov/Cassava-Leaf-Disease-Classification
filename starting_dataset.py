import torch
from PIL import Image
import torchvision.transforms as transforms

class StartingDataset(torch.utils.data.Dataset):

    path = r'C:\Users\Chris\AppData\Local\Programs\Python\Python39\Scripts\ACMAI\train_images\\'
    transform = transforms.Compose([transforms.PILToTensor( )])

    def __init__(self, items, labels):
        self.items = items.to_numpy()
        self.labels = labels.to_numpy()

    def __getitem__(self, index):
        item = self.items[index]
        label = self.labels[index]

        temp_path = StartingDataset.path + self.items[index]
        image = Image.open(temp_path)
        image_tensor = StartingDataset.transform(image)
        image_tensor = image_tensor.to(torch.float32)
        return image_tensor, label

    def __len__(self):
        return len(self.labels)
