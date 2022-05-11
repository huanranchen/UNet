import torchvision
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loader(batch_size = 32, ):
    train_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    dataset = VOCSegmentation(root = './', image_set = 'train',
                              download=False,
                              transform=train_transforms,
                              target_transform=train_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

if __name__ == '__main__':
    loader = get_loader(batch_size=1)

    for x, y in loader:
        print(x, y)
        assert False