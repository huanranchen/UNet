import torchvision
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader


def get_loader(batch_size=32, ):
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_transforms = transforms.Compose([
        # add transforms here
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    ])
    dataset = VOCSegmentation(root='./', image_set='train',
                              download=False,
                              transform=test_transforms,
                              target_transform=test_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dataset = VOCSegmentation(root='./', image_set='val',
                              download=False,
                              transform=train_transforms,
                              target_transform=test_transforms)
    v = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, v


if __name__ == '__main__':
    loader = get_loader(batch_size=1)

    for x, y in loader:
        print(x, y)
        assert False
