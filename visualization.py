import torch
from PIL import Image
import numpy as np


def visualization(x, mask, color = 255, weight = 0):
    '''
    visualization for semantic segmentation
    :param x: pytorch tensor [1, 3, m, n]    0<=x<=1 so you need to * 255 to visualize it
    :param mask: pytorch tensor [1,1, m, n],   values are 0 or 1, is the mask.
    :return: a PIL image which can directly image.show
    '''
    mask_tensor = mask * color / 255
    non_masked = (1-mask) * x     #没mask的地方有数值
    masked = mask * x*weight + mask_tensor* mask*(1-weight)               # mask的地方有数值
    x = non_masked+masked
    x = x.permute(0, 2, 3, 1)
    x = x.squeeze().numpy()
    x *= 255
    x = np.uint8(x)
    x = Image.fromarray(x)
    x.show()


if __name__ == '__main__':
    from UNet import UNet
    # model = UNet()
    # model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

    from data import get_loader

    loader, _ = get_loader(batch_size=1)

    for x, y in loader:
        visualization(x, y)
        assert False
