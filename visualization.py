
def visualization(x, mask):
    '''
    visualization for semantic segmentation
    :param x: pytorch tensor [1, 3, m, n]    0<=x<=1 so you need to * 255 to visualize it
    :param mask: pytorch tensor [1,1, m, n],   values are 0 or 1, is the mask.
    :return: a PIL image which can directly image.show
    '''