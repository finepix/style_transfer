import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave


# util function to load masks
def load_mask(mask_path, shape):
    mask = imread(mask_path, mode="L") # Grayscale mask load
    width, height, _ = shape
    mask = imresize(mask, (width, height), interp='bicubic').astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    return mask


# util function to apply mask to generated image
def mask_content(content, generated, mask):
    width, height, channels = generated.shape

    for i in range(width):
        for j in range(height):
            if mask[i, j] == 0.:
                generated[i, j, :] = content[i, j, :]

    return generated


IMAGE_SAVE_DIR = '/temp/smg/transfer/'

def mask(
        content_image,
        generated_image,
        content_mask
):
    '''
    
    :param content_image: 内容图
    :param generated_image: 生成图
    :param content_mask: 遮罩
    :return:  图片保存地址
    '''
    image_path = IMAGE_SAVE_DIR + os.path.basename(content_image).split('.')[0]+ "_masked.png"
    print image_path
    generated_image = imread(generated_image, mode="RGB")
    img_width, img_height, channels = generated_image.shape
    content_image = imread(content_image, mode='RGB')
    content_image = imresize(content_image, (img_width, img_height), interp='bicubic')

    mask = load_mask(content_mask, generated_image.shape)
    img = mask_content(content_image, generated_image, mask)
    imsave(image_path, img)
    print("Image saved at path : %s" % image_path)
    return image_path


if __name__ == '__main__':
    mask('images/inputs/content/ancient_city.jpg','path_to_generated_image','images/inputs/mask/m1.jpg')







