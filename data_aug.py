from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def fixed_rotation(image, angles=[90, 180, 270]):
    """Rotate the image by fixed angles."""
    return [image.rotate(angle) for angle in angles]

def fixed_flip(image):
    """Flip the image horizontally and vertically."""
    horizontal_flip = ImageOps.mirror(image)
    vertical_flip = ImageOps.flip(image)
    return [horizontal_flip]

def fixed_brightness(image, factors=[0.99, 1.01]):
    """Adjust the brightness of the image by fixed factors."""
    image = image.convert('RGB')
    enhancer = ImageEnhance.Brightness(image)
    return [enhancer.enhance(factor) for factor in factors]

def fixed_contrast(image, factors=[0.99, 1.01]):
    """Adjust the contrast of the image by fixed factors."""
    image = image.convert('RGB')
    enhancer = ImageEnhance.Contrast(image)
    return [enhancer.enhance(factor) for factor in factors]

def fixed_saturation(image, factors=[0.99, 1.01]):
    """Adjust the saturation of the image by fixed factors."""
    image = image.convert('RGB')
    enhancer = ImageEnhance.Color(image)
    return [enhancer.enhance(factor) for factor in factors]

def fixed_hue(image, shifts=[0.01, -0.01]):
    """Shift the hue of the image by fixed amounts."""
    images = []
    for shift in shifts:
        img = image.convert('HSV')
        h, s, v = img.split()
        np_h = np.array(h, dtype=np.uint8)
        np_h = (np_h + int(shift * 255)) % 255
        h = Image.fromarray(np_h, 'L')
        img = Image.merge('HSV', (h, s, v)).convert('RGB')
        images.append(img)
    return images

def data_augmentation(org_data):
    augmented_data = {}
    for key in org_data:
        augmented_data[key] = {}
        for num in [2,3,4,5]:
            augmented_data[key][num] = []
            for sample in org_data[key][num]:

                new_imgs = [sample['img']]
                new_imgs.extend(fixed_rotation(sample['img']))
                new_imgs.extend(fixed_flip(sample['img']))
                new_imgs.extend(fixed_brightness(sample['img']))
                new_imgs.extend(fixed_contrast(sample['img']))
                new_imgs.extend(fixed_hue(sample['img']))

                augmented_data[key][num] += [
                    {
                    'key':sample['key'],
                    'url':sample['url'],
                    'img':new_img,
                    'true number': sample['true number']
                    } for new_img in new_imgs
                ]
    return augmented_data

