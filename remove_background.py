from pymatting import estimate_alpha_knn
import numpy as np
import PIL.ImageOps
from PIL import ImageFilter
from rembg import remove
import cv2
from util import tensor2np_uint8
from util import np2tensor
from rembg.session_factory import new_session

session = new_session("u2net")

def remove_background(input_image, white_background):    
    print('Removing image background')
    np_image = tensor2np_uint8(input_image)
    mask = get_person_mask(np_image)

    mask = np.repeat(mask[:,:,np.newaxis], repeats=3, axis=2) # from (256,256) to (256,256,3)

    masked_image=np_image*mask

    # to make background white
    if white_background:
        bg = np.ones(np_image.shape)*255
        bg=bg*(1-mask)

        masked_image = masked_image + bg

    output_tensor = np2tensor(masked_image)

    return output_tensor

def get_person_mask(image):
    mask = remove(
        data=image,
        only_mask=True,
        session=session,
    )

    for _ in range(0):
        mask = mask.filter(ImageFilter.MinFilter(3))

    trimap = gen_trimap(mask)
    mask = alpha_mating(image, trimap)

    return mask

def gen_trimap(mask):
    mask = np.array(mask)
    mask[mask <= 128] = 0
    mask[mask > 128] = 255
    iterations = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = mask
    eroded = cv2.erode(mask, kernel, iterations=iterations)
    trimap = np.zeros(mask.shape)
    trimap.fill(128)
    trimap[eroded >= 205] = 255
    trimap[dilated <= 128] = 0
    # trimap = eroded
    return PIL.Image.fromarray(np.uint8(trimap))

def alpha_mating(image, trimap):
    image = np.array(image)/255.
    trimap = np.array(trimap)/255.
    # estimate alpha from image and trimap
    alpha = estimate_alpha_knn(image, trimap) # ~8sec total (masking + painting)

    return alpha
