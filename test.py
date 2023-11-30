import requests
from PIL import Image
from io import BytesIO
import base64
from time import time
import os
import cv2
import numpy as np

def send_image_and_decode_response(image_path):
    # Load image from file
    with open(image_path, 'rb') as f:
        image_bytes = f.read()

    # Send image to the endpoint
    i_t = time()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    payload={
        'input_image': encoded_image,
        'FPS': 25,
        'duration_per_image': 1,
        'scale_image': False,
        'skip_vtoonify': True,
        'latent_mask': []
    }
    response = requests.post('http://127.0.0.1:8080/predictions/vToonify', json=payload)
    print(f'Response took {time() - i_t} seconds')

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return None

    # Parse the response
    response_json = response.json()

    # Decode the 'video_frames' from base64 strings to images
    video_frames = []
    for frame_base64 in response_json['video_frames']:
        frame_bytes = base64.b64decode(frame_base64)
        frame_array = np.frombuffer(frame_bytes, np.uint8)
        frame_image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
        video_frames.append(frame_image)

    # Calculate and print the size of the response in MB
    response_size_bytes = len(response.content)
    response_size_mb = response_size_bytes / (1024 * 1024)
    print(f"Size of the response: {response_size_mb} MB")

    return video_frames

if __name__ == "__main__":
    # image_path = '/home/carles/repos/matriu.id/ideal/Datasets/sorolla-test-faces/minimum-subset/CFD-AM-229-224-N.jpg'
    image_path = '/home/carles/repos/matriu.id/ideal/Datasets/pretty-faces/MiamiModels_crops/straight_faces_subset/men-faces00008.jpg'
    # image_path = '/home/carles/repos/matriu.id/ideal/Datasets/pretty-faces/MiamiModels_crops/straight_faces_subset/new-woman-faces00005.jpg'
    # image_path = '/home/carles/repos/matriu.id/ideal/Datasets/pretty-faces/MiamiModels_crops/straight_faces_subset/men-faces00014.jpg'
    # image_path = '/home/carles/repos/matriu.id/ideal/Datasets/pretty-faces/MiamiModels_crops/straight_faces_subset/new-woman-faces00001.jpg'
    video_frames = send_image_and_decode_response(image_path)

    test_output_path = './test'
    os.makedirs(test_output_path, exist_ok=True )
    for i, image in enumerate(video_frames):
        save_path = f'{test_output_path}/{i}.jpeg'
        cv2.imwrite(save_path, image)

    
