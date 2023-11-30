import requests
from time import time
import argparse
import base64

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="13.50.64.194", help="IP for the machine running the model")

def downloadTestImg():
    image_url = "https://www.physio-network.com/es/wp-content/profile-pics/mary.png"

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream = True)

    if r.status_code == 200:
        return r.content
    else:
        print('Image Couldn\'t be retreived')

def send_image_and_decode_response(model_url):
    test_image_bytes = downloadTestImg()
    payload={
        'input_image': base64.b64encode(test_image_bytes).decode('utf-8'),
        'FPS': 25,
        'duration_per_image': 1,
        'scale_image': False,
        'skip_vtoonify': True,
        'latent_mask': [10,11,12,13,14],
        'set_background': "BLACK",
        'style_index': 1,
    }
    i_t = time()
    response = requests.post(f'http://{model_url}:8080/predictions/vToonify', json=payload)
    print(f'Response took {time() - i_t} seconds')

    # Check if the request was successful
    assert response.status_code==200

    # Parse the response and check that there are frames in the response
    response_json = response.json()

    response_video_frames = response_json['video_frames']
    assert response_video_frames!=[]
    assert len(response_video_frames)>0

if __name__ == "__main__":
    args = parser.parse_args()

    print('Testing send image to model')
    video_frames = send_image_and_decode_response(args.url)
    print('Test passed')
