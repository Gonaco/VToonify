import argparse
import os
from pathlib import Path

import cv2
import torch
import dlib
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
if 'MAR-INF' in os.listdir():
    print('Inside Trochserve')
    os.environ['USING_TORCHSERVE'] = 'true'
if 'USING_TORCHSERVE' in os.environ:
    from align_all_parallel import align_face
    from bisnet_model import BiSeNet
    from vtoonify import VToonify
else:
    from model.bisenet.bisnet_model import BiSeNet
    from model.encoder.align_all_parallel import align_face
    from model.vtoonify import VToonify
from remove_background import remove_background
from torchvision import transforms
from util import get_video_crop_parameter
from util import load_psp_standalone
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context
from util import interpolate
import base64


def setup_parser():
    parser = argparse.ArgumentParser(description='Style Transfer')

    parser = argparse.ArgumentParser(description='Style Transfer')
    parser.add_argument('--content', type=str, default='./data', help='path of the folder with the content image/video')
    parser.add_argument('--style_id', type=int, default=26, help='the id of the style image')
    parser.add_argument('--style_degree', type=float, default=0.5, help='style degree for VToonify-D')
    parser.add_argument('--color_transfer', action='store_true', help='transfer the color of the style')
    parser.add_argument('--ckpt', type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help='path of the saved model')
    parser.add_argument('--output_path', type=str, default='./output/', help='path of the output images')
    parser.add_argument('--scale_image', action='store_true', help='resize and crop the image to best fit the model')
    parser.add_argument('--style_encoder_path', type=str, default='./checkpoint/encoder.pt', help='path of the style encoder')
    parser.add_argument('--exstyle_path', type=str, default=None, help='path of the extrinsic style code')
    parser.add_argument('--style_image_folder', type=str, default=None, help='path of the style image used instead of the exstyle path and style_id')
    parser.add_argument('--faceparsing_path', type=str, default='./checkpoint/faceparsing.pth', help='path of the face parsing model')
    parser.add_argument('--cpu', action='store_true', help='if true, only use cpu')
    parser.add_argument('--backbone', type=str, default='dualstylegan', help='dualstylegan | toonify')
    parser.add_argument('--padding', type=int, nargs=4, default=[200, 200, 200, 200], help='left, right, top, bottom paddings to the face center')
    parser.add_argument('--skip_vtoonify', action='store_true', help='Skip VToonify Styling and create final image only with generator model')
    parser.add_argument('--psp_style', type=int, nargs='*', help='Mix face and style after pSp encoding', default=[])
    parser.add_argument('--set_background', type=str, default=None,
                        help='Set the image background to "BLACK" or "WHITE". Default is None to leave original background. Options other than "BLACK" or "WHITE" fallback to "BLACK"' 
    )

    return parser

def parse(parser):
    opt = parser.parse_args()
    if opt.exstyle_path is None:
        opt.exstyle_path = os.path.join(os.path.dirname(opt.ckpt), 'exstyle_code.npy')
    return opt

class VToonifyHandler(BaseHandler): # for TorchServe  it need to inherit from BaseHandler
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.embeddings_buffer = []
        self.vtoonify_input_image_size = (256,256)
        self.default_sliding_window_size = 100
        self.padding = [200, 200, 200, 200]
        self.kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])
        self.default_FPS = 25
        self.default_duration_per_image = 1
        self.default_scale_image = False
        self.default_latent_mask = []
        self.default_style_degree = 0.1
        self.default_skip_vtoonify = True
        self.default_style_index = 0
        self.default_set_background = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """

        # From Torchserve on how to init. I guess manifest and other variables are necessary for propper working.
        # Content will be handled by Torchserve and include the necessary information
        self._context = context

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        # Read model serialize/pt file
        # serialized_file = self.manifest['model']['serializedFile']
        # model_pt_path = os.path.join(model_dir, serialized_file)

        # Load VToonify
        if 'USING_TORCHSERVE' in os.environ: # if we are realy in torchserve
            self.backbone = "dualstylegan"
            vtoonify_path = "vtoonify_s_d.pt"
            faceparsing_path = "faceparsing.pth"
            face_landmark_modelname = 'shape_predictor_68_face_landmarks.dat'
            style_encoder_path = "encoder.pt" # pSp encoder url https://drive.google.com/file/d/1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0/view
            exstyle_path = "exstyle_code.npy"
            self.exstyle_images = ['Peter_Mohrbacher-0048.jpeg', 'Peter_Mohrbacher-0054.jpeg']
        else:
            self.backbone = self.manifest['models']['backbone']
            vtoonify_path = self.manifest['models']['vtoonify']
            faceparsing_path = self.manifest['models']['faceparsing']
            face_landmark_modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
            style_encoder_path = self.manifest['models']['style_encoder']
            exstyle_path = self.manifest['models']['exstyle']
            exstyle_image_folder = self.manifest['models']['style_image_folder']
            self.exstyle_images = [f'{exstyle_image_folder}/{image_name}' for image_name in os.listdir(exstyle_image_folder)]

        self.sliding_window_size = self.default_sliding_window_size
        self.FPS = self.default_FPS
        self.duration_per_image = self.default_duration_per_image
        self.scale_image = self.default_scale_image
        self.latent_mask = self.default_latent_mask
        self.style_degree = self.default_style_degree
        self.skip_vtoonify = self.default_skip_vtoonify
        self.set_background = self.default_set_background
        self.style_index = self.default_style_index

        self.vtoonify = VToonify(backbone=self.backbone)
        self.vtoonify.load_state_dict(torch.load(vtoonify_path, map_location=lambda storage, loc: storage)['g_ema'])
        self.vtoonify.to(self.device)
        self.color_transfer = True

        self.parsingpredictor = BiSeNet(n_classes=19)
        self.parsingpredictor.load_state_dict(torch.load(faceparsing_path, map_location=lambda storage, loc: storage))
        self.parsingpredictor.to(self.device).eval()

        # Load Landmark Predictor model
        if not os.path.exists(face_landmark_modelname):
            import wget
            import bz2
            wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', face_landmark_modelname+'.bz2')
            zipfile = bz2.BZ2File(face_landmark_modelname+'.bz2')
            data = zipfile.read()
            open(face_landmark_modelname, 'wb').write(data)
        self.landmarkpredictor = dlib.shape_predictor(face_landmark_modelname)

        # Load pSp
        self.pspencoder = load_psp_standalone(style_encoder_path, self.device)

        # Load External Styles
        if self.backbone == 'dualstylegan':
            if self.exstyle_images:
                self.exstyles = []
                for exstyle_image_path in self.exstyle_images:
                    print(f'Loading Style image from {exstyle_image_path}')
                    exstyle_image = cv2.imread(exstyle_image_path)
                    with torch.no_grad():
                        exstyle_image = self.pre_processingImage(exstyle_image)
                        self.exstyle = self.encode_face_img(exstyle_image).to(self.device)
                    self.exstyles.append(self.exstyle)
                self.exstyle = self.exstyles[0]
            else:
                self.exstyles = np.load(self.manifest['models']['exstyle'], allow_pickle='TRUE').item()
                # stylename = list(self.exstyles.keys())[self.manifest['models']['style_id']]
                # self.exstyle = torch.tensor(self.exstyles[stylename]).to(self.device)
                # with torch.no_grad():
                #     self.exstyle = self.vtoonify.zplus2wplus(self.exstyle)  
                # self.exstyles = [self.exstyle]


        self.initialized = True

    def handle(self, data, context):
        input_item = data[0]['body']
        # Load image
        image_bytes = base64.b64decode(bytes(input_item['input_image'], encoding="utf8"))
        # Get arguments
        self.sliding_window_size = input_item.get('sliding_window_size', self.default_sliding_window_size)
        self.FPS = input_item.get('FPS', self.default_FPS)
        self.duration_per_image = input_item.get('duration_per_image', self.default_duration_per_image)
        self.scale_image = input_item.get('scale_image', self.default_scale_image)
        self.latent_mask = input_item.get('latent_mask', self.default_latent_mask)
        self.style_degree = input_item.get('style_degree', self.default_style_degree)
        self.style_index = input_item.get('style_index', self.default_style_index)
        self.skip_vtoonify = input_item.get('skip_vtoonify', self.default_skip_vtoonify)
        self.set_background = input_item.get('set_background', self.default_set_background)

        print(f"Handling image with parametters:")
        print(f"\tsliding_window_size: {self.sliding_window_size}")
        print(f"\tFPS: {self.FPS}")
        print(f"\tduration_per_image: {self.duration_per_image}")
        print(f"\tscale_image: {self.scale_image}")
        print(f"\tlatent_mask: {self.latent_mask}")
        print(f"\tstyle_degree: {self.style_degree}")
        print(f"\tstyle_index: {self.style_index}")
        print(f"\tskip_vtoonify: {self.skip_vtoonify}")
        print(f"\tset_background: {self.set_background}")

        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Preprocess Image
        with torch.no_grad():
            model_input = self.pre_processingImage(frame)

            # Encode Image
            s_w = self.encode_face_img(model_input)

            # Stylize pSp image
            print('Stylizing image with pSp')
            s_w = self.applyExstyle(s_w, self.exstyles[self.style_index], self.latent_mask)

            self.embeddings_buffer.append(torch.squeeze(s_w))
            self.window_slide()

            mean_s_w = self.pSpFeaturesBufferMean()

            animation_frames = self.generate_animation(
                [self.embeddings_buffer[-1], mean_s_w],
                FPS=self.FPS,
                duration_per_image=self.duration_per_image,
            )
            model_output = animation_frames

        return self.postprocess(model_output)

    def pre_processingImage(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
        # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
        if self.scale_image:
            paras = get_video_crop_parameter(frame, self.landmarkpredictor, self.padding)
            if paras is not None:
                h, w, top, bottom, left, right, scale = paras
                # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                if scale <= 0.75:
                    frame = cv2.sepFilter2D(frame, -1, self.kernel_1d, self.kernel_1d)
                if scale <= 0.375:
                    frame = cv2.sepFilter2D(frame, -1, self.kernel_1d, self.kernel_1d)
                frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

        frame = align_face(frame, self.landmarkpredictor)
        frame = self.transform(frame).unsqueeze(dim=0).to(self.device)
        if self.set_background:
            frame = remove_background(frame, white_background=self.set_background=="WHITE").to(self.device)

        return frame

    def s_w_to_stylized_image(self, s_w):
        # Update VToonify Frame to mean face
        print('Decoding mean image')
        frame = self.decodeFeaturesToImg(s_w)

        if self.skip_vtoonify:
            return frame

        print('Using VToonify to stylize image')
        # Resize frame to save memory
        frame = cv2.resize(frame, self.vtoonify_input_image_size)

        vtoonfy_output_image = self.apply_vtoonify(frame, s_w)
        return vtoonfy_output_image

    def encode_face_img(self, face_image):
        s_w = self.pspencoder(face_image)

        return s_w

    @staticmethod
    def applyExstyle(s_w, exstyle, latent_mask):
        if exstyle is None:
            print('No exstyle, skipping pSp styling')
            return s_w

        for i in latent_mask:
            s_w[:, i] = exstyle[:, i]

        return s_w

    def window_slide(self):
        if len(self.embeddings_buffer) > self.sliding_window_size:
            self.embeddings_buffer.pop(0)

    def pSpFeaturesBufferMean(self):
        if len(self.embeddings_buffer) > 1:
            all_latents = torch.stack(self.embeddings_buffer, dim=0)
            s_w = torch.mean(all_latents, dim=0)
        else:
            s_w = self.embeddings_buffer[0]
        s_w = s_w.unsqueeze(0)

        return s_w

    def generate_animation(self, encodings, FPS=25, duration_per_image=1):
        encodings = [encoding.squeeze() for encoding in encodings]
        animation_frames = []

        for s_w in tqdm(interpolate(
                latents_list=encodings, duration_list=[duration_per_image]*len(encodings),
                interpolation_type="linear",
                loop=False,
                FPS=FPS,
            ), desc='Generating morphing animation'):
            animation_frame = self.s_w_to_stylized_image(s_w)
            animation_frames.append(animation_frame)

        return animation_frames

    def decodeFeaturesToImg(self, s_w):
        frame_tensor, _ = self.vtoonify.generator.generator([s_w], input_is_latent=True, randomize_noise=False)
        # frame_tensor, _ = self.vtoonify.generator([s_w], s_w, input_is_latent=True, randomize_noise=True, use_res=False)
        frame = ((frame_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5)
        # frame = frame_tensor.detach().cpu().numpy().astype(np.uint8)
        # clip image to remove color spots on white background
        frame[frame < 0] = 0
        frame[frame > 255] = 255

        return frame.astype(np.uint8)

    def apply_vtoonify(self, frame, s_w):
        # Compute VToonify Features
        s_w, inputs = self.processingStyle(frame, s_w)

        # Process Image with VToonify
        y_tilde = self.vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=self.style_degree)
        y_tilde = torch.clamp(y_tilde, -1, 1)

        # Save Output Image
        output_image = self.normalize_image(y_tilde[0].cpu())

        return output_image

    def processingStyle(self, frame, s_w):
        s_w = self.vtoonify.zplus2wplus(s_w)
        if self.vtoonify.backbone == 'dualstylegan':
            if self.color_transfer:
                s_w = self.exstyle
            else:
                s_w[:, :7] = self.exstyle[:, :7]

        x = self.transform(frame).unsqueeze(dim=0).to(self.device)
        # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
        # followed by downsampling the parsing maps
        x_p = F.interpolate(
            self.parsingpredictor(2*(F.interpolate(
                x,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)))[0],
            scale_factor=0.5,
            recompute_scale_factor=False).detach()
        # we give parsing maps lower weight (1/16)
        inputs = torch.cat((x, x_p/16.), dim=1)

        return s_w, inputs

    @staticmethod
    def normalize_image(img):
        tmp = ((img.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        return tmp

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        
        encoded_frames = [self.image_to_base64(image) for image in inference_output]
        postprocess_output = {'video_frames': encoded_frames}
        # postprocess_output = {'video_frames': [encoded_frames[0], encoded_frames[-1]]}
        return [postprocess_output]

    @staticmethod
    def image_to_base64(image):
        print('Encoding image')
        _, im_arr = cv2.imencode('.JPEG', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])  # im_arr: image in Numpy one-dim array format.
        byte_arr = im_arr.tobytes()
        return base64.b64encode(byte_arr).decode('utf-8')

if __name__ == '__main__':
    parser = setup_parser()
    args = parse(parser)
    print('Loaded arguments')
    for name, value in sorted(vars(args).items()):
        print('%s: %s' % (str(name), str(value)))
    print('*'*98)

    device = 'cpu' if args.cpu else 'cuda'
    model_dir = "pretrained_models"
    model_name = "psp_ffhq_encode.pt"
    manifest = {
        'models': {
            'vtoonify': args.ckpt,
            'faceparsing': args.faceparsing_path,
            'style_encoder': args.style_encoder_path,
            'backbone': args.backbone,
            'exstyle': args.exstyle_path,
            'style_id': args.style_id,
            'style_image_folder': args.style_image_folder,
        }
    }
    context = Context(model_dir=model_dir, model_name="vtoonify", manifest=manifest,batch_size=1,gpu=0,mms_version="1.0.0")

    vtoonify_handler = VToonifyHandler()
    vtoonify_handler.initialize(context)

    print('Loaded models successfully!')

    Path(args.output_path).mkdir(parents=True, exist_ok=True)  # Creates the output folder in case it does not exists
    for file in Path(args.content).glob('*'):
        filename = args.content + '/' + file.name
        basename = os.path.basename(filename).split('.')[0]

        with open(filename, "rb") as image:
            f = image.read()
            input_image = bytearray(f)

        print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])

        model_response = vtoonify_handler.handle([{'body':{
            "input_image": base64.b64encode(input_image).decode('utf-8'),
            "FPS": 2,
            "duration_per_image": 2,
            "scale_image": args.scale_image,
            "latent_mask": args.psp_style,
            "style_degree": args.style_degree,
            "skip_vtoonify": args.skip_vtoonify,
            "style_index": args.style_id,
            "set_background": args.set_background,
            }},
        ], context)

        test_output_path = args.output_path + '/' + basename
        os.makedirs(test_output_path, exist_ok=True)
        for i, frame_base64 in enumerate(model_response[0]['video_frames']):
            frame_bytes = base64.b64decode(frame_base64)
            frame_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            frame_path = f'{test_output_path}/{i}.jpeg'
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) # save only last frame for test
