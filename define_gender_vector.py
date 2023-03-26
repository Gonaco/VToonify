import os
from pathlib import Path
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import math
import cv2
import dlib
import torch
from torchvision import transforms
import torch.nn.functional as F
# from tqdm import tqdm
# from model.vtoonify import VToonify
from model.vtoonify_sum import VToonifySum
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_video_crop_parameter, tensor2cv2

from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


def landmarking(args, landmarkpredictor, frame):
    # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
    # Centered on the eyes, we crop the image to almost 400x400 (based on args['padding]).
    if args['scale_image']:
        paras = get_video_crop_parameter(frame, landmarkpredictor, args['padding'])
        if paras is not None:
            h,w,top,bottom,left,right,scale = paras
            H, W = int(bottom-top), int(right-left)
            # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
            if scale <= 0.75:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
            frame = cv2.resize(frame, (w, h))[top:bottom, left:right]
    return frame

def pre_processingImage(args, filename, basename, landmarkpredictor):

    cropname = os.path.join(args['output_path'], basename + '_input.jpg')
    savename = os.path.join(args['output_path'], basename + '_vtoonify_' +  args['backbone'][0] + '.jpg')
    sum_savename = os.path.join(args['output_path'], basename + '_vtoonify_SUM_' +  args['backbone'][0] + '.jpg')

    frame = cv2.imread(filename)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame = landmarking(args, landmarkpredictor, frame)

    return cropname, savename, sum_savename, frame


def processingStyle(device, frame, landmarkpredictor):
            I = align_face(frame, landmarkpredictor)
            if I is None:
                raise ValueError
            I = transform(I).unsqueeze(dim=0).to(device)

            s_w = pspencoder(I)
            # if previous_embedding is not None:
            #     vector_encoded = (vector_encoded + previous_embedding)/2  # SUM OPERATION? HERE?
            s_w = vtoonify.zplus2wplus(s_w)
            if vtoonify.backbone == 'dualstylegan':
                if args['color_transfer']:
                    s_w = exstyle
                else:
                    s_w[:,:7] = exstyle[:,:7]

            x = transform(frame).unsqueeze(dim=0).to(device)
            # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
            # followed by downsampling the parsing maps
            x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], scale_factor=0.5, recompute_scale_factor=False).detach()
            # we give parsing maps lower weight (1/16)
            inputs = torch.cat((x, x_p/16.), dim=1)

            return s_w, inputs


def concatenateTensors():
    current_size = embeddings_buffer[-1].size()
    current_dim = current_size[2]
    result = []
    for t in embeddings_buffer:
        d = current_dim - t.size()[2]
        if d < 0:
            t = t.narrow(2,0,current_size[2])
            t = t.narrow(3,0,current_size[3])
            result.append(t)
        # elif (d % 2) == 0:
        #     result.append(F.pad(input=t, pad=(int(d/2), int(d/2), int(d/2), int(d/2), 0, 0, 0, 0), mode='constant', value=0))
        else:
            result.append(F.pad(input=t,
                                pad=(math.ceil((current_size[3] - t.size()[3])/2),
                                     math.floor((current_size[3] - t.size()[3])/2),
                                     math.ceil((current_size[2] - t.size()[2])/2),
                                     math.floor((current_size[2] - t.size()[2])/2),
                                     0, 0, 0, 0),
                                mode='constant', value=0))

    add_embedding = torch.stack(result)
    return add_embedding


def lookForSum():
    add_embedding = concatenateTensors()
    sum_embedding = torch.mean(add_embedding, 0)
    return sum_embedding


if __name__ == "__main__":

    # PROCESSING THE INPUT VALUES

    device = "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])

    args = {
        # 'content' : '/Users/daniel/projects/matriu-id/espai-ideal/elface/Elface Model/options/VToonify/data/gender_vector/archive/Training/female/',
        'content': '/Users/daniel/projects/matriu-id/espai-ideal/elface/Elface Model/options/VToonify/data/celeba',
        'style_id': 26,
        'style_degree': 0,
        'ckpt': './checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt',
        'output_path': './output/gender_vector_test/',
        'style_encoder_path': './checkpoint/encoder.pt',
        'exstyle_path': os.path.join(
            os.path.dirname('./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt'),
            'exstyle_code.npy'
        ),
        'faceparsing_path': './checkpoint/faceparsing.pth',
        'backbone': 'dualstylegan',
        'padding': [600, 600, 600, 600],
        'scale_image': True,
        'color_transfer': False,
    }
    print('*'*98)

    vtoonify = VToonifySum(backbone = args['backbone'])
    vtoonify.load_state_dict(torch.load(args['ckpt'], map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args['faceparsing_path'], map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    modelname = './checkpoint/shape_predictor_68_face_landmarks.dat'
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data)
    landmarkpredictor = dlib.shape_predictor(modelname)

    pspencoder = load_psp_standalone(args['style_encoder_path'], device)

    if args['backbone'] == 'dualstylegan':
        exstyles = np.load(args['exstyle_path'], allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[args['style_id']]
        exstyle = torch.tensor(exstyles[stylename]).to(device)
        with torch.no_grad():
            exstyle = vtoonify.zplus2wplus(exstyle)

    male_embeddings_buffer = []
    female_embeddings_buffer = []

    print('Load models successfully!')

    with torch.no_grad():
        # dataset = Path(args['content']).glob('*')
        celeba = CelebA(args['content'], transform=transform, download=True)
        # dataset = DataLoader(celeba)
        # for entry in dataset:
        for n_e, entry in enumerate(celeba):

            try:

                filename = args['content'] + "/celeba/img_align_celeba/" + celeba.filename[n_e]
                basename = os.path.basename(filename).split('.')[0]

                Path(args['output_path']).mkdir(parents=True, exist_ok=True)  # Creates the output folder in case it does not exists

                print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args['backbone'][0])

                scale = 1
                kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])

                male = entry[1][20]

                # PROCESSING THE IMAGES
                cropname, savename, sum_savename, frame = pre_processingImage(args, filename, basename, landmarkpredictor)

                # cropname = os.path.join(args['output_path'], basename + '_input.jpg')
                # savename = os.path.join(args['output_path'], basename + '_vtoonify_' +  args['backbone'][0] + '.jpg')
                # sum_savename = os.path.join(args['output_path'], basename + '_vtoonify_SUM_' +  args['backbone'][0] + '.jpg')

                # frame = landmarking(args, landmarkpredictor, entry[0])

                # frame = entry[0]
                # landmarkpredictor = celeba.landmarks_align[n_e]

                s_w, inputs = processingStyle(device, frame, landmarkpredictor)

                out, skip, encoder_features, adastyles = vtoonify(inputs,
                                                                  s_w.repeat(inputs.size(0), 1, 1),
                                                                  d_s = args['style_degree'],
                                                                  return_feat=True)

                if male == 1:
                    male_embeddings_buffer.append(out)
                else:
                    female_embeddings_buffer.append(out)

                # d_s has no effect when backbone is toonify
                y_tilde = vtoonify((inputs, out,skip, encoder_features, adastyles), s_w.repeat(inputs.size(0), 1, 1), d_s = args['style_degree'], just_decoder=True)
                y_tilde = torch.clamp(y_tilde, -1, 1)

                cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                save_image(y_tilde[0].cpu(), savename)

                # if len(embeddings_buffer) > 1:
                #     sum_embedding = lookForSum()
                #     y_tilde_sum = vtoonify((inputs, sum_embedding, skip, encoder_features, adastyles), s_w.repeat(inputs.size(0), 1, 1), d_s = args['style_degree'], just_decoder=True)
                #     y_tilde_sum = torch.clamp(y_tilde, -1, 1)
                
                #     # cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                #     save_image(y_tilde_sum[0].cpu(), sum_savename)

            except ValueError:
                print("No face detected")

        print('Transfer style successfully!')