import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob 

import cv2
import argparse
from models.archs.WaveNet_arch import WaveNet_S,WaveNet_T,WaveNet_B

parser = argparse.ArgumentParser(description='Demo Image Enhancement')
# ####LOL
parser.add_argument('--input_dir', default='/Users/dangjiachen/Desktop/LLIE/dataset/LOLdataset/eval15/low', type=str, help='Input images')
####5k
# parser.add_argument('--input_dir', default='../dataset/MITfivek/test/low', type=str, help='Input images')
#####SICE
# parser.add_argument('--input_dir', default='../dataset/SICE/test/low_resize', type=str, help='Input images')
#####VE-LOL-real
# parser.add_argument('--input_dir', default='../dataset/VE-LOL/test/low', type=str, help='Input images')
#####VE-LOL-syn
# parser.add_argument('--input_dir', default='../dataset/VE-LOL-L-Syn/VE-LOL-L-Syn-Low_test/', type=str, help='Input images')
#####LIME
# parser.add_argument('--input_dir', default='../dataset//No_ref/LIME', type=str, help='Input images')#bmp
#####DICM
# parser.add_argument('--input_dir', default='../dataset/No_ref/DICM', type=str, help='Input images')#jpg
#####VV
# parser.add_argument('--input_dir', default='../dataset/No_ref/VV', type=str, help='Input images')#jpg
#####MEF
# parser.add_argument('--input_dir', default='../dataset/No_ref/MEF', type=str, help='Input images')#png
#####NPE
# parser.add_argument('--input_dir', default='../dataset/No_ref/NPE', type=str, help='Input images')#png
#####dark face
# parser.add_argument('--input_dir', default='../dataset/Dark Face/image/', type=str, help='Input images')

parser.add_argument('--result_dir', default='../visual/LOL/', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='../checkpoints/SID/WaveNet_B_sid.pth', type=str,
                    help='Path to weights')

args = parser.parse_args()

def match_size(input_,mul=8):
        h, w = input_.shape[1], input_.shape[2]
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        input_ = F.pad(input_, ( 0,padw, 0, padh), 'reflect')
        return input_

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights,device):
    checkpoint = torch.load(weights, map_location=device)
    try:
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint['params'])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

inp_dir = args.input_dir
out_dir = args.result_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(#glob(os.path.join(inp_dir, '*.jpg')) +
                  glob(os.path.join(inp_dir, '*.jpg'))
                  #+ glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.png'))
                  + glob(os.path.join(inp_dir, '*.bmp')))

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights
print('==> Build the model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=WaveNet_B()
load_checkpoint(model, args.weights,device)
model.to(device)
model.eval()

print('enhancing images......')
index=0
gt=0
for file_ in files:
    img = Image.open(file_).convert('RGB')
    
    input_ = TF.to_tensor(img)
    C,H,W=input_.shape
    # input_=match_size(input_,mul=32)
    # input_ =transforms.CenterCrop((512,512))(input_)
    restored=input_.unsqueeze(0)
    with torch.no_grad():
        # for cpu
        # restored = model(restored)
        # for gpu
          retored = model(restored.cuda())

    # print(restored.shape)
    restored = torch.clamp(restored, 0, 1)
    # restored = restored[:, :, :H, :W]
    restored = restored[:, :, :, :]
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_)[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored)
    index += 1
    print('%d/%d' % (index, len(files)))

print(f"Files saved at {out_dir}")
print('finish !')

