import torch
import torch.nn.functional as F
import cv2
import os
import imageio
import argparse
from Utils.SINet import SINet_ResNet50
from Utils.Dataloader import test_dataset

def numpy2tensor(numpy, device):
    return torch.from_numpy(numpy).cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str, default='./Model/SINet_4.pth')
parser.add_argument('--test_save', type=str, default='./Result/')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path, map_location=device))
model.eval()

folder_path = 'Dataset\TestDataset\CAMO'
orig_image = []
orig_image_path = []

for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path,filename)
    image = cv2.imread(image_path)
    orig_image_path.append(image_path)
    if image is not None:
        height,width,channels = image.shape
        orig_image.append((width,height))
        
for dataset in ['CAMO']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/'.format(dataset), testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        image, name = test_loader.load_data()
        image = image.cuda()
        _, cam = model(image)
        (width,height) = orig_image[img_count-1]
        cam = F.upsample(cam, size=(height,width), mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        imageio.imwrite(save_path + name, cam)
        print("Creating output for {}".format(orig_image_path[img_count-1]))
        img_count += 1
print("\n[INFO] Testing Complete")