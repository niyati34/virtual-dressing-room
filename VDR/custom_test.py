import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
import json

from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images


def load_image(path, size, is_mask=False):
    img = Image.open(path)
    img = img.resize(size, Image.NEAREST if is_mask else Image.BILINEAR)
    if is_mask:
        arr = np.array(img)
        arr = (arr >= 128).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ])
        return transform(img)  # [3, H, W]


def load_pose_json(json_path, image_size):
    with open(json_path, 'r') as f:
        pose_label = json.load(f)
        keypoints = pose_label['people'][0]['pose_keypoints_2d']
        keypoints = np.array(keypoints).reshape((-1, 3))[:, :2]
        canvas = Image.new('RGB', image_size, 'black')
        return transforms.ToTensor()(canvas), keypoints


def test_single(
    person_img_path, cloth_img_path, mask_path,
    parse_path, pose_json_path, save_path, opt
):
    device = torch.device("cpu")
    img_size = (opt.load_width, opt.load_height)

    # Load inputs
    img = load_image(person_img_path, img_size)
    cm = load_image(mask_path, img_size, is_mask=True)
    cloth = load_image(cloth_img_path, img_size)
    parse_agnostic = load_image(parse_path, img_size, is_mask=True).long()
    pose, pose_data = load_pose_json(pose_json_path, img_size)

    # Fake agnostic image = gray + head + lower body
    agnostic_img = img.clone()

    # Prep parse map
    parse_map = torch.zeros(20, opt.load_height, opt.load_width)
    parse_map.scatter_(0, parse_agnostic, 1.0)
    new_parse_map = torch.zeros(opt.semantic_nc, opt.load_height, opt.load_width)
    label_groups = {
        0: [0, 10], 1: [1, 2], 2: [4, 13], 3: [5, 6, 7],
        4: [9, 12], 5: [14], 6: [15], 7: [16], 8: [17],
        9: [18], 10: [19], 11: [8], 12: [3, 11]
    }
    for i in range(opt.semantic_nc):
        for j in label_groups[i]:
            new_parse_map[i] += parse_map[j]
    parse_agnostic = new_parse_map.unsqueeze(0)

    # Load network
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc).to(device).eval()
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3).to(device).eval()
    alias = ALIASGenerator(opt, input_nc=9).to(device).eval()

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    # Begin forward pass
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')

    pose = pose.unsqueeze(0)
    cloth = cloth.unsqueeze(0)
    cm = cm.unsqueeze(0)
    img = img.unsqueeze(0)
    agnostic_img = agnostic_img.unsqueeze(0)

    # SEG
    inputs = torch.cat([
        F.interpolate(cm, (256, 192), mode='bilinear'),
        F.interpolate(cloth * cm, (256, 192), mode='bilinear'),
        F.interpolate(parse_agnostic, (256, 192), mode='bilinear'),
        F.interpolate(pose, (256, 192), mode='bilinear'),
        gen_noise(cm.size())
    ], dim=1)

    out_seg = seg(inputs)
    parse_pred = gauss(up(out_seg)).argmax(dim=1)[:, None]
    parse_onehot = torch.zeros(parse_pred.size(0), 13, *parse_pred.shape[2:])
    parse_onehot.scatter_(1, parse_pred, 1.0)

    label_remap = {
        0: [0], 1: [2, 4, 7, 8, 9, 10, 11], 2: [3],
        3: [1], 4: [5], 5: [6], 6: [12]
    }
    parse_remap = torch.zeros(parse_pred.size(0), 7, *parse_pred.shape[2:])
    for i in label_remap:
        for j in label_remap[i]:
            parse_remap[:, i] += parse_onehot[:, j]

    # GMM
    gmm_input = torch.cat([
        F.interpolate(parse_remap[:, 2:3], (256, 192), mode='nearest'),
        F.interpolate(pose, (256, 192), mode='nearest'),
        F.interpolate(agnostic_img, (256, 192), mode='nearest')
    ], dim=1)

    _, grid = gmm(gmm_input, F.interpolate(cloth, (256, 192), mode='nearest'))
    warped_c = F.grid_sample(cloth, grid, padding_mode='border')
    warped_cm = F.grid_sample(cm, grid, padding_mode='border')

    # ALIAS
    misalign_mask = parse_remap[:, 2:3] - warped_cm
    misalign_mask[misalign_mask < 0] = 0
    parse_div = torch.cat((parse_remap, misalign_mask), dim=1)
    parse_div[:, 2:3] -= misalign_mask

    output = alias(torch.cat((agnostic_img, pose, warped_c), dim=1), parse_remap, parse_div, misalign_mask)
    save_images(output, ['tryon_result'], save_path)

    print(f"✅ Output saved at {save_path}/tryon_result.jpg")


# Example call
class Opt:
    name = 'custom_run'
    batch_size = 1
    workers = 1
    load_height = 1024
    load_width = 768
    shuffle = False
    dataset_dir = './'
    dataset_mode = 'test'
    dataset_list = ''
    checkpoint_dir = './checkpoints/'
    save_dir = './results/'
    display_freq = 1
    seg_checkpoint = 'seg_final.pth'
    gmm_checkpoint = 'gmm_final.pth'
    alias_checkpoint = 'alias_final.pth'
    semantic_nc = 13
    init_type = 'xavier'
    init_variance = 0.02
    grid_size = 5
    norm_G = 'spectralaliasinstance'
    ngf = 64
    num_upsampling_layers = 'most'

if __name__ == '__main__':
    opt = Opt()
    test_single(
        person_img_path='viton_resize/test/image/person.jpg',
        cloth_img_path='viton_resize/test/cloth/cloth.jpg',
        mask_path='viton_resize/test/cloth-mask/cloth.png',
        parse_path='viton_resize/test/image-parse/person.png',
        pose_json_path='viton_resize/test/openpose-json/person_keypoints.json',
        save_path='./results/custom_run',
        opt=opt
    )
