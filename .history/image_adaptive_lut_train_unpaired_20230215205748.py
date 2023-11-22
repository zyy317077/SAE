import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import lpips
import torchvision.transforms as transforms
from torchvision.utils import save_image
from skimage.metrics._structural_similarity import \
    structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import \
    peak_signal_noise_ratio as compare_psnr
from os import listdir
from os.path import join
from lib.data import (get_Low_light_training_set, get_training_set,
                      is_image_file)
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd
from models_x import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch

import network
import utils
from utils import ext_transforms as et
from data.cityscapes import Cityscapes

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from, 0 starts from scratch, >0 starts from saved checkpoints")
parser.add_argument("--n_epochs", type=int, default=50, help="total number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="fiveK", help="name of the dataset")
# parser.add_argument("--input_color_space", type=str, default="sRGB", help="input color space: sRGB or XYZ")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr_GD", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_pixel", type=float, default=1000, help="content preservation weight: 1000 for sRGB input, 10 for XYZ input")
parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty weight in wgan-gp")
parser.add_argument("--lambda_smooth", type=float, default=1e-4, help="smooth regularization")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0, help="monotonicity regularization: 10 for sRGB input, 100 for XYZ input (slightly better)")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--output_dir", type=str, default="LUTs/unpaired/cityscapes", help="path to save model")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")

parser.add_argument("--ckpt", default="./checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", type=str,help="restore from checkpoint")
parser.add_argument("--data_root", type=str, default='./data/syn',help="path to Dataset")
parser.add_argument("--dataset", type=str, default='cityscapes',choices=['voc', 'cityscapes'], help='Name of dataset')
parser.add_argument("--loss_type", type=str, default='cross_entropy',choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
parser.add_argument("--num_classes", type=int, default=19,help="num classes (default: None)")
parser.add_argument("--crop_size", type=int, default=384)
parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
parser.add_argument("--separable_conv", action='store_true', default=False,
                    help="apply separable conv to decoder and aspp")
parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.1)")
parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
parser.add_argument("--continue_training", action='store_true', default=False)
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
opt = parser.parse_args()
# opt.output_dir = opt.output_dir + '_' + opt.input_color_space
# print(opt)
os.makedirs("saved_models/%s" % opt.output_dir, exist_ok=True)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.MSELoss()

# Initialize generator and discriminator
LUT0 = Generator3DLUT_identity()
LUT1 = Generator3DLUT_zero()
LUT2 = Generator3DLUT_zero()
#LUT3 = Generator3DLUT_zero()
#LUT4 = Generator3DLUT_zero()
classifier = Classifier_unpaired()
discriminator = Discriminator()
TV3 = TV_3D()

if cuda:
    LUT0 = LUT0.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    #LUT3 = LUT3.cuda()
    #LUT4 = LUT4.cuda()
    classifier = classifier.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    discriminator = discriminator.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

if opt.epoch != 0:
    # Load pretrained models
    LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.output_dir, opt.epoch))
    LUT0.load_state_dict(LUTs["0"])
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    #LUT3.load_state_dict(LUTs["3"])
    #LUT4.load_state_dict(LUTs["4"])
    classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.output_dir, opt.epoch)))
else:
    # Initialize weights
    classifier.apply(weights_init_normal_classifier)
    torch.nn.init.constant_(classifier.model[12].bias.data, 1.0)
    discriminator.apply(weights_init_normal_classifier)

# Set up model
model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
}
model = model_map[opt.model](num_classes=opt.num_classes, output_stride=opt.output_stride)
if opt.separable_conv and 'plus' in opt.model:
    network.convert_to_separable_conv(model.classifier)
utils.set_bn_momentum(model.backbone, momentum=0.01)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(classifier.parameters(), LUT0.parameters(),LUT1.parameters(),LUT2.parameters()), lr=opt.lr_GD, betas=(opt.b1, opt.b2)) #,LUT3.parameters(),LUT4.parameters()
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_GD, betas=(opt.b1, opt.b2))
optimizer = torch.optim.SGD(params=[
    {'params': model.backbone.parameters(), 'lr': 0.1 * opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr},], lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
if opt.ckpt is not None and os.path.isfile(opt.ckpt):
    checkpoint = torch.load(opt.ckpt, map_location=torch.device(device))
    # model.load_state_dict(checkpoint["model_state"],strict=False)
    model.load_state_dict(checkpoint,strict=False)
    # print(model)
    for k in model.parameters():
        k.requires_grad = True

    # model = nn.DataParallel(model)
    # model = model.eval()
    model = model.to(device).eval()
    
    if opt.continue_training:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        # scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_itrs = checkpoint["cur_itrs"]
        best_score = checkpoint['best_score']
        print("Training state restored from %s" % opt.ckpt)
    print("Model restored from %s" % opt.ckpt)
    # del checkpoint  # free memory
else:
    print("[!] Retrain")
    # model = nn.DataParallel(model)
    model.to(device)

# Set up criterion
# if opt.lr_policy == 'poly':
#     scheduler = utils.PolyLR(optimizer, opt.total_itrs, power=0.9)
# elif opt.lr_policy == 'step':
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

def get_dataset(opt):
    """ Dataset And Augmentation
    """
    if opt.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opt.crop_size, opt.crop_size)),
            #et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])

        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        # train_dst = Cityscapes(root=opt.data_root,
        #                        split='train', transform=train_transform)
        # val_dst = Cityscapes(root=opt.data_root,
        #                      split='val', transform=val_transform)
        train_dst = Cityscapes(root=opt.data_root + '/train',
                               train=True, transform=train_transform)
        val_dst = Cityscapes(root=opt.data_root,
                             split='val', transform=val_transform)
    # return train_dst
    return train_dst, val_dst
train_dst, val_dst = get_dataset(opt)
# train_dst = get_dataset(opt)
start_time = time.time()
train_loader = DataLoader(train_dst, batch_size=opt.batch_size, shuffle=True)
# val_loader = DataLoader(val_dst, batch_size=opt.batch_size, shuffle=True)
# print("Dataset: %s, Train set: %d, Val set: %d" %
#       (opt.dataset, len(train_dst), len(val_dst)))
# if opt.input_color_space == 'sRGB':
#     dataloader = DataLoader(
#         ImageDataset_sRGB_unpaired("./data/%s" % opt.dataset_name, mode="train"),
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=opt.n_cpu,
#     )
#
psnr_dataloader = DataLoader(
    ImageDataset_sRGB_unpaired("./data/%s" % opt.dataset_name,  mode="test"),
    batch_size=1,
    shuffle=False,
    # num_workers=1,
)

# elif opt.input_color_space == 'XYZ':
#     dataloader = DataLoader(
#         ImageDataset_XYZ_unpaired("./data/%s" % opt.dataset_name, mode="train"),
#         batch_size=opt.batch_size,
#         shuffle=True,
#         num_workers=opt.n_cpu,
#     )
#
#     psnr_dataloader = DataLoader(
#         ImageDataset_XYZ_unpaired("./data/%s" % opt.dataset_name,  mode="test"),
#         batch_size=1,
#         shuffle=False,
#         num_workers=1,
#     )

def calculate_psnr():
    classifier.eval()
    avg_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))
        fake_B, weights_norm = generator(real_A)
        save_image(torch.cat([real_A, fake_B], dim=0), './out/d-1.jpg')
        fake_B = torch.round(fake_B*255)
        real_B = torch.round(real_B*255)
        mse = criterion_pixelwise(fake_B, real_B)
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        avg_psnr += psnr

    return avg_psnr/ len(psnr_dataloader)


# def visualize_result(epoch):
#     """Saves a generated sample from the validation set"""
#     os.makedirs("images/LUTs/" +str(epoch), exist_ok=True)
#     for i, batch in enumerate(psnr_dataloader):
#         real_A = Variable(batch["A_input"].type(Tensor))
#         real_B = Variable(batch["A_exptC"].type(Tensor))
#         img_name = batch["input_name"]
#         fake_B, weights_norm = generator(real_A)
#         img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -1)
#         fake_B = torch.round(fake_B*255)
#         real_B = torch.round(real_B*255)
#         mse = criterion_pixelwise(fake_B, real_B)
#         psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
#         save_image(img_sample, "images/LUTs/%s/%s.jpg" % (epoch, img_name[0]+'_'+str(psnr)[:5]), nrow=3, normalize=False)

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generator(img):

    pred = classifier(img).squeeze()
    weights_norm = torch.mean(pred ** 2)
    combine_A = pred[0] * LUT0(img) + pred[1] * LUT1(img) + pred[2] * LUT2(img) #+ pred[3] * LUT3(img) + pred[4] * LUT4(img)

    return combine_A, weights_norm


#criterion = utils.get_loss(opt.loss_type)
if opt.loss_type == 'focal_loss':
    criterion = utils.FocalLoss(ignore_index=255, size_average=True)
elif opt.loss_type == 'cross_entropy':
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

criterion=criterion.to(device)


L_segexp = utils.loss.L_segexp()
L_color = utils.loss.L_color()
# L_TV = utils.loss.L_TV()
L_spa = utils.loss.L_spa()
L_exp = utils.loss.L_exp(16,0.6)
L_percept = utils.loss.perception_loss()
# ----------
#  Training
# ----------
# def checkpoint(model, epoch, opt):
#     try:
#         os.stat(opt.save_folder)
#     except:
#         os.mkdir(opt.save_folder)
#     model_out_path = opt.save_folder + "model_{}.pth".format(epoch)
#     torch.save(model.state_dict(), model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))
#     return model_out_path
# def log_metrics(_run, logs, iter, end_str=" "):
#     str_print = ''
#     for key, value in logs.items():
#         _run.log_scalar(key, float(value), iter)
#         str_print = str_print + "%s: %.4f || " % (key, value)
#     print(str_print, end=end_str)
# def eval(model, epoch):
#     print("==> Start testing")
#     tStart = time.time()
#     trans = transforms.ToTensor()
#     channel_swap = (1, 2, 0)
#     classifier.eval()#在非训练的时候是需要加上这句代码的，若没有，一些网络层的值会发生变动，不会固定，神经网络的每一次生成的结果也是不固定的，如dropout层和BN层，生成的质量可能好也可能不好
#     test_LL_folder = "data/LOL/test/low/"
#     test_NL_folder = "data/LOL/test/high/"
#     test_est_folder = "outputs/epoch_%04d/" % (epoch)
#     try:
#         os.stat(test_est_folder)
#     except:
#         os.makedirs(test_est_folder)
#     test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
#     test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
#     est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
#     for i in range(test_LL_list.__len__()):
#         with torch.no_grad():#with torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
#             LL = trans(Image.open(test_LL_list[i]).convert('RGB')).unsqueeze(0).cuda()
#             prediction,_ = model(LL)
#             # prediction = prediction.cpu().numpy().transpose(channel_swap)
#             prediction = prediction * 255.0
#             prediction = prediction.clip(0, 255)
#             Image.fromarray(np.uint8(prediction)).save(est_list[i])
#     psnr_score = 0.0
#     ssim_score = 0.0
#     lpips_score = 0.0
#     loss_fn_alex = lpips.LPIPS(net='alex')
#     for i in range(test_NL_list.__len__()):
#         gt = cv2.imread(test_NL_list[i])
#         est = cv2.imread(est_list[i])
#         # print("est:",est)
#         psnr_val = compare_psnr(gt, est, data_range=255)
#         ssim_val = compare_ssim(gt, est, multichannel=True)
#         gt1=torch.from_numpy(gt).permute(2,0,1).float()
#         est1=torch.from_numpy(est).permute(2,0,1).float()
#         lpips_val=loss_fn_alex(est1.unsqueeze(0).cpu(),gt1.unsqueeze(0).cpu())
#         psnr_score = psnr_score + psnr_val
#         ssim_score = ssim_score + ssim_val
#         lpips_score = lpips_score + lpips_val
#         # save_image(torch.cat([prediction, gt], dim=0), './test.jpg')
#     psnr_score = psnr_score / (test_NL_list.__len__())
#     ssim_score = ssim_score / (test_NL_list.__len__())
#     lpips_score = lpips_score / (test_NL_list.__len__())
    
#     print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
#     # print(f'psnr:{psnr_score:.4f}')
#     return psnr_score,ssim_score,lpips_score



avg_psnr = calculate_psnr()
# print(avg_psnr)
prev_time = time.time()
max_psnr = 0
max_ssim =0
max_lpips =100.0
max_epoch = 0

# pathGT = "./data/fiveK/expertC/JPG/480p/"
# pathDirGT = os.listdir(pathGT)
# picknumber = 1
for epoch in range(opt.epoch, opt.n_epochs):
    transff = transforms.ToTensor()
    losses = []
    # loss_D_avg = 0
    # loss_G_avg = 0
    # loss_pixel_avg = 0
    # loss_avg = 0
    cnt = 0
    psnr_avg = 0
    classifier.train()
    # for i, batch in enumerate(dataloader):

    for i,(images, labels) in enumerate(train_loader):
        # if len(images) <= 1:
        #     continue
        # Model inputs
        # real_A = Variable(batch["A_input"].type(Tensor))
        # real_B = Variable(batch["B_exptC"].type(Tensor))
        # sampleGT = random.sample(pathDirGT, picknumber)
        # # print(sampleGT)

        # real_B = Image.open(pathGT + sampleGT[0])
        # # print(real_B)
        # real_B = real_B.resize((384, 384), Image.ANTIALIAS)
        # # print(type(real_B))
        # # print(real_B.size)

        # real_B=transff(real_B)
        # # print(type(real_B))


        # real_B = torch.unsqueeze(real_B,0)
        # # print(real_B.shape)
        # real_B = real_B.cuda()

        images = images.to(device, dtype=torch.float32)
        # real_A = images.to(device, dtype=torch.float16)
        labels = labels.to(device, dtype=torch.long)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # optimizer_D.zero_grad()
        

        enhanced_image, weights_norm = generator(images)
        save_image(torch.cat([images, enhanced_image], dim=0), './out/g-1.jpg')
        # pred_real = discriminator(real_B)
        # pred_fake = discriminator(fake_B)

        # # Gradient penalty
        # gradient_penalty = compute_gradient_penalty(discriminator, real_B, fake_B)

        # # Total loss
        # loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) + opt.lambda_gp * gradient_penalty

        # loss_D.backward(retain_graph=True)
        # # optimizer_D.step()

        # loss_D_avg += ((-torch.mean(pred_real) + torch.mean(pred_fake)) / 2).item()

        # # ------------------
        # #  Train Generators
        # # ------------------
        # if i % opt.n_critic == 0:
        #     optimizer_G.zero_grad()

        #     fake_B, weights_norm = generator(real_A)
            
        #     pred_fake = discriminator(fake_B)
        #     save_image(torch.cat([real_A, fake_B], dim=0), './out/d-1.jpg')
        #     # Pixel-wise loss
        #     loss_pixel = criterion_pixelwise(fake_B, real_A)

        #     tv0, mn0 = TV3(LUT0)
        #     tv1, mn1 = TV3(LUT1)
        #     tv2, mn2 = TV3(LUT2)
        #     #tv3, mn3 = TV3(LUT3)
        #     #tv4, mn4 = TV3(LUT4)

        #     tv_cons = tv0 + tv1 + tv2 #+ tv3 + tv4
        #     mn_cons = mn0 + mn1 + mn2 #+ mn3 + mn4

        #     loss_G = -torch.mean(pred_fake) + opt.lambda_pixel * loss_pixel + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons
        #     # loss_G = -torch.mean(pred_fake)
        #     loss_G.backward(retain_graph=True)
            # optimizer_D.step()s
            # optimizer_G.step()

        cnt += 1
        # loss_G_avg += -torch.mean(pred_fake)

        # loss_pixel_avg += loss_pixel.item()
        # psnr_avg += 10 * math.log10(1 / loss_pixel.item())
        loss_percent = torch.mean(L_percept(images,enhanced_image))
        loss_color = 1000*torch.mean(L_color(enhanced_image))
        loss_exp = 300*torch.mean(L_exp(enhanced_image))
        loss_segexp = torch.mean(L_segexp(enhanced_image, labels))
        loss_spa = 1000*torch.mean(L_spa(enhanced_image,images))
            # fake_B = fake_B.detach()
            # optimizer.zero_grad()

            # if opt.ckpt is not None and os.path.isfile(opt.ckpt):
            #     checkpoint = torch.load(opt.ckpt, map_location=torch.device(device))
            #     # model.load_state_dict(checkpoint["model_state"],strict=False)
            #     model.load_state_dict(checkpoint,strict=False)
            #     for k in model.parameters():
            #         k.requires_grad = False
            #     # model.eval()
            #     model = model.to(device)
            #     outputs = model(fake_B)  # 增强图像通过预训练的分割网络得到预测图像
            #     cross_entropy = criterion(outputs, labels)  # 预测图像与GT做交叉熵损失
            #     loss = loss_segexp + cross_entropy
            # # loss = loss_segexp.item() + cross_entropy.item()
            # # print(loss)
            #     # loss 
            #     loss_avg += loss.item() 
            #     loss.backward()
        optimizer.zero_grad()
        outputs = model(enhanced_image)
        cross_entropy = criterion(outputs, labels)

        # loss = loss_segexp + cross_entropy
        lowlight_loss = loss_color + loss_segexp + cross_entropy + loss_spa + loss_percent
        # loss_avg += loss.item() 
        # loss.backward()
        optimizer_G.zero_grad()
        lowlight_loss.backward()
        # optimizer_D.step()
        optimizer_G.step()
        # optimizer.step()
        losses.append(lowlight_loss.item())



        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i
        batches_left = opt.n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        # sys.stdout.write(
        #     "\r[Epoch %d/%d] [Batch %d/%d] [D: %f, G: %f] [pixel: %f] [seg_loss: %f] [tv: %f, wnorm: %f, mn: %f] ETA: %s"
        #     % (
        #         epoch,
        #         opt.n_epochs,
        #         i,
        #         len(train_loader),
        #         loss_D_avg / cnt,
        #         loss_G_avg / cnt,
        #         loss_pixel_avg / cnt,
        #         loss_avg / cnt,
        #         tv_cons, weights_norm, mn_cons,
        #         time_left,
        #     )
        # )
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [color: %f, spa: %f ,exp: %f ,seg: %f ,per: %f] [total_loss: %f] [wnorm: %f] "
            % (
                epoch,
                opt.n_epochs,
                i,
                len(train_loader),
                loss_color.item() / cnt,
                loss_spa.item() / cnt,
                loss_exp.item() / cnt,
                loss_segexp.item() / cnt,
                loss_percent.item() / cnt,
                lowlight_loss.item(),
                weights_norm        
            )
        )
        # print(f'\rtrain loss : {lowlight_loss.item():.5f}|batch :{i}/{len(train_loader)}|  ',end='',flush=True)
    avg_psnr = calculate_psnr()
    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch
    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))
        # if epoch % (opt.snapshots) == 0:
        #         # file_checkpoint = checkpoint(generator, epoch, opt)
        #         # exp.add_artifact(file_checkpoint)

        #         psnr_score, ssim_score,lpips_score = eval(generator, epoch)
        #         if max_psnr < psnr_score:
        #             max_psnr=psnr_score
        #             max_epoch=epoch
        #         if max_ssim < ssim_score:
        #             max_ssim = ssim_score
        #         if max_lpips > lpips_score:
        #             max_lpips = lpips_score       
        #         # print("max_psnr:",max_psnr)
        #         # print("max_ssim:",max_ssim)
        #         sys.stdout.write(" [PSNR: %f] [max PSNR: %f, max_ssim: %f, max_lpips: %f,epoch: %d]\n"% (psnr_score, max_psnr,max_ssim,max_lpips, max_epoch))
            # logs = {
            #     "psnr": psnr_score,
            #     "ssim": ssim_score,
            #     "lpips": lpips_score,
            #     "max_psnr":max_psnr,
            #     "max_ssim": max_ssim,
            #     "max_lpips": max_lpips,
            #     "max_epoch":max_epoch,
            # }

    # if (epoch+1) % 10 == 0:
    #    visualize_result(epoch+1)

    if epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        LUTs = {"0": LUT0.state_dict(), "1": LUT1.state_dict(), "2": LUT2.state_dict()} #, "3": LUT3.state_dict(), "4": LUT4.state_dict()
        torch.save(LUTs, "saved_models/%s/LUTs_%d.pth" % (opt.output_dir, epoch))
        torch.save(classifier.state_dict(), "saved_models/%s/classifier_%d.pth" % (opt.output_dir, epoch))
        # file = open('saved_models/%s/result.txt' % opt.output_dir,'a')
        # file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n"% (avg_psnr, max_psnr, max_epoch))
        # file.close()
