import torch
import argparse
from Utils.SINet import SINet_ResNet50
from Utils.Dataloader import get_loader
import torch.optim as optim 
from torch.autograd import Variable
from datetime import datetime
import os
# from apex import amp

LR = 1e-4
DECAY_RATE = 0.1
DECAY_EPOCH = 30
TRAIN_IMG_DIR =  './Dataset/TrainDataset/Image/'
TRAIN_GT_DIR = './Dataset/TrainDataset/GT/'
MODEL_PATH = './Model/'
save_epoch = 10

def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        cam_sm, cam_im = model(images)
        loss_sm = loss_func(cam_sm, gts)
        loss_im = loss_func(cam_im, gts)
        loss_total = loss_sm + loss_im
        scale_factor = 65536
        scaled_loss = loss_total * scale_factor
        scaled_loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data))
    save_path = MODEL_PATH
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path + 'SINet_%d.pth' % (epoch+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=40, help='epoch number')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='size of training image')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)
    model = SINet_ResNet50(channel=32).cuda()
    if (len(os.listdir('Model'))!=0):
        state_dict = torch.load('Model\SINet_2.pth')
        model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), LR)
    LogitsBCE = torch.nn.MSELoss()
    train_loader = get_loader(TRAIN_IMG_DIR, TRAIN_GT_DIR, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=12)
    total_step = len(train_loader)
    print( "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n""Training Save: {}\ntotal_num: {}\n".format(TRAIN_IMG_DIR, TRAIN_GT_DIR, LR, opt.batchsize, MODEL_PATH, total_step))
    for epoch_iter in range(1, opt.epoch):
        adjust_lr(optimizer, epoch_iter, DECAY_RATE, DECAY_EPOCH)
        trainer(train_loader=train_loader, model=model,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=LogitsBCE, total_step=total_step)
    
        