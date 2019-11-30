import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SRCNN import Net
from dataset_h5 import Read_dataset_h5
import numpy as np
import math

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRCNN")
parser.add_argument("--batchSize", type=int, default=128)
parser.add_argument("--nEpochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--start-epoch", default=1, type=int)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument("--gpus", default="0", type=str)

def main():
    global opt, model, optimizer  #opt, model global
    opt = parser.parse_args() # opt < parser
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus # set gpu
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed) # set seed
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True # find optimal algorithms for hardware

    print("===> Loading datasets")
    train_set = Read_dataset_h5("data/train.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True) # read to DataLoader

    print("===> Building model")
    model = Net() # net
    criterion = nn.MSELoss(size_average=False) # set loss

    # set start_epoch
    epoch = opt.start_epoch

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            checkpoint = torch.load(opt.pretrained)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch'] + 1
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda() # set model&loss for use gpu

    print("===> Setting Optimizer")
    if opt.lr:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    else:
        optimizer = optim.Adam(model.parameters())
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("=> start epoch '{}'".format(epoch))
    print("===> Training")
    for epoch_ in range(epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch_)
        save_checkpoint(model, epoch_, optimizer)

def PSNR(loss):
    psnr = 10 * np.log10(1 / (loss + 1e-10))
    return psnr

def train(training_data_loader, optimizer, model, criterion, epoch):

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        optimizer.zero_grad()
        input, label = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False)
        total_loss = 0
        if opt.cuda:
            input = input.cuda()
            label = label.cuda()
        output = model(input)
        loss = criterion(output, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = total_loss/len(training_data_loader)
    psnr = PSNR(epoch_loss)
    print("===> Epoch[{}]: loss : {:.10f} ,PSNR : {:.10f}".format(epoch, epoch_loss, psnr))

def save_checkpoint(model, epoch, optimizer):
    model_out_path = "checkpoint/" + "SRCNN_{}_epoch_{}.tar".format('SGD', epoch)
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
