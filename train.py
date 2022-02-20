import numpy as np
import argparse
import random
import os
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import CocoDataset, safe_collate
from model import PFNet


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Perspective Network on COCO.')
    #Paths
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default="2014",
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--logs', required=False,
                        default="./logs",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # Optimization parameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='set the number of epochs, default is 200')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='set the batch size, default is 32.')
    parser.add_argument('--seed', type=int, default=1987,
                        help='Pseudo-RNG seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data ####################################################################
    train_dataset = CocoDataset(args.dataset, "train", year=args.year)
    val_dataset = CocoDataset(args.dataset, "val", year=args.year)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=safe_collate)
    print(len(train_dataset), len(train_loader), len(val_dataset), len(val_loader))

    # Initialize network ######################################################
    model = PFNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    smooth_l1_loss = nn.SmoothL1Loss()

    # Training ################################################################
    # check path
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
    writer = SummaryWriter(log_dir=args.logs)

    print("start training")
    for epoch in range(args.n_epoch):
        # train
        model.train()
        train_loss = 0.0
        for i, batch_value in enumerate(train_loader):
            glob_iter = epoch * len(train_loader) + i

            train_inputs = batch_value[0].float().to(device)
            train_gt = batch_value[1].float().to(device)

            optimizer.zero_grad()
            train_outputs = model(train_inputs)
            loss = smooth_l1_loss(train_outputs, train_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iter[{:0>4}]/[{:0>4}] loss: {:.4f} lr={:.8f}".format(
                        epoch + 1, args.n_epoch, i + 1, len(train_loader), train_loss / (i + 1), scheduler.get_lr()[0]))

        train_loss /= len(train_loader)
        writer.add_scalar('train_loss', train_loss, epoch)

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, batch_value in enumerate(val_loader):
                val_inputs = batch_value[0].float().to(device)
                val_gt = batch_value[1].float().to(device)

                val_outputs = model(val_inputs)
                loss = smooth_l1_loss(val_outputs, val_gt)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        scheduler.step()

        # save model
        if (epoch+1) % 40 == 0:
            filename = 'pfnet_{:0>4}.pth'.format(epoch + 1)
            torch.save(model, os.path.join(args.logs, filename))

    print("Finished training.")
