"""
Derived from https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
"""
import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import ConvNet


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    print('>>>>>>>>>local rank', rank, 'gpu', gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    print('>>>>>>>> init_process_group complete')
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    val_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=val_sampler)

    start = datetime.now()
    total_train_step = len(train_loader)
    total_val_step = len(val_loader)
    for epoch in range(args.epochs):
        model.train()  # train mode
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_train_step,
                                                                         loss.item()))
        model.eval() # evaluate mode
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max 
                correct += pred.eq(labels.view_as(pred)).sum().item()
                if (i + 1) % 50 == 0 and gpu == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Val Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_val_step,
                                                                             val_loss.item()))

        if gpu == 0:
            print('\nValidation Accuracy: {}/{} ({:.0f}%)\n'.format(
                correct, len(val_sampler),
                100. * correct / len(val_sampler)))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    print(args)
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

        
if __name__ == '__main__':
    main()
