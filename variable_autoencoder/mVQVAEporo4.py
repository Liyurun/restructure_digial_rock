# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import torchvision.utils as vutils
#import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from torch.utils.tensorboard import SummaryWriter


def train(data_loader, model, optimizer, args):
    i = 0
    for images, _ in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        if args.steps % 10 == 0:
            print("Step [{}/{}], loss/training_loss: {:2f} at step {}" 
                    .format(i+1, len(data_loader),loss.item(), str(args.steps)))
        i+=1
            #print('loss/training_loss: {:f} at step {:f}'.format(loss.item(), args.steps))

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)
        writer.add_scalar('loss/train/training_loss', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, args):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), args.steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde = model(images)
    return x_tilde

## ----------------------------------------------------------------------------------------------
# finding the z_e_q given x (by comparing with z_e_x)
class VectorQuantization(Function):

    @staticmethod
    def forward(ctx, inputs, codeBook):
        with torch.no_grad():
            # assigning the dimension of our embedding
            embedding_size = codeBook.size(1)

            inputs_size = inputs.size()
            # Flatten input
            inputs_flatten = inputs.view(-1, embedding_size)

            codeBook_sqr = torch.sum(codeBook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances of the inputs to the codeBook
            distances = torch.addmm(codeBook_sqr + inputs_sqr,
                                    inputs_flatten, codeBook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


# Required to pass gradients received by z_e_q to z_e_x as torch.min function makes the back-propagation gradient
# impossible
class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codeBook):

        # evaluating the indices with the least distance between inputs and codeBook
        indices = vq(inputs, codeBook)
        indices_flatten = indices.view(-1)
        # saving indices for backward pass
        ctx.save_for_backward(indices_flatten, codeBook)
        ctx.mark_non_differentiable(indices_flatten)
        codes_flatten = torch.index_select(codeBook, dim=0,
                                           index=indices_flatten)
        # get embedding corresponding to the inputs
        codes = codes_flatten.view_as(inputs)

        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codeBook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient with respect to the codeBook
            indices, codeBook = ctx.saved_tensors
            embedding_size = codeBook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))

            # for passing gradient backwards
            grad_codeBook = torch.zeros_like(codeBook)
            grad_codeBook.index_add_(0, indices, grad_output_flatten)

        return grad_inputs, grad_codeBook


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
## ----------------------------------------------------------------------------------------------
# function
def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


# Function to initialize the weights of our network
def weights_init(m):
    className = m.__class__.__name__
    if className.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", className)
# Structure of the residual block
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)
# Structure of the embedding layer
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        # creating the embedding
        self.embedding = nn.Embedding(K, D)
        # weights belong to a uniform distribution
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    # z_e_x --> latent code for the input image
    def forward(self, z_e_x):
        # converting BCHW --> BHWC
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        # Retrieving the indices corresponding to the input
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    # z_e_x --> latent code for the input image
    def straight_through(self, z_e_x):
        # converting BCHW --> BHWC
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()    # 64*40*7*7 -> 64*7*7*40
        
        # z_q_x --> latent code from the embedding nearest to the input code
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()    # 64*7*7*40 -> 64*40*7*7

        # z_q_x_bar --> backprop possible
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous() #3136*40 -> 64*40*7*7

        # used for generating the image (decoding)
        return z_q_x, z_q_x_bar

class VectorQuantizedVAE1(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_nums[0], 4, 2, 1),
            nn.BatchNorm2d(hidden_nums[0]),
            nn.ReLU(True),
            nn.Conv2d(hidden_nums[0], hidden_nums[1], 4, 2, 1),
            nn.BatchNorm2d(hidden_nums[1]),
            nn.ReLU(True),
            nn.Conv2d(hidden_nums[1], hidden_nums[2], 4, 2, 1),
            nn.Conv2d(hidden_nums[2], hidden_nums[3], 5, 4, 0),
            nn.Conv2d(hidden_nums[3], hidden_nums[4], 3, 3, 0),
            nn.Conv2d(hidden_nums[4], hidden_nums[5], 2),
            ResBlock(hidden_nums[5]),
            ResBlock(hidden_nums[5]),
            #ResBlock(hidden_nums[5]),
        )
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codeBook(z_e_x)
        return latents

    def forward(self, x):
        z_e_x = self.encoder(x)                                    # 卷积+残差 -> 16*1*28*28 (64*40*7*7)
        # z_q_x_st, z_q_x = self.codeBook.straight_through(z_e_x)    # z的分布
        # x_tilde = self.decoder(z_q_x_st)                           # 反卷积 -> 16*1*28*28
        return z_e_x

class VectorQuantizedVAE2(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()

        self.codeBook = VQEmbedding(K, hidden_nums[5])

        self.decoder = nn.Sequential(
            ResBlock(hidden_nums[5]),
            ResBlock(hidden_nums[5]),
            #ResBlock(hidden_nums[5]),
            # nn.ReLU(True),
            nn.ConvTranspose2d(hidden_nums[5], hidden_nums[4], 2),
            nn.BatchNorm2d(hidden_nums[4]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_nums[4], hidden_nums[3],3,3,0),
            nn.BatchNorm2d(hidden_nums[3]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_nums[3], hidden_nums[2], 5,4,0),
            nn.BatchNorm2d(hidden_nums[2]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_nums[2], hidden_nums[1], 4, 2, 1),
            nn.BatchNorm2d(hidden_nums[1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_nums[1], hidden_nums[0], 4, 2, 1),
            nn.BatchNorm2d(hidden_nums[0]),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_nums[0], input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def decode(self, latents):
        z_q_x = self.codeBook.embedding(latents).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, z_e_x):
        # z_e_x = self.encoder(x)                                    # 卷积+残差 -> 16*1*28*28 (64*40*7*7)
        z_q_x_st, z_q_x = self.codeBook.straight_through(z_e_x)    # z的分布
        x_tilde = self.decoder(z_q_x_st)                           # 反卷积 -> 16*1*28*28
        return x_tilde,z_e_x,z_q_x

class VectorQuantizedVAE(nn.Module):
    def __init__(self, model1,model2,input_dim, dim, K=512):
        super().__init__()
        self.mo1 = model1
        self.mo2 = model2
    def forward(self, x):
        res = self.mo1(x)
        x_tilde,z_e_x,z_q_x = self.mo2(res)
        return x_tilde,z_e_x,z_q_x



def main(args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)

    
    # save_filename = './models/{0}'.format(args.output_folder)

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])


    num_channels = 1
    from torch.utils import data
    # Define the data loaders
    transform1 =transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()])
    train_data = datasets.ImageFolder('./data',transform=transform1)
    train_loader = data.DataLoader(train_data,batch_size=args.batch_size,
                                    shuffle = True,num_workers=args.num_workers,pin_memory= True)
    test_data = datasets.ImageFolder('./test_image',transform=transform1)
    test_loader = data.DataLoader(test_data,batch_size=args.batch_size,shuffle = True,num_workers=args.num_workers)


    # Fixed images for TensorBoard
    fixed_images, _ = next(iter(test_loader))

    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    #plt.imsave('.\images\\real-{}.png'.format('1'),fixed_grid[0].cpu())
    writer.add_image('original', fixed_grid,0)
    model1 = VectorQuantizedVAE1(num_channels, args.hidden_size, args.k).to(args.device)
    model2 = VectorQuantizedVAE2(num_channels, args.hidden_size, args.k).to(args.device)
    model = VectorQuantizedVAE(model1,model2,num_channels, args.hidden_size, args.k).to(args.device)
    #model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer.add_graph(model, fixed_images.to(args.device))  # get model structure on tensorboard
    
    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction[0].to(args.device), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('reconstruction at start', grid, 0)        # 输出结果
    #plt.imsave('fake-{}.png'.format('1'),grid[0].cpu())

    img_list = []
    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args)
        loss, _ = test(test_loader, model, args)
        writer.add_graph(model, fixed_images.to(args.device))  # get model structure on tensorboard
        # writer.add_graph(model1, fixed_images.to(args.device))  # get model structure on tensorboard
        # writer.add_graph(model2, model1(fixed_images))  # get model structure on tensorboard
        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction[0].to(args.device), nrow=8, range=(-1, 1), normalize=True)
        
        writer.add_image('reconstruction at epoch {:f}'.format(epoch + 1), grid, epoch + 1)
        #if epoch%10 == 1:plt.imsave('./images/fake-{}.png'.format(epoch + 1),grid[0].cpu()) 
        print("loss = {} at epoch {}".format(loss, epoch + 1))
        writer.add_scalar('loss/testing_loss', loss, epoch + 1)
        img_list.append(grid)
        if epoch%10 == 0:
            torch.save(model1.state_dict(),'./trained/nabc1-{}.pth'.format(epoch))
            torch.save(model2.state_dict(),'./trained/nabc2-{}.pth'.format(epoch))
            torch.save(model.state_dict(),'./trained/nabc-{}.pth'.format(epoch))
            torch.save(model1,'./trained/na1-{}.pt'.format(epoch))
            torch.save(model2,'./trained/na2-{}.pt'.format(epoch))
            torch.save(model,'./trained/na-{}.pt'.format(epoch))

    

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')

    # General
    hidden_nums = [32,64,128,256,512,1024,2048]
    a = 'J:\\塔里木项目\\VAE'
    b = "D:\\python_jupyter\\数字岩心\\vae\\srVAE-master\\data"
    c = 'E:\\lyr\\VAE'
    parser.add_argument('--data-folder', type=str, default=c,
                        help='name of the data folder')

    # Latent space

    parser.add_argument('--hidden-size', type=int, default=40,
                        help='size of the latent vectors')
    parser.add_argument('--k', type=int, default=512,
                        help='number of latent vectors (default: 512)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=254,
                        help='batch size (default: 64)')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='vqvae',
                        help='name of the output folder (default: vqvae)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cuda',
                        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    main(args)

