# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import torchvision.utils as vutils
import matplotlib.pyplot as plt
#from modules import VectorQuantizedVAE
from datetime import datetime
import cv2
import math
from scipy.interpolate import interp1d

# %%

from torch.utils.tensorboard import SummaryWriter


def train(data_loader, model, optimizer, args):
    i = 0
    for images, _ in data_loader:
        images = images.to(device)

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
            images = images.to(device)
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
        images = images.to(device)
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
            nn.Conv2d(dim, dim, 3, 1, 1),
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
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        #self.codeBook = VQEmbedding(K, dim)

        # self.decoder = nn.Sequential(
        #     ResBlock(dim),
        #     ResBlock(dim),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, dim, 4, 2, 1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
        #     nn.Tanh()
        # )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codeBook(z_e_x)
        return latents

    # def decode(self, latents):
    #     z_q_x = self.codeBook.embedding(latents).permute(0, 3, 1, 2)  # (B, C, H, W)
    #     x_tilde = self.decoder(z_q_x)
    #     return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)                                    # 卷积+残差 -> 16*1*28*28 (64*40*7*7)
        # z_q_x_st, z_q_x = self.codeBook.straight_through(z_e_x)    # z的分布
        # x_tilde = self.decoder(z_q_x_st)                           # 反卷积 -> 16*1*28*28
        return z_e_x

class VectorQuantizedVAE2(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(input_dim, dim, 4, 2, 1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(True),
        #     nn.Conv2d(dim, dim, 4, 2, 1),
        #     ResBlock(dim),
        #     ResBlock(dim),
        # )

        self.codeBook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    # def encode(self, x):
    #     z_e_x = self.encoder(x)
    #     latents = self.codeBook(z_e_x)
    #     return latents

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

# %%
def read_FL_pic(first_path,last_path):
    img1 = cv2.imread(first_path,flags=0)
    img2 = cv2.imread(last_path,flags=0)
    img_t1 = torch.Tensor(img1[np.newaxis,np.newaxis,:,:])
    img_t2 = torch.Tensor(img2[np.newaxis,np.newaxis,:,:])
    plt.subplot(1,2,1)
    plt.imshow(img_t1[0][0])
    plt.title('first')
    plt.subplot(1,2,2)
    plt.imshow(img_t1[0][0])
    plt.title('last')
    return img_t1,img_t2
def mult_pic(matrix,col=8,row=5,pix=50,padding = 5):
    img = np.zeros([row*(pix+padding),col*(pix+padding)])
    print(img.shape)
    for i in range(row):
        for j in range(col):
            img[i*(pix+padding):(i+1)*(pix+padding)-padding,j*(pix+padding):(j+1)*(pix+padding)-padding] = matrix[0][i*col + j][:][:] 
    return img
# %%

def plot_interp(interp_digital,index,plot = False):
    res2a = 255-interp_digital.cpu().detach().numpy()*255
    ret,thresh2=cv2.threshold(res2a[0][0],122,255,cv2.THRESH_BINARY_INV)  
    if plot:
        plt.imshow(thresh)
        # # plt.show()
    plt.imsave('./test_image/rocktrain/berea-in-{}.png'.format(index),thresh2)

# %%
# laod the pth
num_channels = 1
hidden_size = 40
k = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model1 = VectorQuantizedVAE1(num_channels, hidden_size, k).to(device)
model2 = VectorQuantizedVAE2(num_channels, hidden_size, k).to(device)
model = VectorQuantizedVAE(model1,model2,num_channels, hidden_size, k).to(device)
print('read the pth & pic')
a1 = torch.load('./trained/abc1-99.pth')
a2 = torch.load('./trained/abc2-99.pth')
a = torch.load('./trained/abc-99.pth')
model1.load_state_dict(a1)
model2.load_state_dict(a2)
model.load_state_dict(a)




# %%
first_path = './test_image/rocktrain/berea-000.png'
last_path = './test_image/rocktrain/berea-010.png'
img_t1,img_t2 = read_FL_pic(first_path,last_path)
print('finished read')

# %%


# %%
print("cal model1")
matrix1_m1,matrix2_m1 = model1(img_t1.to(device)),model1(img_t2.to(device))


# %%
# plot the middle feature
mult_pic1, mult_pic2= mult_pic(matrix1_m1.cpu().detach().numpy()),mult_pic(matrix2_m1.cpu().detach().numpy())
# # plt.imshow(mult_pic1)
# # plt.show()
# # plt.imshow(mult_pic2)




# %%
def mulit_interp(matrix_strat,matirx_end,nums=2):
    interp_matix = np.zeros([nums,1,40,50,50])
    for i in range(40):
        print("finish inpter-{}%".format(i/40*100))
        for j in range(50):
            for k in range(50):
                x = [1,10]
                y = [matrix_strat[0][i][j][k].cpu().detach().numpy(),matirx_end[0][i][j][k].cpu().detach().numpy()]
                rand_arr = np.random.rand(10)*1
                rand_arr[0],rand_arr[-1] = y[0],y[1]
                arrayx = np.linspace(x[0],x[1],10)
                fun = interp1d(arrayx,rand_arr)
                arrayx_new = np.linspace(x[0],x[1],nums)
                arrayy_new  = fun(arrayx_new)
                
                # arrayx_new = np.linspace(y[0].cpu().detach().numpy(),y[1].cpu().detach().numpy(),nums)
                # arrayx_n = np.linspace(0,2*math.pi,nums)
                # array_sin = np.sin(arrayx_n)

                # interp_matix[:,0,i,j,k] = arrayx_new*array_sin
                interp_matix[:,0,i,j,k] = arrayy_new
    return interp_matix
## -------------------------------------------------------------------------------------------
# interp part
print("start to interp")
interp_matrix =  mulit_interp(matrix1_m1,matrix2_m1,30)
interp_pic = mult_pic(interp_matrix[0])



# %%
plt.imshow(interp_pic)

# %%
m = interp_matrix.shape


# %%

#interp_matrix = np.random.rand(2,1,40,50,50)
# %%
print("cal model2 & figure")
for i in range(m[0]):
    interp_digital = model2(torch.tensor(interp_matrix[i],dtype=torch.float32).to(device))
    plot_interp(interp_digital[0],i)
    print("finish model2 - {}%".format(i/m[0]*100))


# %%


