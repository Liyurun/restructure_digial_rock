# Generated with SMOP  0.41
import numpy as np
import random as random
from scipy.spatial.distance import pdist, squareform
import time 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




    

def sigital_fun():
    data_x1=np.array([np.random.randint(1,100) for x in range(1,96,1)])
    data_x2=np.array([np.random.randint(1,100) for x in range(1,96,1)])
    data_x3=np.array([np.random.randint(1,100) for x in range(1,96,1)])
    rang_list = range(1,96,2)
    x1 = np.array([x for x in rang_list])
    y1 = np.array([x for x in rang_list])
    z1 = np.array([x for x in rang_list])
    #num = np.random.randint(1,3,(len(data_x1)))
    leng = len(x1)
    num = np.random.randint(1,3,(1,len(data_x1)))
    #num = np.array([1,1,1, 2, 2, 1, 2, 1, 2, 2])
    #pos_catagory = [[25, 75, 1],[75, 75, 2],[50, 50, 1],[25, 25, 2],[75, 25, 2]]
    pos_catagory = np.vstack([data_x1,data_x2,data_x3,num]).transpose()
    prop = np.array([0.7,0.3])
    xstart = 10;xend = 100;xstep = 10
    num_index = (xend-xstart)//xstep
    est1 =np.zeros([len(x1),len(x1),len(x1)]);est2=np.zeros([len(x1),len(x1),len(x1)])
    timed = time.time()
    for xx,x_value in enumerate(rang_list):
        for yy,y_value in enumerate(rang_list):
            for zz,z_value in enumerate(rang_list):
                # position
                unknow = np.array([x_value,y_value,z_value])
                #unknow = [40,50,60]
                #var
                variance = prop*(1-prop)


                #possibility matrix
                
                m,n = pos_catagory.shape
                tem = np.zeros([m,2])
                tem1 = np.ones([m,1])      
                tem2 = tem1
                tem1 = tem1*(pos_catagory[:,n-1].reshape(-1,1)==1)
                tem2 = tem2*(pos_catagory[:,n-1].reshape(-1,1)==2)
                prop_cata = [tem1,tem2]
                # distance metrix
                x = np.row_stack((pos_catagory[:,:n-1],unknow))

                dis = squareform(pdist(x))
                dis = np.delete(dis,-1,axis=0)
                dis
                # vargoram
                nugget = 0;spherical = 1;ranges = 300
                tem = dis.copy()
                tem[tem<=ranges] = 1 
                tem[tem>ranges] = 0
                var_mat = nugget+spherical*(1.5*dis/ranges - 0.5*(dis/ranges)**3)*tem
                # covariance
                covar_mat = 1- var_mat
                # left reserve
                inver = np.linalg.inv(covar_mat[:,:-1])

                #weight
                weight = np.dot(inver,covar_mat[:,-1])
                # ranking as sorting
                estima1 = sum(sum(np.transpose(prop_cata[0][:])*weight)) + np.dot(prop[0],(1-sum(weight)))
                estima2 = sum(sum(np.transpose(prop_cata[1][:])*weight)) + np.dot(prop[1],(1-sum(weight)))
                

                est1[xx,yy,zz] = estima1
                est2[xx,yy,zz] = estima2
                iter_num = (xx-1)*leng*leng+(yy-1)*leng + zz
                if iter_num % 1000 == 0:
                    print('finish the ',iter_num, ' time useed ', time.time() - timed)
                    timed = time.time()
    
    result = est1>est2
    result1 = result*1
    array_sq = [x for x in range(xstart,xend+1,xstep)]
    print(result1)
    for i in range(leng):
        plt.imsave('aa'+ str(i)+'b.png',result[i])

if __name__ == "__main__":
    digital_fun()
    #plt.imsave('aac.png',result[1])
    

# array = xstart:xstep:xend
# % toc
# % figure
# % plot3(array,array,array.*result*1)
# % figure
# % surf(array,array,max(est1,est2))
# % figure
# % contour(array,array,max(est1,est2))
# % tempp = repmat(array,10,1,10)
# % 
# % scatter3(tempp,tempp,result1)
# for i = 1:10
#     for j = 1:10
#         for k = 1:10
#             if result1(i,j,k) == 0
#                 scatter3(i*10,j*10,k*10)
#                 hold on
#             end
            
#         end
#     end
#     pause(1)
# end

