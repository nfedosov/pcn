# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 01:21:28 2022

@author: Fedosov
"""




import numpy as np
from matplotlib import pyplot as plt

    
    
# for map generation  
class PCN():

    #delay >= 1
    #min N_layers - 2, N_units - list
    #type - 'unit' (0,1) or 'infinite' (-oo, +oo)
    def __init__(self, N_units = [2,2], delay = 1, N_pairs = 1, intype = 'unit',outtype = 'unit'):
        self.N_units = N_units
        self.delay = delay
        self.N_pairs = N_pairs
        self.N_layers = len(N_units)
        
        self.intype = intype
        self.outtype = outtype
        
        self.x_rate = 0.02
        self.W_rate = 0.005
        self.b_rate = 0.005
        
        
        self.W = list()
        self.b = list()
        for i in range(self.N_layers-1):
            self.W.append(np.random.randn(N_units[i+1], N_units[i])/np.sqrt(N_units[i]))
            self.b.append(np.zeros((N_units[i+1],1)))
            
            
        self.x_history = list()
        for i in range(self.N_layers):
            self.x_history.append(np.zeros((N_pairs,N_units[i],delay+1)))
            
        self.x = list()
        for i in range(self.N_layers):
            self.x.append(np.zeros((N_pairs,N_units[i],1)))
        
        
      
    def change_N_parallel(self, N_new):
        self.N_pairs = N_new
        
        self.x_history = list()
        for i in range(self.N_layers):
            self.x_history.append(np.zeros((N_new,self.N_units[i],self.delay+1)))
            
        self.x = list()
        for i in range(self.N_layers):
            self.x.append(np.zeros((N_new,self.N_units[i],1)))
        
        
    def change_rate(self, x_rate = 0.02, W_rate = 0.005, b_rate = 0.005):
        self.x_rate = x_rate
        self.W_rate = W_rate
        self.b_rate = b_rate
  
        
        
        
    def derivative(self, x):
        x = self.activation(x)
        return x*(1.0-x)
    
    def deactivation(self, x):
        x = x-0.0001*(x>0.9999)+0.0001*(x<0.0001)
        return - np.log(1.0/x-1.0)

    
    def activation(self, x):
        return 1.0/(1.0+np.exp(-x))
    
    
    
    
    def getout(self):
        if self.outtype == 'unit':
            return self.activation(self.x[self.N_layers-1])[:,:,0]
        else:
            return self.x[self.N_layers-1][:,:,0]
            
        
    def getin(self):
        return self.x[0][:,:,0]
        
    
    def step(self, in_x, out_x, size_Batch = 1):
          
        up = list()
        err = list()
        bottom = list()
        
        for i in range(self.N_layers-1):
            if (i == 0) and (self.intype == 'unit'):
               
                up.append(self.W[i]@self.x_history[i][:,:,-1,None]+self.b[i])
            else:
                up.append(self.W[i]@self.activation(self.x_history[i][:,:,-1,None])+self.b[i])
            err.append(up[i]-self.x_history[i+1][:,:,-1,None])
            bottom.append(self.derivative(self.x_history[i][:,:,-1,None])*(self.W[i].T@err[i]))
        
        
            
        
        if len(in_x) > 0:
            self.x[0] = in_x
        else:
            
            if self.intype == 'unit':
                self.x[0] = self.activation(self.deactivation(self.x[0])+self.x_rate*(-bottom[0]))
            else:
                self.x[0] += self.x_rate*(-bottom[0])
                
        
        for i in range(1, self.N_layers-1):
            self.x[i] += self.x_rate*(err[i-1]-bottom[i])
            
        if len(out_x) > 0:
            if self.outtype == 'unit':
                self.x[self.N_layers-1] = self.deactivation(out_x)
            else:
                self.x[self.N_layers-1] = out_x
        else:
            self.x[self.N_layers-1] += self.x_rate*(err[self.N_layers-2])
        

        
        if (len(out_x) > 0) and (len(in_x) > 0) and ((self.W_rate > 0) or (self.b_rate > 0)):
          
            for k in range(0,self.N_pairs,size_Batch):
                for i in range(self.N_layers-1):
                    #self.W[i] += -np.sum(err[i]@(self.activation(self.x_history[i][:,:,-1,None])).transpose([0,2,1]),axis = 0)*self.W_rate/(self.N_pairs)
                    #self.b[i] += -np.sum(err[i], axis = 0)*self.b_rate
                    self.W[i] += -np.sum(err[i][k:k+size_Batch]@(self.activation(self.x_history[i][k:k+size_Batch,:,-1,None])).transpose([0,2,1]),axis = 0)*self.W_rate/(self.N_pairs)
                    self.b[i] += -np.sum(err[i][k:k+size_Batch], axis = 0)*self.b_rate/(self.N_pairs)
        for i in range(self.N_layers):   
            self.x_history[i][:,:,0,None] = self.x[i].copy()
            self.x_history[i] = np.roll(self.x_history[i], 1, axis =  2)
                
       
        err_total = 0
        for i in range(self.N_layers-1):
            err_total += np.linalg.norm(err[i])**2
        err_up = np.sum(err[self.N_layers-2]**2)
   

        return err_total, err_up
    
    

    
    def set_to_zero(self):
        for i in range(self.N_layers):
            self.x[i] = np.zeros(self.x[i].shape)
            self.x_history[i] = np.zeros(self.x_history[i].shape)
        
    

    
    #возвращает реконстр дату, принимает input и output в режиме обучения
    def run(self,N_iter ,in_x = [], out_x = [], draw = False, size_Batch = 1):
        
        #if (len(in_x)>0) and (len(out_x)>0):
            
        #    self.x_history[0][:,:,-1,None] = in_x.copy()
        #    self.x_history[self.N_layers-1][:,:,-1,None] = out_x.copy()
            
            
        
     
        err_total = np.zeros(N_iter)
        err_up = np.zeros(N_iter)
        
        for t in range(N_iter):
      
            e_t, e_u = self.step(in_x, out_x, size_Batch)
          
            
            err_total[t] = e_t
            err_up[t] = e_u
            
        if draw:
            
            
            plt.figure()
            plt.plot(np.log10(err_total[1:]))
            plt.title('err total')
            
            plt.figure()
            plt.plot(np.log10(err_up[1:]))
            plt.title('err up')
                 
        return
    


from mnist import MNIST
import matplotlib.pyplot as plt

mndata = MNIST('samples')
N_batch = 100
N_epochs = 2
N_iter_pre = 250
N_iter_post = 1250

images_train, labels_train = mndata.load_training()

plt.figure()
plt.imshow(np.array(images_train[0]).reshape(28,28)/255.0)

N_train = len(labels_train)//60


#code_map = [np.array([0,0,0,0]),np.array([0,0,0,1]),np.array([0,0,1,0]),np.array([0,0,1,1]),
#            np.array([0,1,0,0]),np.array([0,1,0,1]),np.array([0,1,1,0]),np.array([0,1,1,1]),
#            np.array([1,0,0,0]),np.array([1,0,0,1])]

#labels_code = np.zeros((N_train,4,1))
labels_code = np.zeros((N_train,10,1))
images_code = np.zeros((N_train,28*28,1))


#### labels to code
for i in range(N_train):
    #labels_code[i,:,0] = code_map[labels_train[i]]
    dig = np.zeros(10)+0.0
    dig[labels_train[i]] = 1.0
    labels_code[i,:,0] = dig
    images_code[i,:,0] = np.array(images_train[i])/255.0   #
    
    
    
idx = np.arange(N_train,dtype = 'int')
    
pcn = PCN(N_units = [28*28,300,10],N_pairs = N_train)
pcn.set_to_zero()
pcn.change_rate(W_rate = 0, b_rate = 0)
pcn.run(N_iter = N_iter_pre, in_x = images_code, 
            out_x = labels_code, draw = True)

pcn.change_rate()
pcn.run(N_iter = N_iter_post,  in_x = images_code, 
            out_x = labels_code, draw = True, size_Batch = 100)






images_test, labels_test = mndata.load_testing()

N_test = len(labels_test)



images_code = np.zeros((N_test,28*28,1))

for i in range(N_test):
   
    images_code[i,:,0] = np.array(images_test[i])/255.0   #
    
    
pcn.change_N_parallel(N_test)  
pcn.change_rate(x_rate = 0.02)
pcn.run(1000, in_x = images_code, draw = True)
    
    
    
mean_acc = np.mean(np.argmax(pcn.getout(),axis = 1) == labels_test)
print(mean_acc)

    
    


#code_map = [np.array([0,0,0,0]),np.array([0,0,0,1]),np.array([0,0,1,0]),np.array([0,0,1,1]),
#            np.array([0,1,0,0]),np.array([0,1,0,1]),np.array([0,1,1,0]),np.array([0,1,1,1]),
#            np.array([1,0,0,0]),np.array([1,0,0,1])]

#labels_code = np.zeros((N_train,4,1))









#pcn.change_rate()
#pcn.run(N_iter = N_iter_post, in_x = images_code[idx_copy[:N_batch]], 
#            out_x = labels_code[idx_copy[:N_batch]], draw = True)
    
#    idx_copy = idx_copy[N_batch:]



'''
# 4 - for autoencoder
pcn = PCN(N_units = [28*28,300,10],N_pairs = N_batch)
for i in range(N_epochs):
    np.random.shuffle(idx)
    idx_copy = idx.copy()
    print(i)
    

    for k in range(N_train//N_batch):

        pcn.set_to_zero()
        pcn.change_rate(W_rate = 0, b_rate = 0)
        pcn.run(N_iter = N_iter_pre, in_x = images_code[idx_copy[:N_batch]], 
                out_x = labels_code[idx_copy[:N_batch]], draw = True)
        pcn.change_rate()
        pcn.run(N_iter = N_iter_post, in_x = images_code[idx_copy[:N_batch]], 
                out_x = labels_code[idx_copy[:N_batch]], draw = True)
        
        idx_copy = idx_copy[N_batch:]
        print('batch,', k)

    '''

#images, labels = mndata.load_testing()



        
    