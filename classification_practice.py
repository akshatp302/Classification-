import torch
from torch import nn
import numpy as np 
import sklearn
from sklearn.datasets import make_circles 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import pandas as pd 

class data_feed :
    def __init__(self,samples):
        self.n = samples
        self.visuals()
        self.data_conversition()
        # print(self.visuals())

    def visuals(self):
        
        self.x_ , self.y_ = make_circles(n_samples= self.n,
                            noise= 0.15,
                            random_state=42)
        
        plt.figure(figsize=(8,8))
        plt.scatter(x=self.x_[:,0],
                    y=self.x_[:,1],s=5,
                    c=self.y_)
        plt.show()
        
        table_formate = pd.DataFrame({"X1":self.x_[:,0], "X2 =" :self.x_[:,1], "y =": self.y_})
        return table_formate.head()
        
    def data_conversition(self):
        
        self.X_tensor = torch.from_numpy(self.x_).type(dtype=torch.float32)
        self.Y_tensor = torch.from_numpy(self.y_).type(dtype=torch.float32)
       
        print(f"The previous dtype = {self.x_.dtype}")
        print(f"After conversition = {self.X_tensor.dtype}")
        
    def data_split(self):
        
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X_tensor,
                                                                             self.Y_tensor,
                                                                             test_size = 0.4,
                                                                             random_state = 42)
            
        
        
        
        
        
        



    
    
    
