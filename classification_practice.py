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
        print(table_formate.head())
        
    def data_conversition(self):
        
        self.X_tensor = torch.from_numpy(self.x_).type(dtype=torch.float32)
        self.Y_tensor = torch.from_numpy(self.y_).type(dtype=torch.float32)
       
        print(f"The previous dtype = {self.x_.dtype}")
        print(f"After conversition = {self.X_tensor.dtype}")
        
    def data_split(self):
        
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X_tensor,
                                                                             self.Y_tensor,
                                                                             test_size = 0.3,
                                                                             random_state = 42)
        if len(self.X_train) > 0 and len(self.X_test) > 0:
            print("The Data has been sucssfully splited ")
            print(f"The Lenght of the Traning data is {len(self.X_train)}")
            print(f"The Lenght of the Testing data is {len(self.X_test)}")
        else:
            print("Something error in data spliting")
            
    def accuracy_check(self,Y_labels,Model_output):
        correct = torch.eq(Y_labels,Model_output).sum().item()
        accuracy= (correct/len(Model_output))*100
        return accuracy        
            
       

class Brain(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        
        self.Linear_layer_1 = nn.Linear(in_features= 2 , out_features= 16)
        self.ReLu1 = nn.ReLU()
        
        self.Linear_layer_2 = nn.Linear(in_features= 16 , out_features= 32)
        self.ReLu2 = nn.ReLU()
        
        self.Linear_layer_3 = nn.Linear(in_features= 32 , out_features=8 )
        self.ReLu3 = nn.ReLU()
        
        self.Linear_layer_4 = nn.Linear(in_features= 8 , out_features= 1)
        
        self.sequential = nn.Sequential(self.Linear_layer_1,self.ReLu1, 
                                        self.Linear_layer_2,self.ReLu2,
                                        self.Linear_layer_3,self.ReLu3,
                                        self.Linear_layer_4)
    def forward (self,x):
        return self.sequential(x)


class refine():
    def __init__(self,samples):
        self.my_model = Brain()
        self.data_acess = data_feed(samples)
        self.data_acess.data_split()
        
        
        self.loss_store_traning = []
        self.epoch_store_traning = []
        self.accuracy_values_store_traning = []
        
        
        self.loss_store_test = []
        self.accuracy_values_store_test = []
        
        
        
        self.X_input_train = self.data_acess.X_train
        self.Y_input_train = self.data_acess.Y_train
        
        self.X_input_test = self.data_acess.X_test
        self.Y_input_test = self.data_acess.Y_test
        
        
        
        
        self.optimizer = torch.optim.Adam(params=self.my_model.parameters(),
                                          lr= 0.001)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.epoch = 500
        self.traning()
        self.evulation()
    
    def traning (self):
        
        for epoch in range(self.epoch):
            self.my_model.train()
            
            self.output_raw = self.my_model(self.X_input_train).squeeze()
            self.output_with_activation = torch.sigmoid(self.output_raw)
            self.output_with_roundoff = torch.round(self.output_with_activation)
            
            self.loss_calculated_traning = self.loss_function(self.output_raw,self.Y_input_train)
            
            accuracy_values_traning = self.data_acess.accuracy_check(Y_labels=self.Y_input_train,
                                           Model_output=self.output_with_roundoff)
            
            
            
            self.optimizer.zero_grad()
            self.loss_calculated_traning.backward()
            
            self.optimizer.step()
            
            if epoch % 10 == 0:
                self.loss_store_traning.append(self.loss_calculated_traning.item())
                self.epoch_store_traning.append(epoch)
                
                print(f"This is the loss During the traning {self.loss_calculated_traning:2f} at this epoch {epoch}")
            
            if epoch % 50 == 0:
                self.accuracy_values_store_test.append(accuracy_values_traning)
                print(f"The accuracy during the traning is {accuracy_values_traning:2f}% at the {epoch}th epoch")
        
            
            
            
    def evulation (self):
        with torch.no_grad():
            
            self.output_raw_test = self.my_model(self.X_input_test).squeeze()
            self.output_with_activation_test = torch.sigmoid(self.output_raw_test)
            output_roundoff_test = torch.round(self.output_with_activation_test)
        
            
            
            loss_calculated_testing = self.loss_function(self.output_raw_test,self.Y_input_test)
            self.loss_store_test.append(loss_calculated_testing)
            
            accuracy_value_testing = self.data_acess.accuracy_check(Y_labels=self.Y_input_test,
                                                                    Model_output=output_roundoff_test)
            self.accuracy_values_store_traning.append(accuracy_value_testing)
            
            
            print(f"The test loss after the no_Gradients {loss_calculated_testing:2f}")
            print(f"The accuracy value at the testing is {accuracy_value_testing:2f}% with the test sample of {len(self.Y_input_test)}")
        
        
        
# Prototype_1 = data_feed(samples=300)
Proces_1 = refine(100)

    

        



    
    
    
