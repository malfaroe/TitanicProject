import warnings 
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')

#importing sklearn modules for data splitting and baseline work
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split





class Linear_Regression(object):
    def __init__(self, dataset, n_iter, alpha):
        self.dataset = dataset     #important: dataset enters as a DataFrame 
        self.n_iter = n_iter
        self.alpha = alpha
        
    
    #normalizing the dataset
    def normalizer(self):
        return (self.dataset - self.dataset.mean())/self.dataset.std()

    def data_process(self):      
        X = self.dataset.iloc[:,:-1].values
        y = self.dataset.iloc[:, -1].values
        X, y = np.array(X), np.array(y)# x and y converted to numpy arrays
        #reshape y
        y = np.reshape(y, (len(y), 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
        #add a ones column to X_train y X_test for fitting betas shape
        X_train = np.concatenate((np.ones((len(X_train),1)), X_train), axis = 1)
        X_test = np.concatenate((np.ones((len(X_test),1)), X_test), axis = 1)
        return X_train, X_test, y_train, y_test


    #Hipotesis
    def hipothesis(self, betas, X):
        return np.dot(X, betas)

    #Error of each iteration
    def costo(self, h, y):
        m = len(y) 
        sub = h -y
        cost = np.sum(sub**2)/ (2 * m)
        return cost 

    #Betas update with gradient descent
    def theta_update(self, h, betas, X, y):
        m = len(y)
        loss = h -y
        betas = betas - (self.alpha/m) * np.dot(X.T,loss)  
        return betas

    #error metrics
    def MSE(self, betas_final, X, y): #Mean-square error for training or test set
        return np.sum((self.hipothesis(betas_final, X) - y)**2) / len(y)

    # R-square metric
    def r2(self, bf, X, y):
        SStot = np.sum((y - np.mean(y))**2)
        SSres = np.sum((self.hipothesis(bf,X) - y)**2)
        r2 = np.round( 1 - (SSres / SStot), decimals= 4)
        return r2

    #Main loop
    def regressor(self, summary = False):
        J = [] #save cost of each iteration
        theta = [] #save the betas of each iteration
        #normalizing data
        #self.dataset = self.normalizer()
        X_train, X_test, y_train, y_test = self.data_process()
        m = len(y_train)
        betas = np.zeros((X_train.shape[1],1))#initializing betas
        betas_final =  np.zeros((X_train.shape[1],1))

        for i in range(self.n_iter):
        #hipotesis
            h = self.hipothesis(betas, X_train) 
        #updating thetas
            betas = self.theta_update(h, betas, X_train, y_train)
            theta.append(betas)
            cost = self.costo(h, y_train)
            if (i%1000000 == 0 or i == 0):
                print("Cost iteration number", i, ":", np.round(cost, decimals = 5))
            J.append(cost)
        betas_final = betas
        
        
        if summary:
            print("Summary:")
            print("Coefficients:", betas_final)
            print("Root Mean Square Error: %.4f"% np.expm1(np.sqrt(np.round(self.MSE(betas_final, X_test, y_test), decimals = 4)))) #usa expm1 test
            print("My R2 for Training Set:", self.r2(betas_final, X_train, y_train))
            print("My R2 for Test Set:", self.r2(betas_final, X_test, y_test))
          
            
        else:

            print("---------------------------------------------\n")
            print("Final results for", self.n_iter, "iterations:")
            print("··············································")
            print("Initial cost:", np.round(J[0], decimals = 5))
            print("Final cost:", np.round(J[-1], decimals = 5))
            print("Iteration of minimum:", np.argmin(J))
            print("MSE for training set :", np.round(self.MSE(betas_final, X_train, y_train), decimals = 2))
            print("MSE for test set :", np.round(self.MSE(betas_final, X_test, y_test), decimals = 2))
            print("")
            print("My R2 for Training Set:", self.r2(betas_final, X_train, y_train))
            print("My R2 for Test Set:", self.r2(betas_final, X_test, y_test))
            print("")
            print("My final betas:", betas_final)
            print("Intercept:", betas_final[:1][0][0]) #el [0] indica que muestra lo que hay dentro del arreglo
            print("Coefficients:", betas_final[1:][0])


            #testeo con sklearn
            lin_reg = LinearRegression()
            lin_reg.fit(X_train,y_train)
            y_pred = lin_reg.predict(X_test)
            print("                        ")
            print("Comparison with Sklearn results:")
            print("·································")
            print("Intercept Sklearn:", lin_reg.intercept_)
            print("Coefficients Sklearn:", lin_reg.coef_)

            from sklearn.metrics import r2_score
            print("Sklearn R2 - Score:", r2_score(y_test,y_pred))
            print("                        ")
            print("Plotting Cost Function Evolution:")
            print("··································")
            iter = np.arange(self.n_iter)
            cost_iter = J
            fig, ax = plt.subplots()  
            ax.plot(np.arange(self.n_iter), cost_iter, 'r')  
            ax.set_xlabel('Iterations')  
            ax.set_ylabel('Cost')  
            ax.set_title('Error vs. Training Epoch') 