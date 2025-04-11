cimport numpy as cnp
import numpy as np
import pandas as pd
from math import exp


class Model(object):

    def __init__(self, trainset, K, alpha=0.01, reg=0.5, beta = 0.01, seed=0, mu=0, sigma=0.1):

        self.seed = seed  
        self.K = K                         # number of latent factors         
        self.trainset = trainset           # trainset for user preference learning
        self.W = trainset.n_words          # vocabulary size
        self.I = trainset.n_users          # number of users
        self.J = trainset.n_items          # number of items
        self.D = trainset.n_reviews        # number of reviews in trainset
        self.W_D = 0                       # total number of words in trainset
              
        '''Hyperparameters'''       
        self.beta = beta                   # Hyper-parameter for topic-word distribution 
        self.alpha = alpha                 # trade-off parameter for two objectives
        self.reg = reg                     # regularization parameter 
        
        '''Initialize Parameters'''
        np.random.seed(self.seed)    
        self.b_0 = self.trainset.global_mean                                 # overall offset
        self.U = np.abs(np.random.normal(mu,sigma,size=(self.I,self.K)))     # user factor matrix
        self.V = np.random.normal(mu,sigma,size=(self.J,self.K))             # item factor matrix
        self.b_i = np.random.normal(mu,sigma,size=self.I)                    # user bias vector
        self.b_j = np.random.normal(mu,sigma,size=self.J)                    # item bias vector              
        if self.trainset.vocab is not None:
            self.phi = np.ones((self.K*2, self.W))/self.W
            self.theta = np.ones((self.D,self.K*2))/(self.K*2)
            self.kappa = 1.0                    
            '''Initialize variational parameters'''               
            self.gamma = np.zeros(self.D, dtype = np.object)
            for d, i, j, r, doc in self.trainset.all_reviews():
                self.gamma[d] = np.zeros(len(doc), dtype=np.object)
                self.W_D += len(doc)
                for n in range(len(doc)):
                    self.gamma[d][n] = np.ones(self.K*2)/(self.K*2)
        

        
    def online_item_inference(self, irid, new_corpus, diagnosis_reg = 0, lr_sgd = 0.01, epochs=100, verbose=False):
        
        # fix phi and user parameters (i.e., b_i and U matrix)
        # re-estimate item parameters, i.e., b_j and v_j, based on online new corpus
        
        cdef int j = self.trainset.raw2inner_id_items[irid]
        cdef double b_0 = self.b_0
        cdef double b_j = self.b_j[j]
        cdef cnp.ndarray[cnp.double_t] v_j = self.V[j].copy()
        # user biases
        cdef cnp.ndarray[cnp.double_t] b_i = self.b_i
        # user factors
        cdef cnp.ndarray[cnp.double_t, ndim=2] U = self.U
        cdef cnp.ndarray[cnp.double_t, ndim=2] phi = self.phi
        cdef cnp.ndarray[cnp.double_t, ndim=2] theta = np.ones((len(new_corpus),self.K*2))/(self.K*2)
        
        cdef double reg = diagnosis_reg, lr = lr_sgd, alpha = self.alpha, kappa = self.kappa
        cdef int d, k, w, n, K = self.K                
        cdef double r, dot, err, sse   
        cdef cnp.ndarray[cnp.double_t] derive_vec
        
        gamma = np.zeros(len(new_corpus), dtype = np.object)
        for d, i, j, r, doc in self.trainset.match_online_corpus(new_corpus):
            gamma[d] = np.zeros(len(doc), dtype=np.object)
            for n in range(len(doc)):
                gamma[d][n] = np.ones(self.K*2)/(self.K*2)
      
        for epoch in range(epochs):  
            sse = 0
            for d, i, j, r, doc in self.trainset.match_online_corpus(new_corpus):
                if i is None: # for a new user, assume zero user bias (b_i=0) and zero latent factor (u_i=0)
                    err = r - (b_0 + b_j)   
                    sse += err**2
                    # update item bias
                    b_j += lr * (err - reg * (b_j - self.b_j[j]))
                    continue
                
                for n in range(len(doc)):
                    # excluding the current position from count variables
                    w = doc[n]
                    # update gamma
                    for k in range(K*2):
                        gamma[d][n][k] = theta[d,k] * phi[k,w]
                    gamma[d][n] /= np.sum(gamma[d][n])                  

                # compute current error
                dot = 0  
                for k in range(K):
                    dot += U[i,k] * v_j[k]
                err = r - (b_0 + b_i[i] + b_j + dot)      
                sse += err**2
                
                # update item bias
                # b_j += lr * (err - reg * b_j)
                b_j += lr * (err - reg * (b_j - self.b_j[j]))

                # update latent factors
                # compute derivatives given current values
                derive_vec = np.zeros(K)
                for n in range(len(doc)):
                    for k in range(K):
                        derive_vec[k] += kappa * (gamma[d][n][k] - gamma[d][n][k+K])
                for k in range(K):
                    derive_vec[k] -= kappa * len(doc) * (theta[d,k] - theta[d,k+K] )      
                    # v_j[k] += lr * (err * U[i,k] - reg * v_j[k] + alpha * derive_vec[k] * U[i,k])
                    v_j[k] += lr * (err * U[i,k] - reg * (v_j[k] - self.V[j,k]) + alpha * derive_vec[k] * U[i,k])
                               
                for k in range(K):
                    theta[d,k] = exp(kappa*(U[i,k] * v_j[k]))
                    theta[d,k+K] = exp(-1*kappa*(U[i,k] * v_j[k]))
                theta[d] /= theta[d].sum()
            
            if verbose:
                # print mse
                print("Epoch {}, MSE = {}, {} {}".format(epoch, sse/len(new_corpus), b_j, v_j))
                                
        return b_j, v_j

            
        
    def RatingMF_SGD(self, lr_sgd = 0.005, threshold = 0.001, test_set = pd.DataFrame()):
        # user biases
        cdef cnp.ndarray[cnp.double_t] b_i = self.b_i
        # item biases
        cdef cnp.ndarray[cnp.double_t] b_j = self.b_j
        # user factors
        cdef cnp.ndarray[cnp.double_t, ndim=2] U = self.U
        # item factors
        cdef cnp.ndarray[cnp.double_t, ndim=2] V = self.V

        cdef int i, j, k
        cdef double r, err, dot, prev_L
        cdef double L = np.inf
        cdef double b_0 = self.b_0
        cdef double lr = lr_sgd
        cdef double reg = self.reg

        for current_epoch in range(1000):                
            for d, i, j, r in self.trainset.all_ratings():

                # compute current error
                dot = 0  
                for k in range(self.K):
                    dot += U[i,k] * V[j,k]
                err = r - (b_0 + b_i[i] + b_j[j] + dot)
               
                # update biases
                b_i[i] += lr * (err - reg * b_i[i])
                b_j[j] += lr * (err - reg * b_j[j])
                b_0 += lr * err
                
                # update factors
                for k in range(self.K):
                    #U[i,k] += lr * (err * V[j,k] - reg * U[i,k])
                    U[i,k] = max(U[i,k] + lr * (err * V[j,k] - reg * U[i,k]) , 0)
                    V[j,k] += lr * (err * U[i,k] - reg * V[j,k])
            
            # update parameters
            self.b_0 = b_0
            self.b_i = b_i
            self.b_j = b_j
            self.U = U
            self.V = V

            # compute mse on test set
            if test_set.shape[0] > 0:
                prev_L = L
                L = self.validate_mse(test_set)
                print("Epoch {}, test MSE = {}".format(current_epoch,L))
                if (prev_L-L)/L < threshold:
                    break



    def UnitedMF_SGD(self, lr_sgd = 0.005, threshold = 0.001, test_set = pd.DataFrame(), update_kappa=True):
        cdef double L = np.inf, prev_L, L_rating, L_review
        cdef double reg = self.reg, lr = lr_sgd, alpha = self.alpha
        cdef int index, i, j, k, w, n, n_w=self.W, K = self.K
        cdef double r, dot, err, p_w, beta = self.beta
        cdef cnp.ndarray[cnp.double_t] derive_vec
        cdef double derive_kappa
        
        # initialize parameters
        # overall offset
        cdef double b_0 = self.trainset.global_mean
        # user biases
        cdef cnp.ndarray[cnp.double_t] b_i = self.b_i 
        # item biases
        cdef cnp.ndarray[cnp.double_t] b_j = self.b_j 
        # user factors
        cdef cnp.ndarray[cnp.double_t, ndim=2] U = self.U 
        # item factors
        cdef cnp.ndarray[cnp.double_t, ndim=2] V = self.V 
        # topic model parameters
        cdef cnp.ndarray[cnp.double_t, ndim=2] phi = self.phi
        cdef cnp.ndarray[cnp.double_t, ndim=2] theta = self.theta
        cdef double kappa = self.kappa
              
        # initialize variational counts
        cdef cnp.ndarray[cnp.double_t] n_k 
        cdef cnp.ndarray[cnp.double_t, ndim=2] n_kw
        n_k = np.zeros(K*2)            
        n_kw = np.zeros((K*2, n_w))
        for index, i, j, r, doc in self.trainset.all_reviews():        
            for n in range(len(doc)):
                w = doc[n]
                for k in range(K*2):
                    n_kw[k,w] += self.gamma[index][n][k]
                    n_k[k] += self.gamma[index][n][k]
                
        for current_epoch in range(1000):
            for index, i, j, r, doc in self.trainset.all_reviews():                  
                for n in range(len(doc)):
                    # excluding the current position from count variables
                    w = doc[n]
                    n_k -= self.gamma[index][n]
                    n_kw[:,w] -= self.gamma[index][n]
                    # update gamma
                    for k in range(K*2):
                        self.gamma[index][n][k] = theta[index,k] * (n_kw[k,w] + beta)/(n_k[k] + n_w*beta)
                    self.gamma[index][n] /= np.sum(self.gamma[index][n])                  
                    # update the current position with count variables
                    n_k += self.gamma[index][n]
                    n_kw[:,w] += self.gamma[index][n]

                # compute current error
                dot = 0  
                for k in range(K):
                    dot += U[i,k] * V[j,k]
                err = r - (b_0 + b_i[i] + b_j[j] + dot)              
                
                # update biases
                b_i[i] += lr * (err - reg * b_i[i])
                b_j[j] += lr * (err - reg * b_j[j])
                b_0 += lr * err

                # update latent factors
                # compute derivatives given current values
                derive_vec = np.zeros(K)
                for n in range(len(doc)):
                    for k in range(K):
                        derive_vec[k] += kappa * (self.gamma[index][n][k] - self.gamma[index][n][k+K])
                for k in range(K):
                    derive_vec[k] -= kappa * len(doc) * (theta[index,k] - theta[index,k+K])                    
                    U[i,k] = max(0, U[i,k] + lr * (err * V[j,k] - reg * U[i,k] + alpha * derive_vec[k] * V[j,k]))
                    V[j,k] += lr * (err * U[i,k] - reg * V[j,k] + alpha * derive_vec[k] * U[i,k] )
                
                if update_kappa:
                    # update kappa               
                    derive_kappa = 0
                    for k in range(K):
                        for n in range(len(doc)):
                            derive_kappa += (self.gamma[index][n][k]-self.gamma[index][n][k+K]) * (U[i,k]*V[j,k])                  
                        derive_kappa -= len(doc) * (theta[index,k] - theta[index,k+K]) * (U[i,k]*V[j,k])
                    kappa += lr * alpha * derive_kappa
                             
                # update theta
                for k in range(K):
                    theta[index,k] = exp(kappa*(U[i,k] * V[j,k]))
                    theta[index,k+K] = exp(-1*kappa*(U[i,k] * V[j,k]))

                theta[index] /= theta[index].sum()
                               
            # update phi
            for k in range(K*2):
                for w in range(n_w):
                    phi[k,w] = (n_kw[k,w] + beta)/(n_k[k] + n_w*beta)          
            
            # update parameters
            self.b_0 = b_0
            self.b_i = b_i
            self.b_j = b_j
            self.U = U
            self.V = V            
            self.kappa = kappa
            self.theta = theta
            self.phi = phi

            # compute mse on test set
            if test_set.shape[0] > 0:
                prev_L = L
                L = self.validate_mse(test_set)
                print("Epoch {}, test MSE = {}".format(current_epoch,L))
                if (prev_L-L)/L < threshold:
                    break

                
     
    def predict_rating(self, urid, irid):
        r_hat = self.b_0
        if self.trainset.knows_raw_user(urid):
            i = self.trainset.raw2inner_id_users[urid]
            r_hat += self.b_i[i]
            if self.trainset.knows_raw_item(irid):
                j = self.trainset.raw2inner_id_items[irid]
                r_hat += self.b_j[j] + self.U[i].dot(self.V[j])
        else:
            if self.trainset.knows_raw_item(irid):
                j = self.trainset.raw2inner_id_items[irid]
                r_hat += self.b_j[j]
        return r_hat

    

    def validate_mse(self,test_set):
        rating_loss = 0
        for index, row in test_set.iterrows():
            j = row['item']
            r = row['rating'] 
            i = row['user']
            r_hat = self.predict_rating(i,j)
            rating_loss += (r_hat-r)**2
        return rating_loss/test_set.shape[0]
    
        