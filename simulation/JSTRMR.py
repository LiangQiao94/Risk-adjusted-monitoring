# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:34:22 2024

@author: liang
"""

from scipy.stats import norm
from dataset import Dataset
from math import log
from numpy.random import RandomState
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import pandas as pd
import nltk,os
import string
import gensim
from gensim.models import CoherenceModel
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import gzip
import _pickle as cPickle
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



class JSTRMR(object):

    def __init__(self, trainset, K, S = 2, weight = 1, rho=0.8/0.2, beta = 0.01, seed=0):
        
        self.K = K
        self.weight = weight
        self.rho = rho
        self.S = S
        self.trainset = trainset
        self.W = trainset.n_words
        self.D = trainset.n_reviews
        self.random_state = RandomState(seed)
        self.seed = seed
        np.random.seed(self.seed)
        
        '''Hyperparameters'''       
        self.beta = beta       # Hyper-parameter for phi 
        self.alpha = 0.05*30/(self.S*self.K)
        self.gamma = np.ones(self.S)
        self.rho = rho
        
        '''Initialize Parameters'''
      
        if self.trainset.vocab is not None:
            self.phi = np.ones((self.K, self.S, self.W))/self.W
            self.theta = np.ones((self.D,self.S,self.K))/self.K
            self.pi = np.ones((self.D,self.S))/self.S
            self.mu = np.zeros(self.S)
            self.sigma = np.ones(self.S)
     
        
            '''Initialize latent variables and count variables'''     

            self.z = np.zeros(self.D, dtype=np.object)   
            self.word_l = np.zeros(self.D, dtype=np.object)   
            self.rating_l = np.zeros(self.D, dtype = np.int)
            
            self.nw = np.zeros((self.W,self.K,self.S))  
            self.nwsum = np.zeros((self.K,self.S))  
            self.ndw = np.zeros((self.D,self.K,self.S)) 
            self.ndwsum = np.zeros((self.D,self.S))
            self.nrsum = np.zeros(self.S)
            self.nrcount = np.zeros(self.S)
            self.ndr = np.zeros((self.D,self.S)) 
            
            rating_count = []
            for s in range(self.S):
                rating_count.append([])
                
            for d, i, j, r, doc in self.trainset.all_reviews():
                # initialize word assignmnts
                self.z[d] = np.zeros(len(doc), dtype=np.int)
                self.word_l[d] = np.zeros(len(doc), dtype=np.int)
                for n in range(len(doc)):
                    w = doc[n]
                    k = np.random.choice(self.K)
                    s = np.random.choice(self.S)
                    self.z[d][n] = k
                    self.word_l[d][n] = s
                    self.nw[w,k,s] += 1  
                    self.nwsum[k,s] += 1  
                    self.ndw[d,k,s] += 1 
                    self.ndwsum[d,s] += 1
                # initialize rating assignmnts
                s = 0
                if r < 0:
                    s = 1
                self.rating_l[d] = s
                self.ndr[d,s] += 1
                self.nrsum[s] += r
                self.nrcount[s] += 1
                rating_count[s].append(r)
                
            for s in range(self.S):
                self.mu[s] = np.mean(rating_count[s])
                self.sigma[s] = np.std(rating_count[s])
            

    def Inference(self, epochs=1000,threshold=0.001):
        
        p = np.ones((self.K,self.S))
        p_rating = np.ones(self.S)
        L_w = np.inf

        '''model inference'''
        for current_epoch in range(epochs):
            rating_count = []
            for s in range(self.S):
                rating_count.append([])
                
            for m, i, j, r, doc in self.trainset.all_reviews():   
                # sample sentiment label for the rating in document m
                sentiment = self.rating_l[m]
                
                self.nrcount[sentiment] -= 1
                self.nrsum[sentiment] -= r
                self.ndr[m,sentiment] -= 1
               
            	# do sampling via cumulative method
                p_rating = norm.pdf(r, loc = self.nrsum/self.nrcount, scale = self.sigma) * \
                        (self.ndwsum[m] + self.gamma)

                p_rating /= np.sum(p_rating)
                sentiment = np.random.choice(self.S, p=p_rating)
                
                # add newly assigned sentiment label to count variables
                self.nrcount[sentiment] += 1
                self.nrsum[sentiment] += r
                self.ndr[m,sentiment] += 1
                
                # update rating_l
                self.rating_l[m] = sentiment
                
                rating_count[sentiment].append(r)
                
                
                # sample sentiment and topic label for each word w_n in document m
                for n in range(len(doc)):
                    w = doc[n]
                    topic = self.z[m][n]
                    sentiment = self.word_l[m][n]
                    # remove current sentiment/topic assignment from the count variables
                    self.nw[w,topic,sentiment] -= 1
                    self.ndw[m,topic,sentiment] -= 1
                    self.nwsum[topic,sentiment] -= 1
                    self.ndwsum[m,sentiment] -= 1
                	        
                	# do sampling 
                    for k in range(self.K):
                        for s in range(self.S):
                    	    p[k,s] = (self.nw[w,k,s] + self.beta) / (self.nwsum[k,s] + self.W * self.beta) \
            					* (self.ndw[m,k,s] + self.alpha) / (self.ndwsum[m,s] + self.K * self.alpha) \
            					* (self.ndwsum[m,s] + self.weight*self.ndr[m,s] + self.gamma[s]) 
                    	    
                    p_flat = p.flatten()
                    p_flat /= np.sum(p_flat)
                    ind = np.random.choice(self.K*self.S, p=p_flat)
                    topic = ind // self.S
                    sentiment = ind - topic * self.S  

                    # add newly assigned topic and sentiment to count variables
                    self.nw[w,topic,sentiment] += 1
                    self.ndw[m,topic,sentiment] += 1
                    self.nwsum[topic,sentiment] += 1
                    self.ndwsum[m,sentiment] += 1


                    # update topic and sentiment vectors
                    self.z[m][n] = topic
                    self.word_l[m][n] = sentiment
                
                # update pi and theta
                for s in range(self.S):
                    self.pi[m,s] = (self.ndwsum[m,s] + self.weight*self.ndr[m,s] + self.gamma[s]) / (len(doc) + self.weight + self.gamma.sum())
                    for k in range(self.K):
                        self.theta[m,s,k] = (self.ndw[m,k,s] + self.alpha) / (self.ndwsum[m,s] + self.K * self.alpha)
                				
            
            # update mu
            for s in range(self.S):
                self.mu[s] = np.mean(rating_count[s])
                self.sigma[s] = np.std(rating_count[s])
            
            # update phi
            self.compute_phi()
            
            # word-likelihood
            prev_L = L_w
            L_w = 0
            L_r = 0
            for m, i, j, r, doc in self.trainset.all_reviews():         
                p_r = 0
                for s in range(self.S):
                    p_r += self.pi[m,s] * norm.pdf(r, loc = self.mu[s], scale = self.sigma[s])
                L_r -= np.log(p_r)
                for n in range(len(doc)):
                    w = doc[n]
                    p_w = 0
                    for s in range(self.S):
                        for k in range(self.K):
                            p_w += self.pi[m,s] * self.theta[m,s,k] * self.phi[k,s,w]
                    L_w -= np.log(p_w)
            print("Processing epoch {}, loss of reviews = {}, loss of ratings = {}".format(current_epoch,L_w,L_r))

            if (prev_L-L_w)/L_w < threshold:
                break  
                
                
    def online_Inference(self, new_corpus, epochs=10, verbose=False):
        
        p = np.ones((self.K,self.S))
        p_rating = np.ones(self.S)
        
        theta = np.empty((len(new_corpus),self.S,self.K))
        pi = np.empty((len(new_corpus),self.S))
        
        #L = np.inf
        '''model inference'''        
        for m, i, j, r, doc in self.trainset.match_online_corpus(new_corpus):
            
            nd = np.zeros((self.K,self.S))
            ndsum = np.zeros(self.S)
            mdsum = np.zeros(self.S)
            z = np.zeros(len(doc), dtype = np.int)
            word_l = np.zeros(len(doc), dtype = np.int)

            # initialize
            rating_l = np.random.choice(self.S)
            mdsum[rating_l] += 1        
            for n in range(len(doc)):
                k = np.random.choice(self.K)
                z[n] = k
                s = np.random.choice(self.S)
                word_l[n] = s
                nd[k,s] += 1
                ndsum[s] += 1
            # gibbs sampling  
            for current_epoch in range(epochs):
                for n in range(len(doc)):
                    topic = z[n]
                    sentiment = word_l[n]
                    w = doc[n]
                    # exclude current position from counts
                    nd[topic,sentiment] -= 1
                    ndsum[sentiment] -= 1

                	# sample probability
                    for k in range(self.K):
                        for s in range(self.S):
                            p[k,s] = self.phi[k,s,w] * (nd[k,s] + self.alpha) / (ndsum[s] + self.K * self.alpha) \
            					* (ndsum[s] + self.weight*mdsum[s] + self.gamma[s]) 
                    p_flat = p.flatten()
                    p_flat /= np.sum(p_flat)
                    ind = np.random.choice(self.K*self.S, p=p_flat)
                    topic = ind // self.S
                    sentiment = ind - topic * self.S
                    
                    # update count variables
                    nd[topic,sentiment] += 1
                    ndsum[sentiment] += 1 
                    
                    # update topic and sentiment vectors
                    z[n] = topic
                    word_l[n] = sentiment
                
                # sample sentiment label for the rating in document m                
                sentiment = rating_l
                # remove current sentiment assignment from the count variables
                mdsum[sentiment] -= 1
           	
                p_rating = norm.pdf(r, loc = self.mu, scale = self.sigma) * \
                            (ndsum + self.gamma)
                
                p_rating /= np.sum(p_rating)
                sentiment = np.random.choice(self.S, p=p_rating)

                # update variables
                mdsum[sentiment] += 1  
                # update sentiment vector
                rating_l = sentiment
            
            # compute theta and pi for the current collection
            for s in range(self.S):
                pi[m,s] = (ndsum[s] + self.weight*mdsum[s] + self.gamma[s]) / (len(doc) + self.weight + self.gamma.sum())
                for k in range(self.K):
                    theta[m,s,k] = (nd[k,s] + self.alpha) / (ndsum[s] + self.K * self.alpha)
        
        if verbose:
            # word-likelihood
            #prev_L = L
            L_w = 0
            L_r = 0
            for m, i, j, r, doc in self.trainset.match_online_corpus(new_corpus):         
                p_r = 0
                for s in range(self.S):
                    p_r += pi[m,s] * norm.pdf(r, loc = self.mu[s], scale = self.sigma[s])
                L_r -= np.log(p_r)
                for n in range(len(doc)):
                    w = doc[n]
                    p_w = 0
                    for s in range(self.S):
                        for k in range(self.K):
                            p_w += pi[m,s] * theta[m,s,k] * self.phi[k,s,w]
                    L_w -= np.log(p_w)
            print("loss of reviews = {}, loss of ratings = {}".format(L_w,L_r))
        
        return pi, theta                
            
            
                
                
    
    def online_sequential_Inference(self, trans_pi, trans_theta, new_corpus, epochs=10, verbose=False):
        
        # initialize assignments and count variables
        totalnd = 0
        totalmd = 0
        nd = np.zeros((self.K,self.S))
        ndsum = np.zeros(self.S)
        mdsum = np.zeros(self.S)
        z = np.zeros(len(new_corpus), dtype = np.object)
        word_l = np.zeros(len(new_corpus), dtype = np.object)
        rating_l = np.zeros(len(new_corpus), dtype=np.int)
        
        for m, i, j, r, doc in self.trainset.match_online_corpus(new_corpus):
            z[m] = np.zeros(len(doc), dtype=np.int)
            word_l[m] = np.zeros(len(doc), dtype=np.int)
            rating_s = np.random.choice(self.S)
            rating_l[m] = rating_s
            mdsum[rating_s] += 1
            totalmd += 1
            totalnd += len(doc)           
            for n in range(len(doc)):
                k = np.random.choice(self.K)
                z[m][n] = k
                s = np.random.choice(self.S)
                word_l[m][n] = s
                nd[k,s] += 1
                ndsum[s] += 1
        
        p = np.ones((self.K,self.S))
        p_sentiment = np.ones(self.S)
               
        for epoch in range(epochs):  
            for m, i, j, r, doc in self.trainset.match_online_corpus(new_corpus):
                # sample sentiment and topic label for each word w_n in document m
                for n in range(len(doc)):
                    topic = z[m][n]
                    sentiment = word_l[m][n]
                    w = doc[n]
                    # exclude current position from counts
                    nd[topic,sentiment] -= 1
                    ndsum[sentiment] -= 1
                    totalnd -= 1

                	# sample probability
                    for k in range(self.K):
                        for s in range(self.S):
                            p[k,s] = self.phi[k,s,w] * ((ndsum[s] + self.weight*mdsum[s]) + (totalnd + self.weight*totalmd) * self.rho * trans_pi[s]) / ((totalnd+self.weight*totalmd)*(1+self.rho)) \
                                * ((nd[k,s] + totalnd * self.rho * trans_pi[s] * trans_theta[s,k]) / (ndsum[s] + totalnd * self.rho * trans_pi[s]))
                    p_flat = p.flatten()
                    p_flat /= np.sum(p_flat)
                    ind = np.random.choice(self.K*self.S, p=p_flat)
                    topic = ind // self.S
                    sentiment = ind - topic * self.S
                    
                    # update count variables
                    nd[topic][sentiment] += 1
                    ndsum[sentiment] += 1 
                    totalnd += 1
                    
                    # update topic and sentiment vectors
                    z[m][n] = topic
                    word_l[m][n] = sentiment
                
                # sample sentiment label for the rating in document m                
                sentiment = rating_l[m]
                # remove current sentiment assignment from the count variables
                mdsum[sentiment] -= 1
                totalmd -= 1
           	
                for s in range(self.S):
                	p_sentiment[s] = norm.pdf(r, loc = self.mu[s], scale = self.sigma[s]) \
                      * ((ndsum[s] + self.weight*mdsum[s]) + (totalnd+self.weight*totalmd) * self.rho*trans_pi[s])  \
                      / ((totalnd + self.weight*totalmd) * (1+self.rho))
                
                p_sentiment /= np.sum(p_sentiment)
                sentiment = np.random.choice(self.S, p=p_sentiment)

                # update variables
                mdsum[sentiment] += 1  
                totalmd += 1
                # update sentiment vector
                rating_l[m] = sentiment
            
            # compute theta and pi for the current collection
            pi = ((ndsum + self.weight*mdsum) + (totalnd+self.weight*totalmd)*self.rho*trans_pi)  \
                / ((totalnd + self.weight*totalmd) * (1+self.rho))
            
            theta = np.empty((self.S,self.K))
            for k in range(self.K):
                for s in range(self.S):
                    theta[s,k] = (nd[k,s] + totalnd * self.rho * trans_pi[s] * trans_theta[s,k]) \
                        / (ndsum[s] + totalnd * self.rho * trans_pi[s])
            
            if verbose:
                L_w = 0
                L_r = 0
                for m, i, j, r, doc in self.trainset.match_online_corpus(new_corpus):        
                    p_r = 0
                    for s in range(self.S):
                        p_r += pi[s] * norm.pdf(r, loc = self.mu[s], scale = self.sigma[s])
                    L_r -= np.log(p_r)
                    for n in range(len(doc)):
                        w = doc[n]
                        p_w = 0
                        for s in range(self.S):
                            for k in range(self.K):
                                p_w += pi[s] * theta[s,k] * self.phi[k,s,w]
                        L_w -= np.log(p_w)
                print("Epoch {}, review loss = {}, rating loss = {}".format(epoch, L_w, L_r))
            
        return pi, theta
                           
	
    def compute_phi(self):        
        for k in range(self.K):
            for s in range(self.S):
                for w in range(self.W):
                    self.phi[k,s,w] = (self.nw[w,k,s] + self.beta) / (self.nwsum[k,s] + self.W * self.beta)
    

               
    def print_nodes(self, n_words, with_weights):
        # print root node
        node = self.topics[-1]
        out = 'root: \n'
        out += node.get_top_words(n_words, with_weights)
        print(out)
        # print each internal node
        for i in range(self.I):
            node = self.topics[i]
            out = 'internal node %d: ' % (i+1)
            out += node.get_top_events(True) 
            out += '\n'
            out += node.get_top_words(n_words, with_weights)
            print(out)
      

    def coherence(self, n_words):
        topics_top_words = []
        for i in range(self.I+1):
            topic = self.topics[i]
            top_idx = np.argsort(topic.phi)[::-1][:n_words] 
            top_words = [self.vocab[idx] for idx in top_idx]
            topics_top_words.append(top_words)
        
        docs = []
        for d in range(len(self.corpus)):
            doc = self.corpus[d]
            doc = [self.vocab[w] for w in doc]
            docs.append(doc)               
        bow_corpus = [self.vocab.doc2bow(doc) for doc in docs]
            
        cm = CoherenceModel(topics = topics_top_words, dictionary = self.vocab, corpus = bow_corpus, coherence='u_mass' )
        score = cm.get_coherence()                
        return score 




