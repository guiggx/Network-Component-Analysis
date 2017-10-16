# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:15:23 2017

@author: hubert
"""

import numpy as np
import time
import scipy.optimize as optimization
from random import *



def func(params, xdata, ydata):
    return (ydata - np.dot(xdata, params))


class nca():
    def __init__(self, Z = np.zeros((2,2)), P = np.zeros((2,2))):
        self.Z = Z
        A = np.zeros(Z.shape)
        for i in range(N):
            for j in range(L):
                if Z[i][j]  : 
                    A[i][j] = random()    
        A = np.matrix(A)
        self.A = A
        self.P = P
        self.interp = np.array(np.dot(A,P))
        self.Ecart = 0
        
        
    def check_criteria(self):
        crit_1_2 = True
        rang_A = np.linalg.matrix_rank(self.A)
        
        if rang_A < self.A.shape[1]:
            crit_1_2 = False
        else : 
            for i in range(self.A.shape[1]):
                bad_ind = np.where(self.A.T[i] !=0 )[1]
                A_bis = np.zeros(self.A.shape)
                comp = 0
                for j in range(self.A.shape[0]):
                    
                    if comp > len(bad_ind)-1 or j < bad_ind[comp] :
                        for k in range(self.A.shape[1]):
                            if i != k :
                                A_bis[j][k] = self.A[j, k]
                    comp += 1
                
                if np.linalg.matrix_rank(np.matrix(A_bis)) < self.A.shape[1] - 1:
                    crit_1_2 = False
                    print i, np.linalg.matrix_rank(np.matrix(A_bis))
                    
        crit_3 = True 
        if np.linalg.matrix_rank(self.P.T) < len(self.Z[0]) : 
            crit_3 =False
        return crit_1_2*crit_3
        


    def train(self, E, num_it = 2):
        
        crit = self.check_criteria()

        debut = time.time() 
        it = 0  
        while it < num_it : 
            crit = self.check_criteria()
            if crit == False :
                print("Certains critères ne sont pas respectés. Try to change the network topology (Z matrix)")
                return 0
            else :
                print 'itération numéro : ', it+1
                
                print 'Préparation à optimisation de P ...'
                
                M = len(E[0])
                N = len(E)
                L = len(self.Z[0])
                
                
                E_bis = np.zeros(M*N)
                A_bis = np.zeros((M*N, M*L))
                
                comp =0
                
                for i in range(M):
                    for j in range(N):
                        E_bis[comp] = E[j][i]
                        comp += 1          
                for i in range(M):
                    for j in range(N):
                        for k in range(L):
                            A_bis[i*N +j][i*L+k] = self.A[j, k]
                            
                print 'Optimization de P ...'
                P_bis =  optimization.leastsq(func,P_bis,  args=(A_bis, E_bis) ) [0]  
                print 'Actualisation de P ...'
                
                self.P = np.zeros(P_bis.shape)
                
                comp =0
                for i in range(len(self.P[0])):
                    for j in range(len(P)):
                        self.P[j][i] = P_bis[comp]
                        comp += 1
                        
                print 'Optimization de A ...'
            
                new_A = np.zeros(self.A.shape) 
                for i in range(len(self.A)):  
                    v = E[i]                    
                    dis = np.where(self.Z[i] != 0)[0]
                    xo = np.zeros(len(dis))    
                    for k in range(len(dis)):
                        xo[k] = self.A[i, dis[k]]    
                    u = np.zeros((len(dis), len(self.P[0])))    
                    for k in range(len(dis)):
                        for l in range(len(self.P[0])):            
                            u[k][l] = self.P[dis[k]][l]
                    res = optimization.leastsq(func,xo,  args=(u.T, v)) [0]
                    for j in range(len(dis)):
                        new_A[i][dis[j]] = res[j]         
            
                print 'Actualisation de A ...'
                
                for i in range(len(self.A)):
                    for j in range(L):
                        self.A[i,j] = new_A[i][j] 
                self.interp = np.array(np.dot(self.A,self.P))
                self.Ecart = np.abs(E - np.dot(self.A, self.P))
                
                print 'Ecart moyen avec E = ', np.mean(Ecart)  
                it += 1
            
        fin = time.time()
        print fin-debut
        