# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:15:23 2017

@author: hubert
"""

import numpy as np
import time
import scipy.optimize as optimization
from numpy.random import random


def func(params, xdata, ydata):
    """
    Objective function for optimization.
    """
    return ydata - np.dot(xdata, params)


class NCA():
    """
    Network Component Analysis (NCA)

    This class implements the Network Component Analysis algorithm as described in
    Liao & al: http://www.pnas.org/content/100/26/15522.abstract
    """
    def __init__(self, Z=np.zeros((2, 2)), P=np.zeros((2, 2))):
        """
        Initialize the NCA object.

        Args:
            Z (np.ndarray): The connectivity matrix.
            P (np.ndarray): The initial transcription factor activities.
        """
        self.Z = Z
        N, L = Z.shape
        A = np.zeros(Z.shape)
        for i in range(N):
            for j in range(L):
                if Z[i][j]:
                    A[i][j] = random()
        self.A = np.matrix(A)
        self.P = P
        self.interp = np.array(np.dot(self.A, self.P))
        self.Ecart = 0

    def check_criteria(self):
        """
        Check the three criteria for NCA to be applicable.
        """
        # Criterion 1 & 2
        crit_1_2 = True
        if np.linalg.matrix_rank(self.A) < self.A.shape[1]:
            crit_1_2 = False
        else:
            for i in range(self.A.shape[1]):
                bad_ind = np.where(self.A.T[i] != 0)[1]
                A_bis = np.zeros(self.A.shape)
                comp = 0
                for j in range(self.A.shape[0]):
                    if comp > len(bad_ind) - 1 or j < bad_ind[comp]:
                        for k in range(self.A.shape[1]):
                            if i != k:
                                A_bis[j][k] = self.A[j, k]
                    comp += 1
                if np.linalg.matrix_rank(np.matrix(A_bis)) < self.A.shape[1] - 1:
                    crit_1_2 = False

        # Criterion 3
        crit_3 = True
        if np.linalg.matrix_rank(self.P.T) < len(self.Z[0]):
            crit_3 = False

        return crit_1_2 and crit_3

    def train(self, E, num_it=2):
        """
        Train the NCA model.

        Args:
            E (np.ndarray): The expression data.
            num_it (int): The number of iterations.
        """
        if not self.check_criteria():
            print("Certains critères ne sont pas respectés. Try to change the network topology (Z matrix)")
            return 0

        debut = time.time()
        for it in range(num_it):
            print(f'Iteration number: {it + 1}')

            M = len(E[0])
            N = len(E)
            L = len(self.Z[0])

            # Optimize P
            print('Optimizing P ...')
            E_bis = E.flatten()
            A_bis = np.zeros((M * N, M * L))
            for i in range(M):
                for j in range(N):
                    for k in range(L):
                        A_bis[i * N + j][i * L + k] = self.A[j, k]

            P_bis_initial = np.zeros((M*L))
            P_bis = optimization.leastsq(func, P_bis_initial, args=(A_bis, E_bis))[0]

            print('Updating P ...')
            self.P = P_bis.reshape((L, M))

            # Optimize A
            print('Optimizing A ...')
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
                res = optimization.leastsq(func, xo, args=(u.T, v))[0]
                for j in range(len(dis)):
                    new_A[i][dis[j]] = res[j]

            print('Updating A ...')
            self.A = new_A
            self.interp = np.array(np.dot(self.A, self.P))
            self.Ecart = np.abs(E - np.dot(self.A, self.P))

            print(f'Mean deviation with E = {np.mean(self.Ecart)}')

        fin = time.time()
        print(f"Training finished in {fin - debut:.2f} seconds")
