# -*- coding: utf-8 -*-
"""
Homework one

Try to write a python code for diagonalizing the spin-1/2 AF Heisenberg
model on a 4-site ring. Try to implement both U(1) and translational
symmetries.

Created on Fri Oct 12 15:27:55 2018

@author: Xinwei
"""
#%%
import numpy as np
from math import pi,e,sqrt
from scipy.sparse import csr_matrix
#%%
def PopCntBit( i ):
    return bin(i).count('1')

def PickBit( i , k , n ):
    return (i & ( ( 2**n - 1 ) << k ) ) >> k

def RotLBit( i , L , n ):
    return ( PickBit( i, 0, L-n ) << n ) + ( i >>( L-n ) )

def RotRBit( i , L , n ):
    return ( PickBit(i, 0, n) << (L-n) ) + ( i >> n )

def FlipBit( i , n ):
    return i ^ ( 1 << n )

def ReadBit( i , n ):
    return ( i & ( 1 << n ) ) >> n
#%%
# calculate Na
def Cyc(s,N):
    r = 1
    t = RotLBit(s, N , 1)
    while t != s:        
        t = RotLBit( t , N , 1 )
        r = r+1        
    return r
# check list
def Check(s,N):
    flag = 0 #whether the configuration is a principle configuration, 0 for yes, 1 for no
    R = 0 #translation steps
    sr = s # principle configuration
    
    t = s    
    for i in range(N):
        t = RotRBit( t , N , 1 )
        if t < sr:
            flag = 1
            sr = t
            R = i+1
        elif t == sr:
            continue

    return [flag,sr,R]
#%%   
#Generating Hamiltonian in each (Sz,k) subspace
def ham_gen(x,m,k,N):
    dim = len(x)
    row = []
    col = []
    data = []    
    for i in range(dim):
        s = x[i]
        Na = PriR[m][PriC[m].index(s)]

        for j in range(N):
            # diag-element
            diag = (ReadBit(s,j%N)-0.5)*(ReadBit(s,(j+1)%N)-0.5)
            row.append(i)
            col.append(i)
            data.append(diag)
            # off-diag-element                        
            if ReadBit(s,j%N)^ReadBit(s,(j+1)%N):
                t = FlipBit(s,j%N)
                t = FlipBit(t,(j+1)%N)                                   
                l = Check(t,N)[2] # translation steps
                tr = Check(t,N)[1] # principle configuration
                if tr in x: 
                    pos = x.index(tr)
                    Nb = PriR[m][PriC[m].index(tr)]
                    off = 0.5*sqrt(float(Na)/float(Nb))*e**(-1j*k*2*pi*l/N)
                    row.append(i)
                    col.append(pos)
                    data.append(off)
    ham = csr_matrix((data, (row, col)), shape=(dim, dim)).toarray()
    return ham  
#%%   
N = 4 # site number
Ndim = 2**4 # Hilbert space dimension
mdim = N+1 # sub-Hilbert space dimension of fixed Sz

PriC = [] # principle configuration in each subspace
PriR = [] #corresponding numbers
checklist = []

for ind in range(mdim):
    PriC.append([])
    PriR.append([])
    
for i in range(Ndim):
    m = PopCntBit(i)
    r = Cyc(i,N) 
    tmp = Check(i,N)
    checklist.append(tmp)
    
    if tmp[0] == 0:
        PriC[m].append(tmp[1])
        PriR[m].append(r)
            
#%%
# Generating basis in each [Sz,k] subspace        
Basis = []

for m in range(mdim):
    config = PriC[m]
    basm = []
    for k in range(N):
        basm.append([])
        for c in config:
            r = Cyc(c,N)
            if (float(k*r)/N)%1 == 0:
                basm[k].append(c)
                    
    Basis.append(basm)            
               
#%% calculation of all eigenvalues and eigenvectors  
            
eigvalues = [] # eigenvalus
eigvectors = [] # eigenvectors in each subspace
  
for m in range(mdim):
    for k in range(N):
        x = Basis[m][k]
        if x:
            h = ham_gen(x,m,k,N)# Hamiltonian in each (Sz,k) subspace
            print(h)
            a,b = np.linalg.eig(h)
            eigvalues.append(a)
            eigvectors.append(b)
            
print(eigvalues)
print(eigvectors)
