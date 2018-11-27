# -*- coding: utf-8 -*-
"""
Homework Two

Try to write a python code for diagonalizing the transverse-field Ising model
on a 4-site ring. Try to implement translational symmetry.
 
Created on Fri Oct 12 15:27:55 2018

@author: Xinwei
"""
#%%
import numpy as np
from math import pi,e,sqrt,cos
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag
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
# calculate Ra
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
#Generating Hamiltonian in each k subspace
def ham_gen(x,k,g):
    dim = len(x)
    row = []
    col = []
    data = []    
    for i in range(dim):
        s = x[i]
        Na = PriR[PriC.index(x[i])]

        for j in range(N):
            # diag-element
            diag = 4*(ReadBit(s,j%N)-0.5)*(ReadBit(s,(j+1)%N)-0.5)
            row.append(i)
            col.append(i)
            data.append(diag)
            # off-diag-element  
            t = FlipBit(s,j) 
            l = Check(t,N)[2]
            tr = Check(t,N)[1]
            if tr in x:                
                pos = x.index(tr)
                Nb = PriR[PriC.index(tr)]
                off = g*sqrt(float(Na)/float(Nb))*e**(-1j*k*2*pi*l/N)
                row.append(i)
                col.append(pos)
                data.append(off)
                
    ham = csr_matrix((data, (row, col)), shape=(dim, dim)).toarray()
    return ham                
    
#%%
N = 4
Ndim = 2**4 # Hilbert space dimension

PriC = [] # principle configuration in each subspace
PriR = [] #corresponding numbers
checklist = []


for i in range(Ndim):
    tmp = Check(i,N)
    r = Cyc(i,N) 
    tmp = Check(i,N)
    checklist.append(tmp)
    
    if tmp[0] == 0:
        PriC.append(tmp[1])
        PriR.append(r)
            
#%%
# Generating basis in each k subspace        
Basis = []

for k in range(N):
    Basis.append([])
    for c in PriC:
        r = Cyc(c,N)
        if (float(k*r)/N)%1 == 0:
            Basis[k].append(c)
                              

#%%               
gvec = np.arange(0,2.0,0.1) # parameter g
gs = [] # ground state energy

for g in gvec:
    k0 = 0
    x0 = Basis[k0]
    Ham = ham_gen(x0,k0,g)
    for k in range(N-1):
        x = Basis[k+1]
        if x:
            h = ham_gen(x,k+1,g) # Hamiltonian in each subspace
            Ham = block_diag(Ham,h)

    eigvalue,eigvector = np.linalg.eig(Ham)
    gs.append(-max(eigvalue))
print(gs)
    
#%%
# Analytic result  
gsa = [] 

for g in gvec:
    e0 = 0
    for k in range(N):
        e0 = e0 - sqrt(1+g**2-2*g*cos(2*pi*k/N))
    gsa.append(e0)
print(gsa)   
  