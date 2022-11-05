#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
from scipy.optimize import minimize
from sympy import symbols, Eq, solve


# In[2]:


# importing the data from the given csv file

df = pd.read_csv('01_data_mars_opposition_updated.csv')
mars_longitude = df.values[:,5:9]
mars_geocentriclattitude = df.values[:,9:11]
mars_meanlongitude = df.values[:,11:15]
time = df.values[:,0:5]


# In[3]:


Mars_heliocentriclongitude = np.array((mars_longitude[:,0]*30)+mars_longitude[:,1]+(mars_longitude[:,2]/60)+(mars_longitude[:,3]/3600))


# In[4]:


Mars_heliocentriclongitude


# In[5]:


def dergree_convertor(degree):
    ZodiacIndex = int(degree/30)
    Deg_ = math.floor(degree-(ZodiacIndex*30))
    minute = (degree - ((ZodiacIndex*30)+Deg_))*60
    second = (minute - math.floor(minute))*60
    if (round(second)==60):
        minute = minute +1
        second = 0
    
    return ZodiacIndex,Deg_,math.floor(minute),round(second)


# In[6]:


def diff_minute(a,b):
    a_ = datetime.datetime(a[0],a[1],a[2],a[3],a[4])
    b_ = datetime.datetime(b[0],b[1],b[2],b[3],b[4])
    c = b_-a_
    minutes = c.total_seconds() / 60
    return minutes


# In[7]:


def time_data(t):
    Min = []
    for i in range(1,len(t)):
        Min.append(diff_minute(t[i-1],t[i]))
        
    return Min


# In[8]:


Time_diff = time_data(time)


# In[9]:


Time_diff=[0] + Time_diff
Time_diff


# In[10]:


def get_theta(theta,theta_1,theta_2):
    
    
    theta_1 = math.degrees(theta_1)
    theta_2 = math.degrees(theta_2)
    theta = math.degrees(theta)
    
    if (theta_1<0):
        theta_1 = theta_1 +360
    if (theta_2<0):
        theta_2 = theta_2 +360
        
    A1 = abs(theta - theta_1)
    B1 = abs(theta - theta_2)
    
    if (A1>=B1):
        t = theta_2
    else:
        t = theta_1
        
    return math.radians(t)

        


# In[11]:


def get_theta(theta,theta_1,theta_2):
    
    
    theta_1 = math.degrees(theta_1)
    theta_2 = math.degrees(theta_2)
    theta = math.degrees(theta)
    
    if (theta_1<0):
        theta_1 = theta_1 +360
    if (theta_2<0):
        theta_2 = theta_2 +360
        
    A1 = abs(theta - theta_1)
    B1 = abs(theta - theta_2)
    
    if (A1>=B1):
        t = theta_2
    else:
        t = theta_1
        
    return math.radians(t)


# In[12]:


def pred(time,s,Z_0):
    Z = math.degrees(Z_0) + ((time*s)/(24*60))
    J = Z/360
    if (math.floor(J)!=0):
        for i in range(0,math.floor(J)):
            Z = Z-360
    return math.radians(Z)


# In[13]:


def get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,theta):
    A = (1+(M**2))
    B = 2*((M*U)-x_c)
    C = ((x_c)**2)+(U**2)-(r**2)
    P = (B**2) - (4*A*C)
    x_1 = (-B + math.sqrt(P))/(2*A)
    y_1 = (M*x_1)+Q
    x_2 = (-B - math.sqrt(P))/(2*A)
    y_2 = (M*x_2)+Q
    
    theta_1 = np.arctan2(y_1,x_1)
    theta_2 = np.arctan2(y_2,x_2)
    
    t = get_theta(theta,theta_1,theta_2)
    
    return t


# In[14]:


def loss(a,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = a[0]
        r = a[1]
        e_1 = a[2]
        e_2 = a[3]
        s = math.degrees(a[5])
        
        if (i ==0):    
            Z = a[4]
        else:
            Z = pred(v[i],s,Z)
        
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return theta_diff,max(theta_diff)
    


# In[15]:


a = [2.5991373255251933, 8.18212891, 1.519648971343584, 2.597157108784485, 0.9744796022943893, 0.00914694657579844]
(u,v) = (Mars_heliocentriclongitude,Time_diff)
loss(a,u,v)


# In[16]:


math.degrees(0.003783544798805938)*60


# # PROBLEM 2

# In[17]:


def problem2_loss(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):

        c = math.radians(a[0])
        
        e_1 = a[1]
        e_2 = math.radians(a[2])
        
        
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return max(theta_diff)


# In[18]:


def problem2(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = math.radians(a[0])
        
        e_1 = a[1]
        e_2 = math.radians(a[2])
        
        
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return theta_diff, max(theta_diff)


# In[19]:



(u,v) = (Mars_heliocentriclongitude,Time_diff)


# In[20]:


def bestOrbitInnerParams(r,s,u,v):
    
    bnds = ((0,360),(1,0.5*8),(0,360),(0,70))
    a = [150, 1.2, 170, 60]
    min = minimize(problem2_loss,a,args=(r,s,u,v), bounds=bnds,method='Powell')
    
    a = min.x
    theta_diff, theta_d = problem2(a,r,s,u,v)
    
    return a[0],a[1],a[2],a[3],theta_diff, theta_d


# In[21]:


r = 9
s = 0.524
bestOrbitInnerParams(r,s,u,v)


# # PROBLEM 3

# In[22]:


def Problem4(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = math.radians(a[0])
        e_1 = a[1]
        e_2 = math.radians(a[2])
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return max(theta_diff)


# In[23]:


def Problem4_(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = math.radians(a[0])
        e_1 = a[1]
        e_2 = math.radians(a[2])
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return theta_diff


# In[24]:


def best_r(a,u,v):
    E =[]
    F = []
    T = np.linspace(8, 10, 50, endpoint = True)
    for r in T:
        s = 0.524 
        bnds = ((0,360),(1,0.5*r),(0,360),(0,70))
        min = minimize(Problem4,a,args=(r,s,u,v), bounds=bnds,method='Powell')
    
        F.append(min.fun)
        E.append(Problem4_(min.x,r,s,u,v))
        
   
    E = np.array(E)
    F = np.array(F)
    
    id = np.argmin(F)

    E = E[id]
    F = F[id]
    
    return T[id],E,F


# In[25]:


a = [150, 1.2, 170, 60]
(u,v) = (Mars_heliocentriclongitude,Time_diff)


# In[26]:


best_r(a,u,v)


# # Problem 4

# In[27]:


def Problem3(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = math.radians(a[0])
        e_1 = a[1]
        e_2 = math.radians(a[2])
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return max(theta_diff)


# In[28]:


def Problem3_(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = math.radians(a[0])
        e_1 = a[1]
        e_2 = math.radians(a[2])
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return theta_diff


# In[29]:


def best_s(a,r,u,v):
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    T = np.linspace(0.5235, 0.5245, 120, endpoint = True)
    for s in T:
        bnds = ((0,360),(1,0.5*r),(0,360),(0,70))
        min = minimize(Problem3,a,args=(r,s,u,v), bounds=bnds,method='Powell')
        A.append(min.x[0])
        B.append(min.x[1])
        C.append(min.x[2])
        D.append(min.x[3])
        F.append(min.fun)
        E.append(Problem3_(min.x,r,s,u,v))
        
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    
    id = np.argmin(F)
    A = A[id]
    B = B[id]
    C = C[id]
    D = D[id]
    E = E[id]
    F = F[id]
    
    return T[id],A,B,C,D,E,F


# In[30]:


a = [150, 1.5, 150, 55]
r= 9
(u,v) = (Mars_heliocentriclongitude,Time_diff)


# In[31]:


best_s(a,r,u,v)


# # PROBLEM 5

# In[32]:


def problem_5loss(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = math.radians(a[0])
        e_1 = a[1]
        e_2 = math.radians(a[2])
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return max(theta_diff)


# In[33]:


def Problem5_(a,r,s,u,v):
    theta_diff=[]
    for i in range(len(u)):
        c = math.radians(a[0])
        e_1 = a[1]
        e_2 = math.radians(a[2])
        if (i ==0):    
            Z = math.radians(a[3])
        else:
            Z = pred(v[i],s,Z)
        x_c = math.cos(c)
        y_c = math.sin(c)
        x_e = e_1*(math.cos(e_2))
        y_e = e_1*(math.sin(e_2))
        M = math.tan(Z)
        Q = y_e - (x_e*M)
        U = Q - y_c
    
        theta_pred = get_x_y(x_c,y_c,x_e,y_e,M,Q,U,r,math.radians(u[i]))
        
  
        theta_diff.append(abs(theta_pred - math.radians(u[i])))
        
        
    return theta_diff


# In[34]:


def bestMarsOrbitParams(a,u,v):
    A = []
    B = []
    C = []
    D = []
    E = []
    F = []
    G = []
    T = np.linspace(8, 10, 20, endpoint = True)
    for r in T: 
            Q = best_s(a,r,u,v)
            A.append(Q[0])
            B.append(Q[1])
            C.append(Q[2])
            D.append(Q[3])
            E.append(Q[4])
            F.append(Q[5])
            G.append(Q[6])
            
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    E = np.array(E)
    F = np.array(F)
    G = np.array(G)
    
    id = np.argmin(G)
    A = A[id]
    B = B[id]
    C = C[id]
    D = D[id]
    E = E[id]
    F = F[id]
    G = G[id]
    
    return T[id],A,B,C,D,E,F,G


# In[35]:


a = [150, 1.2, 170, 60]
(u,v) = (Mars_heliocentriclongitude,Time_diff)


# In[36]:


bestMarsOrbitParams(a,u,v)


# In[ ]:




