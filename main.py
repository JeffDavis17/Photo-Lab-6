import numpy as np
import pandas as pd
from sympy import *
import math as m
from numpy.linalg import *


# Data 
data = pd.read_csv(r"C:/Users/jcdav/Documents/GitHub/Photo-Lab-6/data.csv")
data = np.array(data)
ex = np.array(data[1:3,1:-2],dtype=float) # Exterior Orientation Parameters
im_coord = np.array(data[6:9,1:5],dtype=float) # Image Coordinates
f = 101.4 # Focal Length in mm



# Step 1 - Initial Values for the Ground Coordinates

# Baseline = L2 - L1
B = np.array([[ex[1,0] - ex[0,0]],
[ex[1,1] - ex[0,1]],
[ex[1,2] - ex[0,2]]])


# Rotation Matrices for im 1 and 2
def Rot(o,p,k):
    o = m.radians(o)
    p = m.radians(p)
    k = m.radians(k)

    m11 = cos(p)*cos(k)
    m12 = sin(o)*sin(p)*cos(k) + cos(o)*sin(k)
    m13 = sin(o)*sin(k) - cos(o)*sin(p)*sin(k)
    m21 = -cos(p)*sin(k)
    m22 = cos(o)*cos(k) - sin(o)*sin(p)*sin(k)
    m23 = sin(o)*cos(k) + cos(o)*sin(p)*sin(k)
    m31 = sin(p)
    m32 = -sin(o)*cos(p)
    m33 = cos(o)*cos(p)
    M = np.array([[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]],dtype=float)
    return M

R1 = Rot(ex[0,3],ex[0,4],ex[0,5])
R2 = Rot(ex[1,3],ex[1,4],ex[1,5])

# Determine Lambda 
pt1 = np.array([im_coord[0,0],im_coord[0,1],-f],dtype=float)
pt2 = np.array([im_coord[1,0],im_coord[1,1],-f],dtype=float)
pt3 = np.array([im_coord[2,0],im_coord[2,1],-f],dtype=float)
u1 = R1@pt1
u2 = R2@pt2

a = np.array([[-u1[0],u2[0]],[-u1[1],u2[1]]])
b = np.array([B[0],B[1]])
lam = solve(a,b)

# Determine initial values for the ground coordinates of the selected tie points:
X1 = np.array([ex[0,0],ex[0,1],ex[0,2]],dtype=float) + lam[0]*R1@pt1
X2 = np.array([ex[0,0],ex[0,1],ex[0,2]],dtype=float) + lam[0]*R1@pt2
X3 = np.array([ex[0,0],ex[0,1],ex[0,2]],dtype=float) + lam[0]*R1@pt3


# Step 2 - Observation Equations

# r s and q terms:





