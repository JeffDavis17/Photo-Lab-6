import numpy as np
import pandas as pd
from sympy import *
import math as m


# Data 
data = pd.read_csv(r"C:/Users/jcdav/Documents/GitHub/Photo-Lab-6/data.csv")
data = np.array(data)
ex = np.array(data[1:3,1:-2],dtype=float) # Exterior Orientation Parameters
im_coord = np.array(data[6:9,1:5],dtype=float) # Image Coordinates
f = 101.4 # Focal Length in mm



# Step 1 - Initial Values for the Ground Coordinates
# Im_coords → g_coords

# First Approximate the Ground Coordinates using Coplanarity




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
    M = np.array([[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]])
    return M

R1 = Rot(ex[0,3],ex[0,4],ex[0,5])
R2 = Rot(ex[1,3],ex[1,4],ex[1,5])


