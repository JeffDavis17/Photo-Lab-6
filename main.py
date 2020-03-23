import math as m
import numpy as np
import pandas as pd
from sympy import *
from numpy.linalg import *
from scipy.linalg import *
np.set_printoptions(precision=8,threshold=800,edgeitems=4,linewidth=800,suppress=True)
# Jeff Davis (214238414) - Photogrammetry Lab 06


# Data 
data = pd.read_csv(r"C:/Users/jcdav/Documents/GitHub/Photo-Lab-6/data.csv")
data = np.array(data)
ex = np.array(data[1:3,1:-2],dtype=float) # Exterior Orientation Parameters
im_coord = np.array(data[6:9,1:5],dtype=float) # Image Coordinates
f = 101.4e-3 # Focal Length in m


# Step 1 - Initial Values for the Ground Coordinates
# Baseline = L2 - L1
B = np.array([[ex[1,0] - ex[0,0]],
[ex[1,1] - ex[0,1]],
[ex[1,2] - ex[0,2]]])


# Rotation Matrices
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

R1 = Rot(ex[0,3],ex[0,4],ex[0,5]) # Rotation Matrix Image 1
R2 = Rot(ex[1,3],ex[1,4],ex[1,5]) # Rotation Matrix Image 2


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




# Start Iterative Process:

for i in range(6):

    # Step 2 - Observation Equations - First Design Matrix 
    # r s and q terms:
    def rsq(pt,Xl,Yl,Zl,M):
        r = M[0,0]*(pt[0] - Xl) + M[0,1]*(pt[1] - Yl) + M[0,2]*(pt[2] - Zl)
        s = M[1,0]*(pt[0] - Xl) + M[1,1]*(pt[1] - Yl) + M[1,2]*(pt[2] - Zl)
        q = M[2,0]*(pt[0] - Xl) + M[2,1]*(pt[1] - Yl) + M[2,2]*(pt[2] - Zl)
        return r,s,q

    # r,s and q for each approximated ground tie point, for each image
    r11,s11,q11 = rsq(X1,ex[0,0],ex[0,1],ex[0,2],R1) # Im 1 gr. pt 1
    r12,s12,q12 = rsq(X2,ex[0,0],ex[0,1],ex[0,2],R1) # Im 1 gr. pt 2
    r13,s13,q13 = rsq(X3,ex[0,0],ex[0,1],ex[0,2],R1) # Im 1 gr. pt 3
    r21,s21,q21 = rsq(X1,ex[1,0],ex[1,1],ex[1,2],R2) # Im 2 gr. pt 1
    r22,s22,q22 = rsq(X2,ex[1,0],ex[1,1],ex[1,2],R2) # Im 2 gr. pt 2
    r23,s23,q23 = rsq(X3,ex[1,0],ex[1,1],ex[1,2],R2) # Im 2 gr. pt 3


    # Partial with respect to X,Y,Z
    def A(r,s,q,M):
        f = 101.4e-3
        b14 = (f/q**2)*(r*M[2,0] - q*M[0,0])
        b15 = (f/q**2)*(r*M[2,1] - q*M[0,1])
        b16 = (f/q**2)*(r*M[2,2] - q*M[0,2])
        b24 = (f/q**2)*(s*M[2,0] - q*M[1,0])
        b25 = (f/q**2)*(s*M[2,1] - q*M[1,1])
        b26 = (f/q**2)*(s*M[2,2] - q*M[1,2])
        return np.array([[b14,b15,b16],[b24,b25,b26]])

    # A Matrix elements for each ground coordinate 
    A1 = []
    A1.append(A(r11,s11,q11,R1))
    A1.append(A(r21,s21,q21,R2))
    A1 = np.reshape(np.ravel(A1),[4,3])

    A2 = []
    A2.append(A(r12,s12,q12,R1))
    A2.append(A(r22,s22,q22,R2))
    A2 = np.reshape(np.ravel(A2),[4,3])

    A3 = []
    A3.append(A(r13,s13,q13,R1))
    A3.append(A(r23,s23,q23,R2))
    A3 = np.reshape(np.ravel(A3),[4,3])

    A = block_diag(A1,A2,A3)


    # Misclosure
    def mis(r,s,q,im_pt):
        f = 101.4e-3
        J = im_pt[0] + f*(r/q)
        K = im_pt[1] + f*(s/q)
        return J,K

    J11,K11 = mis(r11,s11,q11,im_coord[0,0:2]) # Im 1 Pt 1
    J21,K21 = mis(r21,s21,q21,im_coord[0,2:4]) # Im 2 Pt 1

    J12,K12 = mis(r12,s12,q12,im_coord[1,0:2]) # Im 1 Pt 2
    J22,K22 = mis(r22,s22,q22,im_coord[1,2:4]) # Im 2 Pt 2

    J13,K13 = mis(r13,s13,q13,im_coord[2,0:2]) # Im 1 Pt 3
    J23,K23 = mis(r23,s23,q23,im_coord[2,2:4]) # Im 2 Pt 3
    w = [J11, K11, J21, K21, J12, K12, J22, K22, J13, K13, J23, K23]
    w = np.array(w,dtype=float)
    print(w)

    # Adjustment
    dx = inv(A.T@A)@(A.T@w)

    X1 += dx[0:3]
    X2 += dx[3:6]
    X3 += dx[6:9]
    #print(dx)

    # Residuals
    v = w - A@dx
    #print(v)


# A-posteriori
apost = v.T@v/(12-9)

# Accuracy
q = inv(A.T@A)
accuracy = m.sqrt(apost)*(q)**1/2
