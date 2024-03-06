import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = r"Housing.csv"

d = pd.read_csv(file_path)
ycol = "price"


def makespecial(d):
    axes = d.axes[1]
    n = len(axes)
    for column in axes : 
        unique  = d[column].unique()
        nu = len(unique)
        if(d[column].dtype != "object"):
            continue
        mapping = {unique[i]:i for i in range(nu) }
         
        d[column] = d[column].replace(mapping)


def cov(a1,a2):
    ma1 = np.mean(a1)
    ma2 = np.mean(a2)
    size =len(a1)
    s=0
    for p1,p2 in zip(a1,a2):
        s+=(p1-ma1)*(p2-ma2)
    return s/size
def cor(a1,a2):
    return cov(a1,a2) / (cov(a1,a1)*cov(a2,a2))**0.5
def coef(a1,a2):
    return cov(a1,a2) / cov(a2,a2)

def dif(a1,a2):
    return np.mean(a1)- coef(a1,a2)*np.mean(a2)

def derror(value,estimaded):
    s = 0
    for e1,e2 in zip(value,estimaded):
        s = (e1 - e2)**2
    return s**0.5

def lr(y,data): 
    weights = []
    a = np.mean(y)
    for d in data :
        Coef = coef(y,d) 
        a-= np.mean(d)* Coef
        weights.append(Coef)
    """
    s=str(a)
    for i,weight in enumerate(weights):
        s+="+" + str(weight)+ "y"+ str(i)
    print(s)
    """
    return  lambda  *xss  :[a + sum(x*weight for x,weight in zip(xs,weights) ) for xs in xss ] 


makespecial(d)
y = np.array(d.pop(ycol))
traind = np.array(d.T.values)


nn = lr(y,traind)
result = nn(d.values[0])
print("predicted : ",result)
print("real : ",y[0])
print("error : ", derror(y,result))