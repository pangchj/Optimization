# We try to do cutting stock from the example in Chvatal's textbook. Please read the book for more details.

import pulp as pl, numpy as np, math
from pulp import *
np.set_printoptions(precision=4)

def knapsack(w,c,b):
    # Outputs the solution of maximize c @ a s.t. w @ a <= b
    
    n=len(w)
    prob = LpProblem("Knapsack", LpMaximize)
    a = LpVariable.dicts("a", range(n),lowBound=0, upBound=None, cat=pl.LpInteger)
    prob+=lpSum(a[i]*c[i] for i in range(n))    , "obj"
    prob+=lpSum(a[i]*w[i] for i in range(n))<=b , "constraint"
    prob.solve(PULP_CBC_CMD(msg=False))
    
    output=np.zeros(n)
    for i in range(n):
        output[i]=prob.variablesDict()['a_'+str(i)].varValue
    cum_sum=output @ c
    print("knapsack output:", output); print("output @ c:",cum_sum)
    
    return output,cum_sum    

def CS_iter(B_mat,w,b,finals):
    # Performs one iteration for the cutting stock problem
    n=len(w)
    y=np.linalg.solve(B_mat.T,np.ones(n))
    rhs=np.linalg.solve(B_mat,finals)
    print("y   :",y); print("rhs :",rhs)
    
    o1,o2=knapsack(w,y,b)
    if o2<=1+1e-7:
        print("Algorithm ends")
        return None, (B_mat,rhs)
    d=np.linalg.solve(B_mat,o1)
    print("d   :",d)
    
    t=np.inf
    idx=-1
    for i in range(n):
        if d[i]>0:
            if rhs[i]/d[i]<t:
                t=rhs[i]/d[i]
                idx=i
    rhs=rhs-t*d
    rhs[idx]=t
    B_mat[:,idx]=o1
    print(B_mat,"\n")
    return B_mat, None    

def cutting_stock(finals,w,b):
    n=len(w)
    if len(finals)!= n:
        print("The data not good")
        return
    
    # Initialize with pure types.
    B_mat=np.zeros((n,n))
    rhs=np.zeros(n)
    for i in range(n):
        B_mat[i,i]=math.floor(b/w[i])
        rhs[i]=finals[i]/B_mat[i,i]        
    print(B_mat,rhs)
    
    # Perform iterations till get optimum.
    while B_mat is not None:
        B_mat,c2=CS_iter(B_mat,w,b,finals)        
    print(c2)
    
if __name__=="__main__":
    # Two examples in Chvatal's textbook
    #cutting_stock([78,40,30,30],[25.5,22.5,20,15],91)
    cutting_stock([97,610,395,211],[45,36,31,14],100)    
    
    # To test the knapsack subroutine.
    '''knapsack([25.5,22.5,20,15],[1/3,1/4,1/4,1/6],91)
    knapsack([25.5,22.5,20,15],[1/3,1/4,1/6,1/6],91)
    knapsack([25.5,22.5,20,15],[7/24,1/4,5/24,1/6],91)'''
