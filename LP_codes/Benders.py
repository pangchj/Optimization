# The nicest writeup of Bender's decomposition is in Leon Lasdon's book 
# "Optimization Theory for large systems", so we follow his notation.

# WARNING: THE CASE WHERE THE SUBPROBLEM IS UNBOUNDED IS NOT ENCOUNTERED IN THIS 
#  EXAMPLE AND THUS NOT IMPLEMENTED.

# In the book, the main problem is P1:
#  min c @ x + f1(y) s.t. A@x +F2(y)>=b, x>=0, y in S.
# The functions f1() and F2() need not even be linear.
# Fixing y gives the subproblem (15). Taking the dual of (15) gives (16)
# The subproblem P2 calculates a lower bound for the objective value. 
#  This lower bound continually increases.

# The structure of this file is as follows. 
#  cut,x=solve_for_dual(y):
#   This program gives a cut and primal solution x when given variable y.
#   Unfortunately, the textbook doesn't give an example of an unbounded subproblem, 
#   so cuts from basic feasible directions is not implemented.
#
#  lb,y=solve_for_y(constraints):
#   From the constraints assembled from the cuts, we find the lower bound and y.
# 
#  Benders(solve_for_dual,solve_for_y,y):
#   Shows how the Benders decomposition is done.

import pulp as pl, numpy as np
from pulp import *

def solve_for_dual(y):
    #print("input: ", y)
    n=4
    f=np.ones(4)*7
    #print(f)
    c=np.array([0,12,20,18,12,0,8,6,20,8,0,6,18,6,6,0]).astype(float)
    #print(c)
    big_y=np.zeros(16)
    for i in range(4):
        for j in range(4):
            big_y[i*4+j]=y[i]
    #print(big_y)

    prob = LpProblem("Benders_Dual_subproblem", LpMaximize)
    v = LpVariable.dicts("v", range(4),lowBound=0, upBound=None)
    u = LpVariable.dicts("u", range(16),lowBound=0, upBound=None)

    prob+=lpSum(v[j] for j in range(4)) - lpSum(big_y[j]*u[j] for j in range(16) )+f@y   , "obj"
    for i in range(4):
        for j in range(4):
            prob+= v[j]-u[i*4+j]<=c[i*4+j], "x_"+str(i)+"_"+str(j) 
    temp=prob.solve(PULP_CBC_CMD(msg=False))
    if temp==1:
        print("BFS of dual subproblem found")
    else:
        print("Exceptional case seen")

    output_v=np.zeros(4)
    for i in range(4):
        output_v[i]=prob.variablesDict()["v_"+str(i)].varValue
    output_u=np.zeros(16)
    for i in range(16):
        output_u[i]=prob.variablesDict()["u_"+str(i)].varValue

    #print(output_v,output_u)
    #print(output_v@np.ones(4)+output_u@big_y)
    
    rhs=np.array(list(7-sum(output_u[i*4+j] for j in range(4)) for i in range(4)))
    #print(output_v@np.ones(4), rhs)
    x=np.zeros(16)
    for a1,a2 in list(prob.constraints.items()):
        idxs=a1.split("_")
        x[int(idxs[1])*4+int(idxs[2])]=a2.pi
    #print(x)
    return np.concatenate((np.array([output_v@np.ones(4)]), rhs),axis=0),x
    
# We implement a custom method for solving the integral part of the problem.

def solve_for_y(constraints):
    rows=constraints.shape[0]
    lb_so_far=np.inf
    best_y=[-1,-1,-1,-1]
    for i0 in range(2):
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    ub_so_far=-np.inf
                    y=np.array([1,i0,i1,i2,i3])
                    for j in range(rows):
                        temp = y @ constraints[j,:]
                        if temp>ub_so_far:
                            ub_so_far=temp
                    if ub_so_far<lb_so_far:
                        lb_so_far=ub_so_far
                        best_y=y[1:]
    #print(lb_so_far, best_y)
    return lb_so_far, best_y
    
def Benders(solve_for_dual,solve_for_y,y):
    print("Startng y          :",y)
    c=np.array([0,12,20,18,12,0,8,6,20,8,0,6,18,6,6,0]).astype(float)
    f=np.ones(4)*7
    cut,x=solve_for_dual(y)
    ub=c@x+f@y
    print("cut found          :",cut)
    print("subpblm primal soln:", x)
    print("Upper bound        :",ub)
    lb=-np.inf
    cut=np.expand_dims(cut, axis=0)
    while lb<ub-1e-13:
        print("\n*** Next iteration ***")
        lb,y=solve_for_y(cut)
        new_cut,x=solve_for_dual(y)
        print("Lower bdd:",lb);print("New y    :",y); print("New cut  :",new_cut)
        new_cut=np.expand_dims(new_cut, axis=0)
        temp=c@x+f@y
        if temp<ub: ub=temp        
        print("subpblm primal soln:", x)
        print("New primal upp bdd :",temp); print("Upper bound        :",ub)
        cut=np.concatenate((cut,new_cut),axis=0)

Benders(solve_for_dual,solve_for_y,np.array([0,1,0,0]))
