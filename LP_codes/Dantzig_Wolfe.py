# This notebook programs the column generation technique in Chvatal's "Linear Programming" textbook. 
# The correspondence between the variables in this program and in the textbook should be easy to 
# figure.

# Hopefully the only thing you need to change is Solve_sub_LP() that allows you to use different LP
# solvers. The current solver that we use now is PULP.

# A technical issue that needed to be addressed was how to find a basic feasible direction in the 
# unbounded case when the subproblem is unbounded. My first attempt was to find a solver that can 
# give such a basic feasible direction, but I wasn't able to find one on Python. So we had to go back
# to the theory.  For the particular case in the textbook, to find a basic feasible direction in the 
# unbounded case for 
# max c@x s.t. A@x=b, x>=0, we solve the problem
# max c@x s.t. A@x=0, e@x=1, x>=0, where e is the vector of all ones. 

# The numbers are not exactly the same as what was in the book because some columns can be scaled. 

import pulp as pl
from pulp import *
import numpy as np
np.set_printoptions(precision=2)

def Dantzig_Wolfe(c,A1,A2,b1,b2):
    # The example problem in Chvatal's textbook: To solve 
    # min c@x s.t. A1@x==b1, A2@x==b2, x>=0.
    
    n1=len(c)
    m1=len(b1)
    m2=len(b2)
    n2=m1+1
    def Solve_sub_LP(temp):
        # This routine solves 
        #   max temp @ x s.t. A2@x==b2, x>=0.
        # We follow the conventions in PULP. 
        # If s_state==1 (can find optimum), then the solution for prob is a BFS for the subproblem.
        # If s_state==-2 (unbounded problem), then the solution for prob is a Basic Feasible 
        #    Direction (BFD) for the subproblem.
        # Outputs:
        #    s_state: 1 if subproblem bounded, -2 if unbounded.
        #    v_sol: The BFS if s_state==1, or BFD if s_state==-2.

        # Now we set the LP in the middle of page 428. 
        prob = LpProblem("Inner_subproblem", LpMaximize)
        x = pulp.LpVariable.dicts("v", range(n1), lowBound=0, upBound=None)
        prob += pulp.lpSum(x[i]*temp[i] for i in range(n1)),"obj"
        for j in range(m2):
            prob += pulp.lpSum(x[i]*A2[j,i] for i in range(n1))==b2[j],"C"+str(j)
        s_state=prob.solve(PULP_CBC_CMD(msg=False))

        # For the case of unbounded direction, we find a basic feasible direction
        # (Hope that there is an LP solver in Python that can output a basic feasible direction
        #  so that this code can be dropped.)
        if s_state==-2:
            prob = LpProblem("Inner_subproblem", LpMaximize)
            x = pulp.LpVariable.dicts("v", range(n1), lowBound=0, upBound=None)
            prob += pulp.lpSum(x[i]*temp[i] for i in range(n1)),"obj"
            for j in range(m2):
                prob += pulp.lpSum(x[i]*A2[j,i] for i in range(n1))==0,"C"+str(j)
            prob += pulp.lpSum(x[i] for i in range(n1))<=1,"cap"
            print(prob.solve(PULP_CBC_CMD(msg=False)))
        
        # Collecting the solution vector.
        v_sol=np.zeros(n1)
        for i in range(n1):       
            v_sol[i]=prob.variablesDict()['v_'+str(i)].varValue
            if abs(v_sol[i])<1e-13: v_sol[i]=0
        print(("BFS" if s_state==1 else "Basic Feasible Direction")+" of subpblm:",v_sol)
        return s_state, v_sol

    def DW_iter(B_mat,cb,xb,cols):
        # The main Dantzig Wolfe iterations

        y=np.linalg.solve(B_mat.T,cb)
        temp=c-y[:-1]@A1
        print("y:",y); print("c- y@A1: ",temp)

        # Solve the subproblem to get a BFS if there is an optimal solution, 
        #   or a BFD (Basic Feasible Direction) if unbounded
        s_state, v_sol=Solve_sub_LP(temp)
        
        cum_sum=v_sol@temp
        print(cum_sum)

        # A new column can either result from a BFS or BFD in subproblem.

        new_col_from_BFS=(cum_sum-y[-1])/max(abs(cum_sum),abs(y[-1]))>1e-13 and s_state==1
        new_col_from_BFD=cum_sum>0 and s_state==-2

        if new_col_from_BFS or new_col_from_BFD:
            at=np.concatenate((A1 @ v_sol,[1] if new_col_from_BFS else [0]),axis=0)
            d=np.linalg.solve(B_mat,at)
            print("d:",d)

            t=np.inf
            idx=-1
            for i in range(m1+1):
                if d[i]>0:
                    if xb[i]/d[i]<t:
                        t=xb[i]/d[i]
                        idx=i
            xb=xb-t*d
            xb[idx]=t        
            cb[idx]=(c @ v_sol)        
            B_mat[:,idx]=at        
            cols[:,idx]=v_sol

            print("xb:",xb); print("cb:",cb); print(B_mat); print(cols)
            return B_mat,cb,xb,cols,True
        elif abs((cum_sum-y[-1])/max(abs(cum_sum),abs(y[-1])))<1e-13:
            print("Solution optimal up to numerical precision")
            print("Solution to original pblm:",cols @ xb)
            return B_mat,cb,xb,cols,None
        else:
            print("Ran into unexpected problems")
            return B_mat,cb,xb,cols,None  
        
    def phase_1(B_mat,xb,ct,cols):
        # Phase 1 to find a first BFS of adjusted problem.
        
        y=np.linalg.solve(B_mat.T,ct)
        temp=-y[:-1]@A1
        print("y:",y); print("-y@A1: ",temp)
        
        # Solve the subproblem to find either a BFS or a Basic Feasible Direction
        s_state, v_sol=Solve_sub_LP(temp) 
        
        at=np.concatenate((A1 @ v_sol,[1] if s_state==1 else [0]),axis=0)
        print("at: ", at)

        # The case where there is a match of columns.
        for i in range(m1+1):
            at_norm=at/np.linalg.norm(at)
            B_col_norm=B_mat[:,i]/np.linalg.norm(B_mat[:,i])
            ratio=np.linalg.norm(B_mat[:,i])/np.linalg.norm(at)
            if np.linalg.norm(at_norm-B_col_norm)/np.linalg.norm(at_norm) < 1e-13:
                ct[i]=0
                print("Match of columns, so not calculating d")                              
                cols[:,i]=ratio*v_sol
                print("xb:",xb); print(B_mat); print(cols)
                return B_mat, xb, ct

        d=np.linalg.solve(B_mat,at)
        print("d:",d)

        t=np.inf
        idx=-1
        if s_state==-2:
            for i in range(m1+1):
                if d[i]<0:
                    if xb[i]/d[i]<t:
                        t=xb[i]/d[i]
                        idx=i
        elif s_state==1:
            for i in range(m1+1):
                if d[i]>0:
                    if xb[i]/d[i]<t:
                        t=xb[i]/d[i]
                        idx=i
        xb=xb-t*d
        xb[idx]=t
        B_mat[:,idx]=at        
        ct[idx]=0
        cols[:,idx]=v_sol
        
        print("xb:",xb); print(B_mat); print(cols)
        return B_mat, xb, ct
    
    print("STARTING PHASE 1 TO FIND A BFS\n")
    xb=np.concatenate((b1,[1.]),axis=0)
    B_mat=np.eye(m1+1).astype(float)
    ct=np.ones(m1+1).astype(float)
    for i in range(m1):
        if b1[i]>0: ct[i]=-1.
    ct[m1]=-1
    cols=np.zeros((n1,m1+1))
    B_mat,xb,ct=phase_1(B_mat,xb,ct,cols)
    for j in range(m1):
        print("\nNext iteration: \n")
        B_mat,xb,ct=phase_1(B_mat,xb,ct,cols)
        
    cb=c@cols
    print("Starting cb:",cb)
    
    print("\n COMPLETED PHASE 1, now doing Dantzig Wolfe iterations.\n")
    
    B_mat, cb, xb,cols,status=DW_iter(B_mat,cb,xb,cols)
    while status is not None:
        print("\nNext iteration: \n")
        B_mat, cb, xb,cols,status=DW_iter(B_mat,cb,xb,cols)
        
if __name__=="__main__":    
    # Run Dantzig Wolfe on the example in Chvatal's textbook.
    A1=np.array([[2,1,-2,-1,2,-1,-2,-3],[1,-3,2,3,-1,2,1,1]]).astype(float)
    A2=np.array([[-1,0,1,0,1,0,0,0],[1,-1,0,1,0,0,0,0],[0,1,-1,0,0,1,-1,0],
                 [0,0,0,-1,0,-1,0,1],[0,0,0,0,-1,0,1,-1]]).astype(float)
    b1=np.array([4,-2]).astype(float)
    b2=np.array([-3,1,4,3,-5]).astype(float)
    c=np.array([9,-1,-4,-2,8,-2,-8,-12]).astype(float)

    Dantzig_Wolfe(c,A1,A2,b1,b2)    
