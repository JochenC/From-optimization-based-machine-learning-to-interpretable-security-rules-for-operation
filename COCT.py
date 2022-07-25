# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:09:23 2018

@author: Jochen Cremer
"""

from __future__ import division
from pyomo.environ import *

import numpy as np
eps = 0.0000001

def createmodel(X, Y, i_te = [], i_tr = [], i_ca = [], D = 1, Nmin = 1, alpha = 0.001, K = 1):
    
    ################
    #Parameters
    #D maximal tree depth
    #Nmin min number of points in each terminal node
    #X = np.array([[0,0.1],[0.2,0.3],[0.1,0.4]])
    #Y = np.array([0,1,1])
   
    N = len(Y) # number of points
    N_tr = len(Y[i_tr])
    N_trca= len(Y[i_tr]) + len(Y[i_ca])
    P = np.size(X,axis=1) #number of features
    T = np.power(2,(D+1))-1 #number of nodes
    L_hat = max(sum(Y[i_tr]),N_tr-sum(Y[i_tr]))
    
    X_sorted_tr = np.zeros(shape=(N_tr,P))
    bmax = np.zeros(P)
    bmin = np.zeros(P)
    for p in range(P):
        temp_sort = np.sort(X[i_tr,p],kind='mergesort')
        X_sorted_tr[:,p] = temp_sort
        if Nmin==0:
            bmin[p] = 0
            bmax[p] = 1
        else:
            bmin[p] = temp_sort[Nmin-1]
            bmax[p] = temp_sort[N_tr-Nmin]
    bmin_all = np.min(bmin)
    bmax_all = np.max(bmax)
    
    
    epsilon = np.zeros(P) 
    for p in range(P):
        temp_sort = np.sort(X[:,p],kind='mergesort')
        temp_diff = np.zeros(len(X)-1)
        for i in range(len(X)-1): temp_diff[i] = np.abs(temp_sort[i+1] - temp_sort[i])        
        epsilon[p] = np.min(temp_diff[np.nonzero(temp_diff)])
    max_epsilon = np.max(epsilon)  
            
    model = ConcreteModel() 
    model.T = Set(initialize = range(1,T+1))
    model.TB = Set(initialize = range(1,np.int(np.round(T/2-0.5)+1)))
    model.TL = Set(initialize = range(np.int(np.round(T/2)),T+1))
    model.P = Set(initialize = range(P))
    #model.I = Set(initialize = range(N))
    model.I_te = Set(initialize = i_te)
    model.I_tr = Set(initialize = i_tr)
    model.I_ca = Set(initialize = i_ca)
    model.I_trca = Set(initialize = np.concatenate((i_tr,i_ca),axis=0))
    model.I = Set(initialize = np.concatenate((i_tr,i_te,i_ca),axis=0))
    model.L = Set(initialize = range(np.int(D)))
    
    #parent
    def pt_init(model,i): 
        if i>1: return np.int(0.5*i)
        else: return i
    model.pt = Param(model.T, initialize = pt_init)
    
    #ancestor left 
    def AL_init(model,tl):
        temp_node = tl
        temp_set = []
        while temp_node>1:        
            if temp_node == np.int(model.pt[temp_node]*2): temp_set.append(model.pt[temp_node]) 
            temp_node = model.pt[temp_node]
        return temp_set
    model.AL = Set(model.T, initialize=AL_init)    
    
    #ancestor right
    def AR_init(model,tl):
        temp_node = tl
        temp_set = []
        while temp_node>1:        
            if temp_node == np.int(model.pt[temp_node]*2+1): temp_set.append(model.pt[temp_node]) 
            temp_node = model.pt[temp_node]
        return temp_set
    model.AR = Set(model.T, initialize=AR_init)
    
    
    def TBinL_init(model,l):
        temp_set = []
        for tb in model.TB:
            c=0
            temp_node = tb
            while temp_node>1:
                temp_node = model.pt[temp_node]
                c=c+1
            if c == l: temp_set.append(tb)
        return temp_set

    model.TBinL = Set(model.L, initialize = TBinL_init)
    
    def CL_init(model,tb):
        temp_set = []
        for tl in model.TL: 
            if tb in model.AL[tl]: temp_set.append(tl)            
        return temp_set
    model.CL = Set(model.TB, initialize = CL_init)
    
    def CR_init(model,tb):
        temp_set = []
        for tl in model.TL: 
            if tb in model.AR[tl]: temp_set.append(tl)            
        return temp_set
    model.CR = Set(model.TB, initialize = CR_init)  
    
    def max_childs_init(model,tb):
        return len(model.CL[tb])
    model.max_childsperside = Param(model.TB, initialize = max_childs_init)
    
    def xL0_init(model,j):
        if Nmin==0: return 0
        else: return X_sorted_tr[Nmin-1,j]
    model.xL0 = Param(model.P, initialize = xL0_init)    

    def xU0_init(model,j):
        if Nmin==0: return 1
        else: return X_sorted_tr[N_tr-Nmin,j]
    model.xU0 = Param(model.P, initialize = xU0_init) 

    def xi_init(model,i,p): return X[i,p]   
    model.xi = Param(model.I, model.P, initialize = xi_init)
    
    def xi_max(model,i): return X[i,np.argmax(X[i,:])]
    model.xi_max = Param(model.I, initialize = xi_max)#, mutable=True)
    
    def xi_min(model,i): return X[i,np.argmin(X[i,:])]
    model.xi_min = Param(model.I, initialize = xi_min)#, mutable=True)

    def yi_init(model,i): return Y[i]  
    model.yi = Param(model.I, initialize = yi_init)
    
    #min number of points per leafes
    model.Nmin = Param(default = Nmin)
    #print(Nmin)    
    
    #normalize to baseline
    model.L_hat = Param(default = L_hat)
    
    #complexity
    model.alpha = Param(default = alpha)
    
    #scalar weight for unreliable point
    model.K = Param(default = K)
    
    #small epsilon
    model.epsilon_max = Param(default = max_epsilon)
    model.epsilon_min = Param(default=epsilon[np.argmin(epsilon)])    
    model.epsilon_mid = Param(default = (epsilon[np.argmax(epsilon)] + epsilon[np.argmin(epsilon)])/2, mutable=True) 
    epsilon_mid = (epsilon[np.argmax(epsilon)] + epsilon[np.argmin(epsilon)])/2
    
    
    def epsilon_init(model,p): return epsilon[p]
    model.epsilon = Param(model.P, initialize = epsilon_init)    
    
    #minimal split threshold
    model.bmin_all = Param(default = bmin_all)

    #maximal split threshold
    model.bmax_all = Param(default = bmax_all)
        
    #tree depth
    model.D = Param(default = D) 
    
#OCT*
    model.bt = Var(model.TB, within=NonNegativeReals, bounds=(0, bmax_all), initialize=0.5)

#OCT
# =============================================================================
#     model.bt = Var(model.TB, within=NonNegativeReals, bounds=(0, 1), initialize=0.5)  
# =============================================================================
    
    
    model.ajt = Var(model.P, model.TB, within=Boolean, initialize=0)
    model.dtb = Var(model.TB, within=Boolean, initialize=0)
    model.zit = Var(model.I, model.TL, within=Boolean, initialize=0)
    model.lt = Var(model.TL, within=Boolean, initialize=0)
    model.Nkt = Var(model.TL, within=NonNegativeReals, initialize=1)
    model.Nt = Var(model.TL, within=NonNegativeReals, initialize=1)
    model.Ntte = Var(model.TL, within=NonNegativeReals, initialize=0)
    model.Lt = Var(model.TL, within=NonNegativeReals, initialize=1)
    model.ckt = Var(model.TL, within=Boolean, initialize=1)

    
    #Constraints
    def feature_selection(model,tb):
        return sum(model.ajt[j,tb] for j in model.P) == model.dtb[tb]
    model.const_fs = Constraint(model.TB, rule = feature_selection, doc='Equal')
    
#OCT*
    def split_selection(model,tb):
        return model.bt[tb] <= model.bmax_all* model.dtb[tb]
    model.const_ss = Constraint(model.TB, rule = split_selection, doc='Inequal')
  
    
    
#OCT
# =============================================================================
#     def split_selection(model,tb):
#         return model.bt[tb] <= model.dtb[tb]
#     model.const_ss = Constraint(model.TB, rule = split_selection, doc='Inequal')
# =============================================================================


    
    def parent_selection(model,t):
        if t==1: return Constraint.Skip
        else: return model.dtb[t] <= model.dtb[model.pt[t]]
    model.const_ps = Constraint(model.TB, rule = parent_selection)
    
    def point_assignment(model,i,tl):
        return model.zit[i,tl] <= model.lt[tl]
    model.const_pa = Constraint(model.I, model.TL, rule = point_assignment)    
    
    
    def min_points(model,tl):
        if model.Nmin == 0:
            return Constraint.Skip        
        else:
            return sum(model.zit[i,tl] for i in model.I_tr) >= model.Nmin * model.lt[tl]
    #model.const_mp = Constraint(model.TL, rule = min_points)
        
    def point_distribution(model,i):
        return sum(model.zit[i,tl] for tl in model.TL) == 1
    model.const_pd = Constraint(model.I, rule = point_distribution)

# #OCT + conditions  
# =============================================================================
#     def split_constraint_leq2(model,i,tb):
#         return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) <= model.bt[tb] + 1 * ( 1 - sum(model.zit[i,tl] for tl in model.CL[tb]) )
#     model.const_scl2 = Constraint(model.I, model.TB, rule = split_constraint_leq2)
#     
#     def split_constraint_geq2(model,i,tb):
#         return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) >= 0.001 + model.bt[tb] -1.001 *( 1 - sum(model.zit[i,tl] for tl in model.CR[tb]) )
#     model.const_scg2 = Constraint(model.I, model.TB, rule = split_constraint_geq2)  
# 
# =============================================================================


# =============================================================================
# #OCT + big -M
#     def split_constraint_leq2(model,i,tb,tl):
#         if tl in model.CL[tb]:
#             return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) <= model.bt[tb] + (- model.bmin_all + model.xi_max[i] )* ( 1 - model.zit[i,tl] )
#         else: return Constraint.Skip
#     model.const_scl2 = Constraint(model.I, model.TB, model.TL, rule = split_constraint_leq2)
#     
#     def split_constraint_geq2(model,i,tb,tl):
#         if tl in model.CR[tb]:
#             return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) >= 0.001 + model.bt[tb] + (-model.bmax_all - 0.001 + model.xi_min[i] ) *( 1 - model.zit[i,tl] ) 
#         else: return Constraint.Skip
#     model.const_scg2 = Constraint(model.I, model.TB, model.TL, rule = split_constraint_geq2)  
# 
# =============================================================================

#OCT
# =============================================================================
#     def split_constraint_leq2(model,i,tb,tl):
#         if tl in model.CL[tb]:
#             return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) <= model.bt[tb] + 1 * ( 1 - model.zit[i,tl] )
#         else: return Constraint.Skip
#     model.const_scl2 = Constraint(model.I, model.TB, model.TL, rule = split_constraint_leq2)
#     
#     def split_constraint_geq2(model,i,tb,tl):
#         if tl in model.CR[tb]:
#             return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) >= 0.001 + model.bt[tb] -1.001 *( 1 - model.zit[i,tl] ) 
#         else: return Constraint.Skip
#     model.const_scg2 = Constraint(model.I, model.TB, model.TL, rule = split_constraint_geq2)  
# =============================================================================

#OCT*   
    def split_constraint_leq2(model,i,tb):
        return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) <= model.bt[tb] + (- model.bmin_all + model.xi_max[i] ) * ( 1 - sum(model.zit[i,tl] for tl in model.CL[tb]) )
    model.const_scl2 = Constraint(model.I, model.TB, rule = split_constraint_leq2)
    
    def split_constraint_geq2(model,i,tb):
        return sum(model.ajt[p,tb] * model.xi[i,p] for p in model.P) >= 0.001 + model.bt[tb] + (-model.bmax_all - 0.001 + model.xi_min[i] )*( 1 - sum(model.zit[i,tl] for tl in model.CR[tb]) ) 
    model.const_scg2 = Constraint(model.I, model.TB, rule = split_constraint_geq2) 


    def num_classlabels(model,tl):
        return model.Nkt[tl] == sum(model.yi[i]*model.zit[i,tl] for i in model.I_tr)
    model.const_nc = Constraint(model.TL, rule = num_classlabels)
    
    def num_classlabels_refit(model,tl):
        return model.Nkt[tl] == sum(model.yi[i]*model.zit[i,tl] for i in model.I_trca)
    model.const_nc_refit = Constraint(model.TL, rule = num_classlabels_refit)
    model.const_nc_refit.deactivate()
    
    
    
    def num_points(model,tl):
        return model.Nt[tl] == sum(model.zit[i,tl] for i in model.I_tr)
    model.const_np = Constraint(model.TL, rule = num_points)
    
    def num_points_refit(model,tl):
        return model.Nt[tl] == sum(model.zit[i,tl] for i in model.I_trca)
    model.const_np_refit = Constraint(model.TL, rule = num_points_refit)
    model.const_np_refit.deactivate()
    
    
    
    def num_pointstest(model,tl):
        return model.Ntte[tl] == sum(model.zit[i,tl] for i in model.I_te)
    model.const_nptest = Constraint(model.TL, rule = num_pointstest)
    

    

    
    def Lt_greater_1(model,tl):
        return model.Lt[tl] >= model.K * (model.Nt[tl] - model.Nkt[tl]) - model.K* N_tr * (1 - model.ckt[tl])
    model.const_Lg1 = Constraint(model.TL, rule = Lt_greater_1)
    
    def Lt_greater_1_refit(model,tl):
        return model.Lt[tl] >= model.K * ( model.Nt[tl] - model.Nkt[tl] ) - N_trca * (1 - model.ckt[tl])
    model.const_Lg1_refit = Constraint(model.TL, rule = Lt_greater_1_refit)
    model.const_Lg1_refit.deactivate()
    
    
    
    
    
    def Lt_greater_2(model,tl):
        return model.Lt[tl] >= model.Nkt[tl] - N_tr * model.ckt[tl]
    model.const_Lg2 = Constraint(model.TL, rule = Lt_greater_2)
    
    def Lt_greater_2_refit(model,tl):
        return model.Lt[tl] >= model.Nkt[tl] - N_trca * model.ckt[tl]
    model.const_Lg2_refit = Constraint(model.TL, rule = Lt_greater_2_refit)    
    model.const_Lg2_refit.deactivate()
    
    
    
    
    def Lt_less_1(model,tl):
        return model.Lt[tl] <= model.K* ( model.Nt[tl] - model.Nkt[tl]) + model.K * N_tr * model.ckt[tl]
    model.const_Ll1 = Constraint(model.TL, rule = Lt_less_1)
        
    def Lt_less_1_refit(model,tl):
        return model.Lt[tl] <= model.K * (model.Nt[tl] - model.Nkt[tl]) + model.K * N_trca * model.ckt[tl]
    model.const_Ll1_refit = Constraint(model.TL, rule = Lt_less_1_refit)   
    model.const_Ll1_refit.deactivate()


    
    
    def Lt_less_2(model,tl):
        return model.Lt[tl] <= model.Nkt[tl] + N_tr * (1 - model.ckt[tl])
    model.const_Ll2 = Constraint(model.TL, rule = Lt_less_2)

    def Lt_less_2_refit(model,tl):
        return model.Lt[tl] <= model.Nkt[tl] + N_trca * (1 - model.ckt[tl])
    model.const_Ll2_refit = Constraint(model.TL, rule = Lt_less_2_refit)    
    model.const_Ll2_refit.deactivate()
    
    
    
    #Objective Function
    def min_loss(model): return 1/model.L_hat * sum(model.Lt[tl] for tl in model.TL) #+ model.alpha * sum(model.dtb[tb] for tb in model.TB)
    model.objective = Objective(rule=min_loss, sense=minimize, doc='Define objective function')
    
    return model #, train_error, test_error    