import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint

def bezier_curve(my_list, len, T, t):
    n = len-1
    result = np.zeros_like(a = my_list[0], dtype=float)
    result = gp.quicksum(np.math.comb(n, i) * ((1 - (t/T)) ** (n - i)) * ((t/T) ** i) * my_list[i] for i in range(n+1))
    return result

def bezier_curve_normie(control_points, len, T, t):
    n = len-1
    result = 0
    for i in range(n + 1):
        binomial_coefficient = np.math.comb(n, i)
        polynomial_term =((1 - (t/T)) ** (n - i)) * ((t/T) ** i)
        result += binomial_coefficient * polynomial_term * control_points[i]
    return result

def diff(P, len, T):
    n = len-1
    temp = P[:,1:] - P[:,0:-1]
    temp = temp * (n/T)
    return temp

def sgn(x):
    if x < 0:
        return 0
    else:
        return 1

M = 1e6
k_1 = 0.1
k_2 = 0.01
k_3 = 0.1
k_4 = 0.01
alpha = 0.005
H = 4
C = 4


R = [50,50,50,50]
Tau = [20,80]

# kappa = np.array([np.array([
#     [1, 0, 1, 1, 0, 1, 0],
#     [0, 1, 1, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 0, 1],
#     [1, 0, 1, 1, 1, 1, 0],
#     [0, 0, 1, 1, 1, 0, 0],
#     [1, 0, 0, 1, 0, 1, 1],
#     [0, 1, 1, 0, 0, 1, 1]
# ])]) 

kappa = np.array([np.array([
    [1, 0, 1, 1],
    [0, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
])]) 


m = len(R)
k = len(Tau)

n = 4
t_0 = 100

l = n+1
test_points = np.array([random.uniform(0, t_0) for _ in range(l)])

model = gp.Model()

# objective_function = 0
 
A =  model.addMVar(shape =(m,k) , vtype = GRB.BINARY, name = 'A')
P =  model.addMVar(shape = (m,n+1) , vtype = GRB.CONTINUOUS, name = 'P')


def Temperature_curves(t, P=P, m=m ,length=n+1, t_0=t_0):
    temp_values = []
    for i in range(m):
        row = [P[i, j] for j in range(length)]
        temp_values.append(bezier_curve(row,length,t_0, t))
    temp_values = np.array(temp_values)
    return temp_values

def Temperature_curves_normie(t, P=P, m=m ,length=n+1, t_0=t_0):
    temp_values = []
    for i in range(m):
        row = [P[i, j] for j in range(length)]
        temp_values.append(bezier_curve_normie(row,length,t_0, t))
    temp_values = np.array(temp_values)
    return temp_values



def g(t, kappa = kappa, P = P , m = m, length = n+1, T = t_0, k_1 = k_1, k_2 = k_2, k_3 = k_3, k_4 = k_4, H = H, C = C, alpha = alpha):
    temp_curves = Temperature_curves(t,P,m,length,T)
    interaction_vec = alpha * (kappa * (temp_curves - temp_curves[:, np.newaxis])) 
    interaction_vec = interaction_vec[0]
    interaction_vec = interaction_vec @ np.ones((m, 1))
    diff_temp_curves = Temperature_curves(t,diff(P,length,T),m,length-1,T)
    g1_vec = diff_temp_curves - k_1*H + k_2*temp_curves - interaction_vec[:,0]
    g2_vec = diff_temp_curves + k_3*C - k_4*temp_curves - interaction_vec[:,0]
    return g1_vec, g2_vec
    
# # print(g(kappa,P,0)[1].shape)

# eps = model.addVars(m,k,l,2, vtype = GRB.CONTINUOUS, name = 'eps')
# objective_function = 0
# objective_function = gp.quicksum(
#     (1 - (sgn(R[i] - Tau[j]))) * (eps[i,j,q,0]**2)
#     + (sgn(R[i] - Tau[j])) * (eps[i,j,q,0]**2)
#     for j in range(k)
#     for i in range(m)
#     for q in range(l)
# )

# model.setObjective(objective_function, GRB.MINIMIZE)

# #Constraints

# for q in range(l):
#     for i in range(m):
#         for j in range(k):
#             model.addConstr(g(test_points[q])[0][i]-eps[i,j,q,0] <= M*(1-A[i,j]))
#             model.addConstr(g(test_points[q])[0][i]+eps[i,j,q,0] >= -M*(1-A[i,j]))
#             model.addConstr(g(test_points[q])[1][i]-eps[i,j,q,1] <= M*(1-A[i,j]))
#             model.addConstr(g(test_points[q])[1][i]+eps[i,j,q,1] >= -M*(1-A[i,j]))
#             model.addConstr(eps[i,j,q,0] <= M*(A[i,j]))
#             model.addConstr(eps[i,j,q,0] >= -M*(A[i,j]))
#             model.addConstr(eps[i,j,q,1] <= M*(A[i,j]))
#             model.addConstr(eps[i,j,q,1] >= -M*(A[i,j]))

# for i in range(m):
#     for j in range(k):
#         model.addConstr((Temperature_curves(t_0)[i]-Tau[j])*(R[i]-Tau[j]) <= M*(1-A[i,j]))

# for i in range(m):
#     model.addConstr(Temperature_curves(0)[i] == R[i])
#     model.addConstr(gp.quicksum(A[i,j] for j in range(k)) == 1)

# #Room Orientation Constraints:
# model.addConstr(gp.quicksum(A[i,1] for i in range(m)) <= 3)
# model.addConstr(gp.quicksum(A[i,1] for i in range(m)) >= 1)
# model.addConstr(gp.quicksum(A[i,0] for i in range(m)) >= 1)
# # model.addConstr(gp.quicksum(A[i,2] for i in range(m)) >= 1)
# # model.addConstr(gp.quicksum(A[i,1] for i in [1,2,3]) == 4)

# model.setParam('OutputFlag', False)
# model.optimize()
# model.update()

# if model.status == gp.GRB.OPTIMAL:
#     print(model.ObjVal)
#     for var in model.getVars():
#         var_name = var.VarName
#         var_value = model.getAttr('X')
#         # print(f'{var_name} {var_value}')

# P_hat = np.empty((m,n+1))
# A_hat = np.empty((m,k))
# for i in range(m):
#     for j in range(n+1):
#         P_hat[i, j] = P[i, j].X  # 'x' is the attribute for variable values

# for i in range(m):
#     for j in range(k):
#         A_hat[i, j] = A[i, j].X  # 'x' is the attribute for variable values

# # Printing the matrix
# print("Matrix:")
# print(A_hat)


# x = np.linspace(0, t_0, 1000)
# fig, axs = plt.subplots(m, 1)

# T_hat = []

# for i in range(m):
#     for point in x:
#         T_hat.append(Temperature_curves_normie(point, P = P_hat)[i])
#     axs[i].plot(x,T_hat)
#     axs[i].set_ylabel(f"Room {i+1}")
#     T_hat = []

# plt.show()


class HVAC_model:
    def __init__(self,M=M,k_1=k_1,k_2=k_2,k_3=k_3,k_4=k_4,alpha=alpha,H=H,C=C,R=R,Tau=Tau,kappa=kappa):
        self.M = M
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.k_4 = k_4
        self.alpha = alpha
        self.H = H
        self.C = C
        self.R = R
        self.Tau = Tau
        self.kappa = kappa
        self.m = len(R)
        self.k = len(Tau)
        
    def set_param(self,t_0,n=n):
        self.n = n
        self.t_0 = t_0
        self.l = n*2
        
    
    def optimise(self,  kappa = kappa, P = P , m = m, length = n+1, k_1 = k_1, k_2 = k_2, k_3 = k_3, k_4 = k_4, H = H, C = C, alpha = alpha):
        model = gp.Model()
        objective_function = 0
        self.test_points = np.array([random.uniform(0, self.t_0) for _ in range(self.l)])
        self.A =  model.addMVar(shape =(self.m,self.k) , vtype = GRB.BINARY, name = 'A')
        self.P =  model.addMVar(shape = (self.m,self.n+1) , vtype = GRB.CONTINUOUS, name = 'P')
        self.eps = model.addVars(self.m,self.k,self.l,2, vtype = GRB.CONTINUOUS, name = 'eps')
        objective_function = gp.quicksum(
            (1 - (sgn(self.R[i] - self.Tau[j]))) * (self.eps[i,j,q,0]**2)
            + (sgn(self.R[i] - Tau[j])) * (self.eps[i,j,q,0]**2)
            for j in range(self.k)
            for i in range(self.m)
            for q in range(self.l)
        )
        model.setObjective(objective_function, GRB.MINIMIZE)

        for q in range(self.l):
            for i in range(self.m):
                for j in range(self.k):
                    model.addConstr(g(self.test_points[q],  kappa = kappa, P = P , m = m, length = n+1, T = self.t_0, k_1 = k_1, k_2 = k_2, k_3 = k_3, k_4 = k_4, H = H, C = C, alpha = alpha)[0][i]-self.eps[i,j,q,0] <= self.M*(1-self.A[i,j]))
                    model.addConstr(g(self.test_points[q],  kappa = kappa, P = P , m = m, length = n+1, T = self.t_0, k_1 = k_1, k_2 = k_2, k_3 = k_3, k_4 = k_4, H = H, C = C, alpha = alpha)[0][i]+self.eps[i,j,q,0] >= -self.M*(1-self.A[i,j]))
                    model.addConstr(g(self.test_points[q],  kappa = kappa, P = P , m = m, length = n+1, T = self.t_0, k_1 = k_1, k_2 = k_2, k_3 = k_3, k_4 = k_4, H = H, C = C, alpha = alpha)[1][i]-self.eps[i,j,q,1] <= self.M*(1-self.A[i,j]))
                    model.addConstr(g(self.test_points[q],  kappa = kappa, P = P , m = m, length = n+1, T = self.t_0, k_1 = k_1, k_2 = k_2, k_3 = k_3, k_4 = k_4, H = H, C = C, alpha = alpha)[1][i]+self.eps[i,j,q,1] >= -self.M*(1-self.A[i,j]))
                    model.addConstr(self.eps[i,j,q,0] <= self.M*(self.A[i,j]))
                    model.addConstr(self.eps[i,j,q,0] >= -self.M*(self.A[i,j]))
                    model.addConstr(self.eps[i,j,q,1] <= self.M*(self.A[i,j]))
                    model.addConstr(self.eps[i,j,q,1] >= -self.M*(self.A[i,j]))

        for i in range(self.m):
            for j in range(self.k):
                model.addConstr((Temperature_curves(self.t_0,P=self.P, m=self.m ,length=length, t_0=self.t_0)[i]-self.Tau[j])*(self.R[i]-self.Tau[j]) <= self.M*(1-self.A[i,j]))

        for i in range(self.m):
            model.addConstr(Temperature_curves(0,P=self.P, m=self.m ,length=length, t_0=self.t_0)[i] == self.R[i])
            model.addConstr(gp.quicksum(A[i,j] for j in range(self.k)) == 1)

        #Room Orientation Constraints:
        model.addConstr(gp.quicksum(A[i,1] for i in range(self.m)) <= 3)
        model.addConstr(gp.quicksum(A[i,1] for i in range(self.m)) >= 1)
        model.addConstr(gp.quicksum(A[i,0] for i in range(self.m)) >= 1)
        # model.addConstr(gp.quicksum(A[i,2] for i in range(m)) >= 1)
        # model.addConstr(gp.quicksum(A[i,1] for i in [1,2,3]) == 4)

        model.setParam('OutputFlag', False)
        model.optimize()
        model.update()

        P_hat = np.empty((self.m,self.n+1))
        A_hat = np.empty((self.m,self.k))
        for i in range(self.m):
            for j in range(self.n+1):
                P_hat[i, j] = self.P[i, j].X  # 'x' is the attribute for variable values

        for i in range(self.m):
            for j in range(self.k):
                A_hat[i, j] = self.A[i, j].X  # 'x' is the attribute for variable values

        if model.status == gp.GRB.OPTIMAL:
            print(f"Objective value for t = {self.t_0} is {model.ObjVal}")
            print(f"Matrix for t = {self.t_0} :")
            print(A_hat)

        return model.ObjVal,A_hat

t0_values = np.linspace(0.1 , 50, 100)
objval_list = []
A_hat_list = []
for t_0 in t0_values:
    hvac_model = HVAC_model()
    hvac_model.set_param(t_0)
    objval, A_hat = hvac_model.optimise()
    objval_list.append(objval)
    A_hat_list.append(A_hat)
plt.plot(t0_values,objval_list)
plt.show()





















