import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.integrate import odeint
import time

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


M = 1e7
k_1 = 1
k_2 = 0.06
k_3 = 1
k_4 = 0.06
alpha = 0
H = 4
C = 4


R = [50,50,50,50,50,50,50,50,50,50,50,50]
Tau = [35,60]


kappa = np.array([
    [1, 0.2, 0.1,  0.01, 0.01, 0.01],
    [0.2, 1, 0.3, 0,      0,      0],
    [0.1, 0.3, 1, 0.05,  0.04, 0.03],
    [0.01, 0, 0.05, 1,     0,  0.01],
    [0.01, 0, 0.04, 0,    1,   0.01 ],
    [0.01, 0, 0.03, 0.01,    0.01,   1 ]
])

kappa = np.diag([1,1,1,1,1,1,1,1,1,1,1,1])


m = len(R)
k = len(Tau)

n = 2 #1 and 1e14, 2 and 1e11 , 4 and 1e8, 6 and 1e8
t_0 = 20



def Temperature_curves(t, P, m,length, t_0):
    temp_values = []
    for i in range(m):
        row = [P[i, j] for j in range(length)]
        temp_values.append(bezier_curve(row,length,t_0, t))
    temp_values = np.array(temp_values)
    return temp_values


def Temperature_curves_normie(t, P, m ,length, t_0):
    temp_values = []
    for i in range(m):
        row = [P[i, j] for j in range(length)]
        temp_values.append(bezier_curve_normie(row,length,t_0, t))
    temp_values = np.array(temp_values)
    return temp_values


def g(t, P, kappa , m , length , T , k_1 , k_2 , k_3 , k_4 , H , C , alpha ):
    temp_curves = Temperature_curves(t,P,m,length,T)
    interaction_vec = alpha * (kappa * (temp_curves - temp_curves[:, np.newaxis]))     
    interaction_vec = interaction_vec @ np.ones((m, 1))
    diff_temp_curves = Temperature_curves(t, diff(P,length,T), m, length-1, T)
    g1_vec = diff_temp_curves - k_1 * H * np.ones(m) + k_2*temp_curves*np.ones(m) - interaction_vec[:,0]
    g2_vec = diff_temp_curves + k_3 * C * np.ones(m) - k_4*temp_curves*np.ones(m) - interaction_vec[:,0]
    return g1_vec, g2_vec


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
        
    def set_time(self,t_0,n = n):
        self.n = n
        self.t_0 = t_0
        self.l = n+1
        
    def optimise(self):
        model = gp.Model()
        objective_function = 0
        self.test_points = np.array([random.uniform(0+1e-3, self.t_0-1e-3) for _ in range(self.l)])
        print(f'Test Points are {self.test_points}')
        self.A =  model.addMVar(shape =(self.m,self.k) ,lb=0, ub=2 , vtype= GRB.INTEGER, name = 'A')
        self.P =  model.addMVar(shape = (self.m,self.n+1) , lb = 0, ub = 100, vtype = GRB.CONTINUOUS, name = 'P')
        self.eps = model.addVars(self.m,self.k,self.l,2, lb = 0, vtype = GRB.CONTINUOUS, name = 'eps')

        # for i in range(self.m):
        #     for j in range(self.k):
        #         objective_function += self.A[i,j]*((self.R[i]-self.Tau[j])**2)

        for q in range(self.l):
            for i in range(self.m):
                for j in range(self.k):
                    objective_function = objective_function + ((1 - (sgn(self.R[i] - self.Tau[j]))) * (self.eps[i,j,q,0]) * (self.eps[i,j,q,0])) + ((sgn(self.R[i] - Tau[j])) * (self.eps[i,j,q,1]) * (self.eps[i,j,q,1]))

        model.setObjective(objective_function, GRB.MINIMIZE)

        for q in range(self.l):
            g1_vec, g2_vec = g(self.test_points[q], P = self.P , kappa = self.kappa, m = self.m, length = self.n+1, T = self.t_0, k_1 = self.k_1, k_2 = self.k_2, k_3 = self.k_3, k_4 = self.k_4, H = self.H, C = self.C, alpha = self.alpha)
            for i in range(self.m):
                for j in range(self.k):
                    model.addConstr(g1_vec[i]-self.eps[i,j,q,0] <= self.M*(1-self.A[i,j]))
                    model.addConstr(g1_vec[i]+self.eps[i,j,q,0] >= self.M*(self.A[i,j]-1))
                    model.addConstr(g2_vec[i]-self.eps[i,j,q,1] <= self.M*(1-self.A[i,j]))
                    model.addConstr(g2_vec[i]+self.eps[i,j,q,1] >= self.M*(self.A[i,j]-1))
                    model.addConstr(self.eps[i,j,q,0] <= self.M*(self.A[i,j]))
                    model.addConstr(self.eps[i,j,q,0] >= -1*self.M*(self.A[i,j]))
                    model.addConstr(self.eps[i,j,q,1] <= self.M*(self.A[i,j]))
                    model.addConstr(self.eps[i,j,q,1] >= -1*self.M*(self.A[i,j]))
            
        temp_final_values = Temperature_curves(self.t_0, P = self.P , m = self.m , length = self.n + 1, t_0 = self.t_0)

        for i in range(self.m):
            for j in range(self.k):
                model.addConstr( ( temp_final_values[i] - self.Tau[j] )*( self.R[i] - self.Tau[j] ) <= self.M * (1-self.A[i,j])-1e-6 )

        for i in range(self.m):
            model.addConstr(Temperature_curves(0, P = self.P, m = self.m , length = self.n + 1, t_0 = self.t_0)[i] == self.R[i])
            model.addConstr(gp.quicksum(self.A[i,j] for j in range(self.k)) == 1)

        #Room Orientation Constraints:
        # model.addConstr(gp.quicksum(A[i,1] for i in range(self.m)) <= 3)
        model.addConstr(gp.quicksum(self.A[i,1] for i in range(self.m)) >= 1)
        model.addConstr(gp.quicksum(self.A[i,0] for i in range(self.m)) >= 1)
        # model.addConstr(gp.quicksum(A[i,2] for i in range(m)) >= 1)
        model.addConstr(gp.quicksum(self.A[i,1] for i in [1,2,3]) == 2)

        model.setParam('OutputFlag', False)
        model.optimize()
        model.update()

        P_hat = np.empty((self.m,self.n+1))
        A_hat = np.empty((self.m,self.k))
        if model.status == gp.GRB.OPTIMAL:
            for i in range(self.m):
                for j in range(self.n+1):
                    P_hat[i, j] = self.P[i, j].X  # 'x' is the attribute for variable values

            for i in range(self.m):
                for j in range(self.k):
                    A_hat[i, j] = self.A[i, j].X  # 'x' is the attribute for variable values

            print(f"Objective value for t = {self.t_0} is {model.ObjVal}")
            print(f"Matrix for t = {self.t_0} :")
            print(P_hat)
            print(A_hat)
        else:
            print("Model didn't converge")
        
        

        return model.ObjVal,A_hat,P_hat

start = time.time()
hvac_model = HVAC_model()
hvac_model.set_time(t_0)
objval, A_hat, P_hat = hvac_model.optimise()
end = time.time()
print(f'Time taken for HVAC System {end - start}')



# def odes(x, t, control_points_f_normie = control_points_f_normie, len = n+1, T = T):
#     # # assign each ODE to a vector element
#     h1 = x[0]
#     h2 = x[1]
#     f = bezier_curve_normie(control_points_f_normie,len,T,t)
#     # # define each ODE
#     dh1dt =  (f/rho) - (a/A)*h1
#     dh2dt = (a/A)*h1

#     return [dh1dt, dh2dt]
    
# x0 = [0, 0]

# # print(odes(x0, control_points_f_normie, n+1, T ,t=0))
# h = odeint(odes,x0,x)

# h1 = h[:,0]
# h2 = h[:,1]


def helper(A_hat, R = R, Tau = Tau, k_1=k_1,k_2=k_2,k_3=k_3,k_4=k_4,alpha=alpha,H=H, C=C , kappa = kappa):
    m = A_hat.shape[0]
    k = A_hat.shape[1]
    mode_vec = np.zeros(m)
    diagonal_A = np.zeros((m,m))
    for i in range(m):
        for j in range(k):
            if A_hat[i,j] == 1:
                if R[i] >= Tau[j] :
                    mode_vec[i] = -k_3*C
                    diagonal_A[i,i] = k_4 - alpha * np.sum(kappa[i])
                elif R[i] < Tau[j] :
                    mode_vec[i] = k_1*H
                    diagonal_A[i,i] = -k_2 - alpha * np.sum(kappa[i])

    return mode_vec, diagonal_A
                
print(helper(A_hat)[0])
print(helper(A_hat)[1])
print(alpha * kappa + helper(A_hat)[1])

def odes(T,t,A_hat=A_hat, kappa = kappa, alpha = alpha):
    mode_vec, diagonal_A = helper(A_hat)
    A = alpha * kappa + diagonal_A
    dTdt = A@T + mode_vec
    return dTdt

T0 = R
x = np.linspace(0,t_0,10000)
Tf = odeint(odes,T0,x)
end2 = time.time()

print(f'Time taken for  simulations in the System {(2**12)*(end2 - end)}')

fig, axs = plt.subplots(m, 1)

T_hat = []

for i in range(m):
    for point in x:
        T_hat.append(Temperature_curves_normie(point, P = P_hat, m = m , length = n + 1, t_0 = t_0)[i])
    axs[i].plot(x,T_hat)
    axs[i].plot(x,Tf[:,i])
    axs[i].set_ylabel(f"Room {i+1}")
    axs[i].legend(['Bezier Solution','Optimal Mapping - Numerical Simulation of the system'])
    T_hat = []

fig.suptitle(r' $\alpha$ ' + f'= {alpha}, k1 = {k_1}, k2 = {k_2} , k3 = {k_3}, k4 = {k_4}, H = {H}, C = {C}')
plt.show()


print(Tf.shape)












# t0_values = np.linspace(2, 30, 25)
# objval_list = []
# A_hat_list = []
# for t_0 in t0_values:
#     hvac_model = HVAC_model()
#     hvac_model.set_time(t_0)
#     objval, A_hat,_ = hvac_model.optimise()
#     objval_list.append(objval)
#     A_hat_list.append(A_hat)
# plt.plot(t0_values,objval_list)
# plt.show()





















