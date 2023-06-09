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

def diff(my_list , len, T):
    temp = []
    n = len-1
    for i in range(n):
        temp.append((n/T)*(my_list[i+1]-my_list[i]))
    return temp

#All the units are SI units

global a,A,rho
T = 50
a = 0.01
A = 1
H1 = 2
H2 = 1
tolerance = 0.0001
rho = 1000

n1 = 6
n2 = 6
n = 12

coefficients = np.array([[-a/A,0],[a/A,0]])
test_points = np.array([random.uniform(0, T) for _ in range(300)])

m = gp.Model('dumb')
objective_function = 0

control_points_h1 = m.addVars(n1+1, lb = 0-tolerance, ub = H1+ tolerance , vtype = GRB.CONTINUOUS, name = 'control_points_h1') # lb is an iterable
control_points_h2 = m.addVars(n2+1, lb = 0-tolerance, ub = H2+ tolerance , vtype = GRB.CONTINUOUS, name = 'control_points_h2') # lb is an iterable
control_points_f = m.addVars(n+1, lb = -gp.GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'control_points_f') # lb is an iterable

for t in test_points:  
    function_at_a_test_point = 0

    h1 = bezier_curve(control_points_h1, n1+1, T, t)
    h2 = bezier_curve(control_points_h2, n2+1, T, t)
    h1_diff = bezier_curve(diff(control_points_h1, n1+1, T), n1, T, t)
    h2_diff = bezier_curve(diff(control_points_h2, n2+1, T), n2, T, t)
    f = bezier_curve(control_points_f, n+1, T, t)

    rhs_vector = np.array([h1, h2])
    lhs_vector = np.array([h1_diff, h2_diff])
    rhs_constants = np.array([f/rho , 0])

    function_at_a_test_point = lhs_vector - coefficients @ rhs_vector - rhs_constants

    objective_function += function_at_a_test_point @ function_at_a_test_point

m.setObjective(objective_function, GRB.MINIMIZE)

m.addConstr( bezier_curve(control_points_h1, n1+1, T, 0) == 0)
m.addConstr( bezier_curve(control_points_h2, n2+1, T, 0) == 0)
m.addConstr( bezier_curve(control_points_h1, n1+1, T, T) == H1)
m.addConstr( bezier_curve(control_points_h2, n2+1, T, T) == H2)

m.setParam('OutputFlag', False)
m.optimize()
m.update()

final_list = []

if m.status == gp.GRB.OPTIMAL:
    final_list = []
    for var in m.getVars():
        var_name = var.VarName
        var_value = m.getAttr('X')
        # print(f'{var_name} {var_value}')
        final_list.append(var_value)
    final_list = final_list[0]
    control_points_h1_normie = np.array(final_list)[0:n1+1]
    control_points_h2_normie = np.array(final_list)[n1+1:n1+n2+2]
    control_points_f_normie = np.array(final_list)[n1+n2+2:]

    print(control_points_h1_normie)
    print(control_points_h2_normie)
    print(control_points_f_normie)

else:
    print("Model did not converge to an optimal solution.")

x = np.linspace(0, T, 10000)
h1_hat = []
h2_hat = []
f_hat= []

for g in x:
    h1_hat.append(bezier_curve_normie(control_points_h1_normie, n1+1, T, g))
    h2_hat.append(bezier_curve_normie(control_points_h2_normie, n2+1, T, g))
    f_hat.append(bezier_curve_normie(control_points_f_normie, n+1, T, g))

h1_hat = np.asarray(h1_hat)
h2_hat = np.asarray(h2_hat)
f_hat = np.asarray(f_hat)

# y_1 = 1/7 * np.exp(-x)+ 6/7 * np.exp(6*x)  
# y_2 = -1/7 * np.exp(-x)+ 8/7 * np.exp(6*x)  


#Validation 

def odes(x, t, control_points_f_normie = control_points_f_normie, len = n+1, T = T):
    # # assign each ODE to a vector element
    h1 = x[0]
    h2 = x[1]
    f = bezier_curve_normie(control_points_f_normie,len,T,t)
    # # define each ODE
    dh1dt =  (f/rho) - (a/A)*h1
    dh2dt = (a/A)*h1

    return [dh1dt, dh2dt]
    
x0 = [0, 0]

# print(odes(x0, control_points_f_normie, n+1, T ,t=0))
h = odeint(odes,x0,x)

h1 = h[:,0]
h2 = h[:,1]


# plot the results

fig, axs = plt.subplots(3, 1)

# Plotting in subplots
axs[0].plot(x, h1_hat, label='h1')
axs[0].plot(x, h1, label='h1')
axs[0].legend(['Approximation', 'True'])
axs[0].set_ylabel('h1')

axs[1].plot(x, h2_hat, label='h2')
axs[1].plot(x, h2, label='h2')
axs[1].legend(['Approximation', 'True'])
axs[1].set_ylabel('h2')

axs[2].plot(x, f_hat, label='f')
axs[2].set_ylabel('Control')

# Set shared x-axis label
axs[-1].set_xlabel('x')

# Display the plot
plt.show()


