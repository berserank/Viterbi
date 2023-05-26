import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(my_list, t):
    if (type(my_list)!= list ):
        n = (my_list.size)//2-1
    else:
        n = len(my_list)-1
    result = np.zeros_like(a = my_list[0], dtype=float)
    result = gp.quicksum(np.math.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i) * my_list[i] for i in range(n+1))
    return result


def bezier_curve_normie(control_points, t):
    n = len(control_points) - 1
    result = np.zeros_like(control_points[0])

    for i in range(n + 1):
        binomial_coefficient = np.math.comb(n, i)
        polynomial_term =((1 - t) ** (n - i)) * (t ** i)
        result += binomial_coefficient * polynomial_term * control_points[i]

    return result

def diff(my_list):
    temp = []
    n = my_list.size//2 - 1  if type(my_list) != list else len(my_list)-1
    for i in range(n):
        temp.append(n*(my_list[i+1]-my_list[i]))
    return temp


coefficients = np.array([[2,3],[4,3]])
test_points = np.random.rand(20)

m = gp.Model('dumb')
objective_function = 0
n = 10
control_points = m.addMVar(shape = (n+1,2), lb = -gp.GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'control_points') # lb is an iterable

for t in test_points:  
    function_at_a_test_point = 0
    my_list = control_points

    function_at_a_test_point = bezier_curve(diff(my_list),t) - coefficients @ bezier_curve(my_list, t)

    objective_function += function_at_a_test_point @ function_at_a_test_point

m.setObjective(objective_function, GRB.MINIMIZE)

m.addConstr( bezier_curve(control_points,0)[0] == 1 )
m.addConstr( bezier_curve(control_points,0)[1] == 1 )


m.setParam('OutputFlag', False)
m.optimize()
m.update()

final_list = []
for v in m.getVars():
    print('%s %g' % (v.VarName, v.X))
    final_list.append(v.X)

x = np.linspace(0, 1, 10000)

final_list_1 = final_list[::2]
final_list_2 = final_list[1::2]

print(bezier_curve_normie(final_list_1,0.2))


y_hat_1 = []
y_hat_2 = []

for g in x:
    y_hat_1.append(bezier_curve_normie(final_list_1, g))
    y_hat_2.append(bezier_curve_normie(final_list_2, g))

y_hat_1 = np.asarray(y_hat_1)
y_hat_2 = np.asarray(y_hat_2)
y_1 = 1/7 * np.exp(-x)+ 6/7 * np.exp(6*x)  
y_2 = -1/7 * np.exp(-x)+ 8/7 * np.exp(6*x)  

plt.plot(x,y_hat_1)
plt.plot(x,y_hat_2)
plt.plot(x,y_1)
plt.plot(x,y_2)
plt.legend(['Approximation of f1','Approximation of f2'])
plt.title('Systems of Differential Equations, Test points = 20 , Control Points = 11')
plt.show()