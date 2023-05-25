import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(control_points, t):
    n = len(control_points) - 1
    # result = np.zeros_like(a = control_points[0], dtype=float)
    result = gp.quicksum(np.math.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i) * control_points[i]  for i in range(n + 1)) 
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
    for i in range(len(my_list)-1):
        temp.append((len(my_list)-1)*(my_list[i+1]-my_list[i]))
    return temp


n = 5
coefficients = [13,-5,1]

test_points = np.random.rand(20)

m = gp.Model('dumb')

k = len(coefficients)
count = k-1
objective_function = 0

control_points = m.addVars(n+1, vtype = GRB.CONTINUOUS, name = 'control_points') # lb is an iterable

for var in control_points.values():
    var.setAttr(gp.GRB.Attr.LB, -gp.GRB.INFINITY)

for t in test_points:  
    function_at_a_test_point = 0
    my_list = control_points
    k = 2
    for i in coefficients:
        function_at_a_test_point += i*bezier_curve(my_list, t)
        function_at_a_test_point = function_at_a_test_point/(t**k)
        k = k-1
        my_list = diff(my_list)
    # function_at_a_test_point -= (np.sin(30*t)-3)
    objective_function += function_at_a_test_point**2


m.setObjective(objective_function, GRB.MINIMIZE)
m.addConstr(bezier_curve(control_points,1) == 1)
m.addConstr(bezier_curve(diff(control_points),1) == 0)

m.setParam('OutputFlag', False)
m.optimize()

final_list = []
for v in m.getVars():
    print('%s %g' % (v.VarName, v.X))
    final_list.append(v.X)

x = np.linspace(0, 1, 10000)
x = x[1:]

y = 0.5*(x**3)*(2*(np.cos(2*np.log(x))) - 3* np.sin(2*np.log(x)))
y_hat = [bezier_curve_normie(final_list, i) for i in x]
# y_hat = [bezier_curve_normie(np.array([1.0,1.0,1.0,1.0]), i) for i in x]
# plt.plot([bezier_curve_normie(final_list, i) for i in sample_points])
# plt.plot([(1145/452)*np.exp(-2*h) + (1/452)*np.sin(30*h) - (15/452)*np.cos(30*h) - (678/452) for h in sample_points])

plt.plot(x,y_hat)
plt.plot(x,y)
plt.legend(['Approximation','True'])
plt.title('Complicated Homogenous ODE, Test points = 20 , Control Points = 6')
# plt.xlim((0,1))
plt.show()