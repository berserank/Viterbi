import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

def bezier_curve(control_points, n_t, n_u, t, u):
    # curve_point = np.zeros(2, dtype=float)  # Initialize the curve point as [0, 0]
    curve_point = gp.quicksum(control_points[i][j] * (t ** i) * ((1 - t) ** (n_t - i)) * (u ** j) * ((1 - u) ** (n_u - j)) * (np.math.comb(n_t, i)) * (np.math.comb(n_u, j)) for j in range(n_u + 1) for i in range(n_t + 1))         
    return curve_point

n_t = 3
n_u = 3

def bezier_curve_normie(control_points, n_t, n_u, t, u):
    curve_point = 0.0  # Initialize the curve point as 0.0

    for j in range(n_u + 1):
        for i in range(n_t + 1):
            term = (
                control_points[i][j]
                * (t ** i)
                * ((1 - t) ** (n_t - i))
                * (u ** j)
                * ((1 - u) ** (n_u - j))
                * (np.math.comb(n_t, i))
                * (np.math.comb(n_u, j))
            )
            curve_point += term

    return curve_point

test_points = np.random.rand(17,2)

def diff(my_list, n_t, n_u, variable):
    if variable == 't':
        temp = (n_t - 1) * (my_list[1:, :] - my_list[:-1, :])
        return temp

    if variable == 'u':
        temp = (n_u - 1) * (my_list[:, 1:] - my_list[:, :-1])
        return temp

control_points = np.array([[1,1,1,1],[2,2,2,2],[4,5,6,7],[8,9,10,11]])

m = gp.Model('2D-Bezier')

objective_function = 0

control_points = m.addMVar(shape = (n_t+1,n_u+1), lb = -gp.GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'control_points')
control_points_t = diff(control_points, n_t+1,n_u+1,'t')
control_points_u =  diff(control_points, n_t+1,n_u+1,'u')


for t in test_points:
    function_at_a_test_point = 0
    function_at_a_test_point = bezier_curve(control_points_t, n_t-1, n_u, t[0], t[1]) +  bezier_curve(control_points_u , n_t, n_u-1 , t[0],t[1])  -  2
    objective_function += function_at_a_test_point * function_at_a_test_point

m.setObjective(objective_function, GRB.MINIMIZE)


for t in test_points:
    x = bezier_curve(control_points,n_t, n_u,t[0],0) - t[0]*t[0]
    m.addConstr( x == 0)

m.setParam('OutputFlag', False)
m.optimize()
m.update()

final_list = []

for v in m.getVars():
    print('%s %g' % (v.VarName, v.X))
    final_list.append(v.X)

final_list = np.array(final_list).reshape(n_t+1,n_u+1)
print(final_list)

# Generate points along the curve
num_points = 100  # Number of points to generate
t_values = np.linspace(0, 1, num_points)
u_values = np.linspace(0, 1, num_points)

curve_points = np.zeros((num_points, num_points))  # Initialize curve_points with the correct shape
true_points = np.zeros((num_points, num_points))
for i, t in enumerate(t_values):
    for j, u in enumerate(u_values):
        curve_points[i, j] = bezier_curve_normie(final_list,n_t, n_u, t, u)
        true_points[i,j] = 2*u + (t-u)**2
X_grid, Y_grid = np.meshgrid(t_values, u_values)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(X_grid, Y_grid, curve_points, cmap='viridis', label = 'Approximation')
surf2 = ax.plot_surface(X_grid, Y_grid, true_points, cmap='RdBu', label = 'True Function')
fig.colorbar(surf1)
fig.colorbar(surf2)
surf1_proxy = plt.Rectangle((0, 0), 1, 1, fc=surf1.get_facecolor()[0])
surf2_proxy = plt.Rectangle((0, 0), 1, 1, fc=surf2.get_facecolor()[0])

# Add a legend
ax.legend([surf1_proxy, surf2_proxy], ['Approximation', 'True'])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('3D Surface Plot')
# ax.legend([surf1[0], surf2[0]], ['Approximation', 'True Function'])
plt.show()

# Flatten the curve points for plotting

# Plot the curve
# plt.plot(X, Y)
# plt.scatter(X, Y, color='red', marker='o')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Two-dimensional Bézier Curve')
# plt.grid(True)
# plt.show()
