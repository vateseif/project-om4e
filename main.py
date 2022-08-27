import time

from plotter import Plotter
from optimizer import Optimizer


logfile = 'Log_20200823_162812_xintong2.csv'

GE = Optimizer(logfile)  # Init GE with logdata
t = time.time()
GE.run_sim_annealing()  # Run grid of n_QP x n_QP convex programs to find best scaling params
print('Elapsed time =', time.time() - t)
print('Optimal cost =', GE.cost_opt)
print('Optimal T_peak_front =', GE.T_peak_front_opt)
print('Optimal T_peak_rear =', GE.T_peak_rear_opt)
print('Optimal T_slope_front =', GE.T_slope_front_opt)
print('Optimal T_slope_rear =', GE.T_slope_rear_opt)


'''Plot results'''
plotter = Plotter(GE)
plotter.plot_accelerations()
plotter.plot_pacejka()
plotter.show()