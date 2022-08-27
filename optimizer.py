import torch
import numpy as np
import cvxpy as cp
from data_processing import DataProcessing
from car import Car, Tire


class Optimizer:
    def __init__(self, logfile):
        self.logfile = logfile
        self.car = Car()                                    # Load car parameters
        self.tire = Tire()                                  # Load prior tire parameters
        self.VE = DataProcessing(logfile)                   # Load data from logfile in .csv format
        self.n = self.VE.n                                  # Window size (number of measurements)
        self.problemIsSetup = False                         # osqp problem state

    def pacejka(self, params, SA):
        '''
        Pacejka function:
            Input:  - Tire parameters (B, C, D, E, H, V)
                    - Slip angles
            Output: - Grip coefficient
        '''
        B = params[0]
        C = params[1]
        D = params[2]
        E = params[3]
        H = params[4]
        V = params[5]

        mu_y = D * np.sin(C * np.arctan(B * (SA + H) - E * (B * (SA + H) + np.arctan(B * (SA + H))))) + V
        return np.array(mu_y)

    def get_constraint_vectors(self):
        '''
        Returns the constraint vectors b for the bicycle model.
        '''

        b_ax = np.array(self.VE.F_rx + self.VE.F_fx*np.cos(self.VE.delta) - self.VE.Fx_drag - self.VE. F_res)
        b_ay = np.array(self.VE.F_fx*np.sin(self.VE.delta))
        #b_ddyaw = np.array(self.VE.F_fx*self.car.lf*np.sin(self.VE.delta))

        b_ddyaw = np.array((self.VE.Fx_RR-self.VE.Fx_RL) * self.car.tw/2 +
                            self.VE.Fx_FR * (self.car.tw/2 *np.cos(self.VE.delta) + self.car.lf*np.sin(self.VE.delta)) +
                            self.VE.Fx_FL * (self.car.lf*np.sin(self.VE.delta)-np.cos(self.VE.delta)*self.car.tw /2))


        return b_ax, b_ay, b_ddyaw

    def solve_optimization(self, T_slope_front, T_slope_rear, gamma=2, tau=0.00005):
        '''
        Solves the convex optimization problem given slope scaling parameters for front and rear.
        :param gamma: Huber penatly function gamma
        :param tau: Tradeoff weight for the  divergence between tire forces and Pacejka
        :return opt_accel: Optimal (ax, ay, ddyaw) for given slope scaling that minimize cost
        :return opt_Fy: Optimal (F_fy, F_ry) for given slope scaling that minimize cost
        :return opt_peak: Optimal (T_peak_front, T_peak_rear) for given slope scaling that minimize cost
        :return opt_cost: Optimal cost for given slope scaling
        '''

        # Change tire parameters as function of slope scaling
        params_front = [self.tire.Bf * T_slope_front, self.tire.Cf, self.tire.Df, self.tire.Ef, self.tire.Hf, self.tire.Vf]
        params_rear = [self.tire.Br * T_slope_rear, self.tire.Cr, self.tire.Dr, self.tire.Er, self.tire.Hr, self.tire.Vr]
        # Get constraints vector representing bicycle model
        b_ax, b_ay, b_ddyaw = self.get_constraint_vectors()

        # Decision variables
        ax = cp.Variable((self.n, 1))
        ay = cp.Variable((self.n, 1))
        ddyaw = cp.Variable((self.n, 1))
        F_fy = cp.Variable((self.n, 1))
        F_ry = cp.Variable((self.n, 1))
        T_peak_front = cp.Variable((1, 1))
        T_peak_rear = cp.Variable((1, 1))

        # Huber penalty for accelerations
        cost = cp.sum(cp.huber(ax - self.VE.ax, gamma))
        cost += cp.sum(cp.huber(ay - self.VE.ay, gamma))
        cost += cp.sum(cp.huber(ddyaw - self.VE.ddyaw, gamma))
        # Norm_2 loss for tire forces and pacejka model
        #cost += tau * cp.sum_squares(F_fy - cp.multiply(T_peak_front, self.VE.F_fz * self.pacejka(params_front, self.VE.SA_f)))
        #cost += tau * cp.sum_squares(F_ry - cp.multiply(T_peak_rear, self.VE.F_rz * self.pacejka(params_rear, self.VE.SA_r)))
        cost = cp.Minimize(cost)

        # Constraints (bicycle model)
        constraints = [self.car.m * ax + cp.multiply(F_fy, np.sin(self.VE.delta)) == b_ax]
        constraints += [self.car.m * ay - cp.multiply(np.cos(self.VE.delta), F_fy) - F_ry == b_ay]
        constraints += [self.car.Iz * ddyaw - cp.multiply(self.car.lf*np.cos(self.VE.delta), F_fy) + cp.multiply(self.car.lr, F_ry) == b_ddyaw]
        constraints += [F_fy == cp.multiply(T_peak_front, self.VE.F_fz * self.pacejka(params_front, self.VE.SA_f))]
        constraints += [F_ry == cp.multiply(T_peak_rear, self.VE.F_rz * self.pacejka(params_rear, self.VE.SA_r))]

        prob = cp.Problem(cost, constraints)
        prob.solve(solver=cp.OSQP)
        # Results
        opt_accel = np.array((ax.value, ay.value, ddyaw.value))
        opt_Fy = np.array((F_fy.value, F_ry.value))
        opt_peak = np.array([T_peak_front.value, T_peak_rear.value]).reshape((2, ))
        opt_cost = cost.value
        return opt_accel, opt_Fy, opt_peak, opt_cost

    def run_sim_annealing(self, n_QP=1):
        '''
        Solves the optimization problem as defined in the overleaf.
        This is done by solving a grid of convex programs and we eventually look for the slope scaling parameters which
        allows to obtain the convex program with minimum cost.
        :param n_QP: Number of slope scaling parameters to try. Total number of convex programs is n_QP x n_QP (front&rear)
        '''
        T_slope_front0 = 0.1
        T_slope_rear0 = 0.1
        # Solve optimization for given slope scaling
        accel0, Fy0, T_peak0, cost0 = self.solve_optimization(T_slope_front0, T_slope_rear0)

        tau = 0.1
        T = 25.0
        dT = 0.1
        n = int(T / dT) - 1
        # Init grid of slope parameters
        self.history_T_slope_front = np.zeros(n)
        self.history_T_slope_rear = np.zeros(n)
        self.history_cost = np.zeros(n)
        
        for i in range(n):
            T_slope_front1 = np.random.normal(T_slope_front0, tau)
            T_slope_rear1 = np.random.normal(T_slope_rear0, tau)

            # Solve optimization for given slope scaling
            accel, Fy, T_peak, cost = self.solve_optimization(T_slope_front1, T_slope_rear1)
            
            if cost < cost0:
                T_slope_front0 = T_slope_front1
                T_slope_rear0 = T_slope_rear1
                cost0 = cost
            else:
                alpha = min([1, np.exp(-abs(cost - cost0)/T)])
                if alpha > np.random.uniform():
                    T_slope_front0 = T_slope_front1
                    T_slope_rear0 = T_slope_rear1

            self.history_cost[i] = cost0
            self.history_T_slope_front[i] = T_slope_front0
            self.history_T_slope_rear[i] = T_slope_rear0

        accel, Fy, T_peak, cost0 = self.solve_optimization(T_slope_front0, T_slope_rear0)

        # Optimal cost
        self.cost_opt = cost0
        # Optimal accelerations
        self.ax_opt = accel[0, :]        # Optimal ax
        self.ay_opt = accel[1, :]  # Optimal ax
        self.ddyaw_opt = accel[2, :]  # Optimal ax
        # Optimal lateral tire forces
        self.F_fy_opt = Fy[0, :]
        self.F_ry_opt = Fy[1, :]
        # Optimal peak scaling
        self.T_peak_front_opt =T_peak[0]
        self.T_peak_rear_opt = T_peak[1]
        # Optimal slope scaling
        self.T_slope_front_opt = T_slope_front0
        self.T_slope_rear_opt = T_slope_rear0

