import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

class Plotter:
    def __init__(self, offlineGE):
        self.GE = offlineGE
        return

    def plot_pacejka(self):
        fig, axis = plt.subplots(2, 1)
        xx = np.linspace(-0.2, 0.2, 100)
        # Front tire
        axis[0].scatter(self.GE.VE.SA_f, np.divide(self.GE.VE.F_fy, self.GE.VE.F_fz), s=0.2, label='mu_y front VE')
        params_front = [self.GE.tire.Bf * self.GE.T_slope_front_opt, self.GE.tire.Cf,
                        self.GE.tire.Df * self.GE.T_peak_front_opt, self.GE.tire.Ef, self.GE.tire.Hf, self.GE.tire.Vf]
        mu_fy = self.GE.pacejka(params_front, xx)
        axis[0].plot(xx, mu_fy, 'r-', label='Scaled Pacejka')
        axis[0].legend(loc='upper right')
        axis[0].set_ylabel('mu_y')
        axis[0].set_xlabel('SA [rad]')
        axis[0].grid()
        # Rear tire
        axis[1].scatter(self.GE.VE.SA_r, np.divide(self.GE.VE.F_ry, self.GE.VE.F_rz), s=0.2, label='mu_y rear VE')
        params_rear = [self.GE.tire.Br * self.GE.T_slope_rear_opt, self.GE.tire.Cr,
                       self.GE.tire.Dr * self.GE.T_peak_rear_opt, self.GE.tire.Er, self.GE.tire.Hr, self.GE.tire.Vr]
        mu_ry = self.GE.pacejka(params_rear, xx)
        axis[1].plot(xx, mu_ry, 'r-', label='Scaled Pacejka')
        axis[1].legend(loc='upper right')
        axis[1].set_ylabel('mu_y')
        axis[1].set_xlabel('SA [rad]')
        axis[1].grid()

        

    def plot_accelerations(self):
        fig, axis = plt.subplots(3, 1)

        # Plot ax: QP vs VE
        axis[0].plot(self.GE.VE.ax, label='ax VE')
        axis[0].plot(self.GE.ax_opt, label='ax GE')
        axis[0].set_ylabel('a_x [m/s^2]')
        axis[0].set_xlabel('t ')
        axis[0].legend(loc='upper right')
        axis[0].grid()
        # Plot ay: QP vs VE
        axis[1].plot(self.GE.VE.ay, label='ay VE')
        axis[1].plot(self.GE.ay_opt, label='ay GE')
        axis[1].set_ylabel('a_y [m/s^2]')
        axis[1].set_xlabel('t ')
        axis[1].legend(loc='upper right')
        axis[1].grid()

        # Plot ddyaw: QP vs VE
        axis[2].plot(self.GE.VE.ddyaw, label='ddyaw VE')
        axis[2].plot(self.GE.ddyaw_opt, label='ddyaw GE')
        axis[2].set_ylabel('ddyaw [rad/s^2]')
        axis[2].set_xlabel('t ')
        axis[2].legend(loc='upper right')
        axis[2].grid()

    def plot_Fy(self):
        fig, axis = plt.subplots(2, 1)
        xx = np.linspace(-0.2, 0.2, 100)
        # Front tire
        axis[0].scatter(self.GE.VE.SA_f, np.divide(self.GE.F_fy_opt, self.GE.VE.F_fz),s=0.2 , label='mu_y front GE')
        params_front = [self.GE.tire.Bf * self.GE.T_slope_front_opt, self.GE.tire.Cf,
                        self.GE.tire.Df * self.GE.T_peak_front_opt, self.GE.tire.Ef, self.GE.tire.Hf, self.GE.tire.Vf]
        mu_fy = self.GE.pacejka(params_front, xx)
        axis[0].plot(xx, mu_fy, 'r-', label='Scaled Pacejka')
        axis[0].legend(loc='upper right')
        axis[0].set_ylabel('mu_y')
        axis[0].set_xlabel('SA [rad]')
        axis[0].grid()
        # Rear tire
        axis[1].scatter(self.GE.VE.SA_r, np.divide(self.GE.F_ry_opt, self.GE.VE.F_rz),s=0.2, label='mu_y rear GE')
        params_rear = [self.GE.tire.Br * self.GE.T_slope_rear_opt, self.GE.tire.Cr,
                       self.GE.tire.Dr * self.GE.T_peak_rear_opt, self.GE.tire.Er, self.GE.tire.Hr, self.GE.tire.Vr]
        mu_ry = self.GE.pacejka(params_rear, xx)
        axis[1].plot(xx, mu_ry, 'r-', label='Scaled Pacejka')
        axis[1].legend(loc='upper right')
        axis[1].set_ylabel('mu_y')
        axis[1].set_xlabel('SA [rad]')
        axis[1].grid()

    def plot_gg(self):
        fig, axis = plt.subplots(2, 1)
        # Plot VE gg
        axis[0].plot(self.GE.VE.ay / 9.81, self.GE.VE.ax / 9.81, '.', label='gg VE')
        axis[0].set_ylabel('a_x / g')
        axis[0].set_xlabel('a_y / g')
        axis[0].legend(loc='upper right')
        axis[0].title.set_text('gg plot')
        axis[0].grid()
        # Plot GE gg
        axis[1].plot(self.GE.ay_opt / 9.81, self.GE.ax_opt / 9.81, '.', label='gg GE')
        axis[1].set_ylabel('a_x / g')
        axis[1].set_xlabel('a_y / g')
        axis[1].legend(loc='upper right')
        axis[1].grid()

    def evaluate_offline_GE(self):
        B = 0.095 * 180/3.14 / 0.7
        C = -2.707
        D = 1.86 * 0.7

        fig, axis = plt.subplots(2, 1)
        xx = np.linspace(-0.2, 0.2, 100)
        # Front tire
        
        axis[0].plot(self.GE.VE.ay, label='ay measurement')
        axis[0].plot(self.GE.ay_opt, label='ay from new tire params')
        axis[0].legend(loc='upper right')
        axis[0].set_ylabel('mu_y')
        axis[0].set_xlabel('SA [rad]')
        axis[0].grid()
        # Rear tire
        axis[1].plot(self.GE.VE.ay, label='ay GT')
        params_front = [B, C,
                       D, 0, 0, 0]
        F_fy = self.GE.pacejka(params_front, self.GE.VE.SA_f) * self.GE.VE.F_fz
        params_rear = [B, C,
                       D, 0, 0, 0]
        F_ry =  self.GE.pacejka(params_rear, self.GE.VE.SA_r) * self.GE.VE.F_rz
        axis[1].plot((F_fy + F_ry * np.cos(self.GE.VE.delta) + self.GE.VE.F_fx * np.sin(self.GE.VE.delta))/self.GE.VE.car.m, label='ay from old tire params')
        axis[1].legend(loc='upper right')
        axis[1].grid()

        e_ay_ge = mean_squared_error(self.GE.VE.ay, self.GE.ay_opt)
        e_ay_mpc = mean_squared_error(self.GE.VE.ay, (F_fy + F_ry*np.cos(self.GE.VE.delta) + 
                                                    self.GE.VE.F_fx * np.sin(self.GE.VE.delta))/self.GE.VE.car.m)
        print(e_ay_ge)
        print(e_ay_mpc)

    def show(self):
        # Plot ax: QP vs VE
        fig, axis = plt.subplots(1, 1)
        axis.plot(self.GE.VE.ax, label='ax VE')
        axis.plot((self.GE.VE.F_rx-self.GE.VE.F_fy*np.sin(self.GE.VE.delta)+self.GE.VE.F_fx*np.cos(self.GE.VE.delta) - self.GE.VE.Fx_drag - self.GE.VE.F_res) / self.GE.VE.car.m , label='ax GE')
        axis.legend(loc='upper right')
        axis.grid()

        plt.show()
