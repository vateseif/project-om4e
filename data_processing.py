from curses.ascii import LF
import numpy as np
from scipy import signal
import csv
from car import Car

class DataProcessing:
    def __init__(self, logfile):

        self.logfile = logfile
        self.car = Car()
        self.g = 9.81
        self.ts = 0.005
        self.load_data()
        self.select()

    

    def load_data(self, ind=None):
        if not self.logfile.endswith('.csv'):
            raise ValueError('logfile is not in .csv format')

        with open(self.logfile) as f:
            reader = csv.reader(f)
            columns = next(reader)
        self.logdata = np.array(np.loadtxt(self.logfile, delimiter=",", skiprows=1))

        # Data size
        self.n = self.logdata.shape[0]  # window size = number of measurements @200Hz
        m = self.logdata.shape[1]        # number of signals received
        
        # Load data 
        self.vx, self.vy, self.dyaw, self.ax, self.ay, self.ddyaw, self.T_FL, self.T_FR, \
            self.T_RL, self.T_RR, self.omega_FL_rpm, self.omega_FR_rpm, self.omega_RL_rpm, self.omega_RR_rpm, \
            self.delta, self.t_ve, self.t_motors, self.t_del = np.split(self.logdata, m, axis=1)

        self.SA_f = np.arctan2((self.vy + self.car.lf*self.dyaw), self.vx) - self.delta
        self.SA_r = np.arctan2((self.vy - self.car.lr*self.dyaw), self.vx)

        # Data conversion
        self.convert()

        # Prefilter data
        self.butter_prefilter()

        # Computed numerical derovtives
        self.compute_derivatives()
        

        '''Force estimation'''
        self.Fx_drag = 0.5 * self.car.rho * self.car.Cd * self.car.A * np.square(self.vx)
        self.Fz_lift = 0.5 * self.car.rho * self.car.Cl * self.car.A * np.square(self.vx)
        self.F_res = (self.car.m + self.Fz_lift) * self.car.R_res


        self.Fx_FL = (self.T_FL * self.car.gr_w - self.car.Iw * self.domega_FL ) / self.car.r_tire 
        self.Fx_FR = (self.T_FR * self.car.gr_w - self.car.Iw * self.domega_FR ) / self.car.r_tire 
        self.Fx_RL = (self.T_RL * self.car.gr_w - self.car.Iw * self.domega_RL ) / self.car.r_tire 
        self.Fx_RR = (self.T_RR * self.car.gr_w - self.car.Iw * self.domega_RR ) / self.car.r_tire 

        self.F_fx = self.Fx_FL + self.Fx_FR
        self.F_rx = self.Fx_RL + self.Fx_RR


        self.Fz_FL = 0.5 * ((self.car.m*self.g + self.Fz_lift)*self.car.lr/self.car.l_wb - 
                    self.car.m*self.ax*self.car.h/self.car.l_wb - self.Fx_drag*self.car.h/self.car.l_wb - self.ay*self.car.h/self.car.tw)
        self.Fz_FR = 0.5 * ((self.car.m*self.g + self.Fz_lift)*self.car.lr/self.car.l_wb - 
                    self.car.m*self.ax*self.car.h/self.car.l_wb - self.Fx_drag*self.car.h/self.car.l_wb + self.ay*self.car.h/self.car.tw)
        self.Fz_RL = 0.5 * ((self.car.m*self.g + self.Fz_lift)*self.car.lf/self.car.l_wb + 
                    self.car.m*self.ax*self.car.h/self.car.l_wb + self.Fx_drag*self.car.h/self.car.l_wb - self.ay*self.car.h/self.car.tw)
        self.Fz_RR = 0.5 * ((self.car.m*self.g + self.Fz_lift)*self.car.lf/self.car.l_wb + 
                    self.car.m*self.ax*self.car.h/self.car.l_wb + self.Fx_drag*self.car.h/self.car.l_wb + self.ay*self.car.h/self.car.tw)
        

        self.F_fz = self.Fz_FL + self.Fz_FR
        self.F_rz = self.Fz_RL + self.Fz_RR

        '''
        self.F_fy = ( self.car.Iz*self.ddyaw - self.F_fx*self.car.lf*np.sin(self.delta) + self.car.lr*self.car.m*self.ay - self.car.lr*self.F_fx*np.sin(self.delta) ) \
                    / ( self.car.l_wb*np.cos(self.delta))
        self.F_ry = self.car.m*self.ay - self.F_fx*np.sin(self.delta) - self.F_fy*np.cos(self.delta)
        

        '''
        self.F_fy = (self.car.Iz*self.ddyaw - (self.Fx_RR-self.Fx_RL)*self.car.tw/2 - self.Fx_FR*(self.car.tw/2*np.cos(self.delta)+self.car.lf*np.sin(self.delta))
                    - self.Fx_FL*(-self.car.tw/2*np.cos(self.delta)+self.car.lf*np.sin(self.delta)) + self.car.lr*self.car.m*(self.ay) -
                     self.car.lr*(self.Fx_FL+self.Fx_FR)*np.sin(self.delta))/(self.car.l_wb*np.cos(self.delta));
        
        self.F_ry = (self.car.m*self.ay*(1-self.car.lr/self.car.l_wb) + (self.Fx_RR-self.Fx_RL)*self.car.tw/(2*self.car.l_wb)
                    - self.Fx_FL*(self.car.tw*np.cos(self.delta))/(2*self.car.l_wb) + self.Fx_FR*(self.car.tw*np.cos(self.delta))/(2*self.car.l_wb)
                    - self.car.Iz * self.ddyaw / self.car.l_wb)
        
        
        

    def convert(self):
        self.omega_FL = self.omega_FL_rpm * 2*3.14/60/self.car.gr_w
        self.omega_FR = self.omega_FR_rpm * 2*3.14/60/self.car.gr_w
        self.omega_RL = self.omega_RL_rpm * 2*3.14/60/self.car.gr_w
        self.omega_RR = self.omega_RR_rpm * 2*3.14/60/self.car.gr_w

    def butter_prefilter(self):
        # Butterworth filter params
        fc = 25
        fs = 200
        b, a = signal.butter(3, fc/(fs/2), analog=False)

        
        self.ax = signal.filtfilt(b, a, np.squeeze(self.ax, axis=1))
        self.ay = signal.filtfilt(b, a, np.squeeze(self.ay, axis=1))
        self.dyaw = signal.filtfilt(b, a, np.squeeze(self.dyaw, axis=1))

        self.ax = np.expand_dims(self.ax, axis=1)
        self.ay = np.expand_dims(self.ay, axis=1)
        self.dyaw = np.expand_dims(self.dyaw, axis=1)
        
        
        self.omega_FL = signal.filtfilt(b, a, np.squeeze(self.omega_FL, axis=1))
        self.omega_FR = signal.filtfilt(b, a, np.squeeze(self.omega_FR, axis=1))
        self.omega_RL = signal.filtfilt(b, a, np.squeeze(self.omega_RL, axis=1))
        self.omega_RR = signal.filtfilt(b, a, np.squeeze(self.omega_RR, axis=1))

        self.omega_FL = np.expand_dims(self.omega_FL, axis=1)
        self.omega_FR = np.expand_dims(self.omega_FR, axis=1)
        self.omega_RL = np.expand_dims(self.omega_RL, axis=1)
        self.omega_RR = np.expand_dims(self.omega_RR, axis=1)

    def compute_derivatives(self):

        self.domega_FL = (np.diff(np.squeeze(self.omega_FL, axis=1)) /  self.ts).reshape((self.n-1,1))
        self.domega_FL = np.append(self.domega_FL, [[0]], axis=0)
        
        self.domega_FR = (np.diff(np.squeeze(self.omega_FR, axis=1)) /  self.ts).reshape((self.n-1,1))
        self.domega_FR = np.append(self.domega_FR, [[0]], axis=0)
        
        self.domega_RL = (np.diff(np.squeeze(self.omega_RL, axis=1)) /  self.ts).reshape((self.n-1,1))
        self.domega_RL = np.append(self.domega_RL, [[0]], axis=0)
        
        self.domega_RR = (np.diff(np.squeeze(self.omega_RR, axis=1)) /  self.ts).reshape((self.n-1,1))
        self.domega_RR = np.append(self.domega_RR, [[0]], axis=0)

        self.ddyaw = (np.diff(np.squeeze(self.dyaw, axis=1)) /  self.ts).reshape((self.n-1,1))
        self.ddyaw = np.append(self.ddyaw, [[0]], axis=0)

    def select(self):
        idx = (np.where((np.abs(self.ax) < 10) & (self.vx > 4) & (np.abs(self.delta) > 0*3.14/180))[0], )

        self.vx = self.vx[idx]
        self.vy = self.vy[idx]
        self.dyaw = self.dyaw[idx]
        self.ax = self.ax[idx]
        self.ay = self.ay[idx]
        self.ddyaw = self.ddyaw[idx]

        self.SA_f = self.SA_f[idx]
        self.SA_r = self.SA_r[idx]

        self.F_fy = self.F_fy[idx]
        self.F_ry = self.F_ry[idx]

        self.Fz_FL = self.Fz_FL[idx]
        self.Fz_FR = self.Fz_FR[idx]
        self.Fz_RL = self.Fz_RL[idx]
        self.Fz_RR = self.Fz_RR[idx]

        self.F_fz = self.F_fz[idx]
        self.F_rz = self.F_rz[idx]

        self.Fx_drag = self.Fx_drag[idx]
        self.Fz_lift = self.Fz_lift[idx]
        self.F_res = self.F_res[idx]

        self.Fx_FL = self.Fx_FL[idx]
        self.Fx_FR = self.Fx_FR[idx]
        self.Fx_RL = self.Fx_RL[idx]
        self.Fx_RR = self.Fx_RR[idx]

        self.F_fx = self.F_fx[idx]
        self.F_rx = self.F_rx[idx]

        self.delta = self.delta[idx]

        self.t = self.t_ve[idx]

        self.n = len(self.vx)


