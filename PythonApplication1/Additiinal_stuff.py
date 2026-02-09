import numpy as np
from scipy.integrate import solve_ivp
import PySide6
from   PySide6 import QtUiTools,QtWidgets
from   PySide6.QtGui import *
from   PySide6.QtWidgets import QApplication,QMainWindow,QVBoxLayout,QWidget,QLabel
from   PySide6 import QtCore
from   PySide6.QtWidgets import QFileDialog
import pyqtgraph as pg
import time
import scipy.optimize as opt
from bayes_opt import BayesianOptimization
from bayes_opt  import acquisition

class LorenzAttr():

    def __init__(self,width=1000,height=750):
        # Create an image of the Lorenz attractor.
        # The maths behind this code is described in the scipython blog article
        # at https://scipython.com/blog/the-lorenz-attractor/
        # Christian Hill, January 2016.
        # Updated, January 2021 to use scipy.integrate.solve_ivp        
        self.MIN_RANGE = width
        self.MAX_RANGE = height        
        self.a = 1
        self.b = 1
        # Lorenz paramters and initial conditions.
        self.sigma =10.
        self.beta = 2.667
        self.rho =  28.
        self.u0 = 0.
        self.v0 = 1.
        self.w0 = 1.05
        self.t=0
                        
    def lorenz(self, t, X, sigma, beta, rho):
            """The Lorenz equations."""
            u, v, w = X
            up = -sigma*(u - v)
            vp = rho*u - v - u*w
            wp = -beta*w + u*v
            return up, vp, wp

    def integrate(self,tmax=100,n=10000):
            # Maximum time point and total number of time points.
            tmax, n = tmax, n #100, 10000        
            soln = solve_ivp(self.lorenz, (self.t, self.t+tmax), (self.u0, self.v0, self.w0), args=(self.sigma, self.beta, self.rho), dense_output=True)
            t=np.linspace(self.t,self.t+tmax)
            x, y, z = soln.sol(t)
            self.u0,self.v0,self.w0 = x[-1],y[-1],z[-1]
            self.t = self.t+tmax
            return x,y,z

    def integrate_ext(self,tmax=100,n=10000):
        x,y,z=self.integrate(tmax=tmax,n=n)
        x_ext=x[-1]*self.a[0]+self.b[0] #np.interp(x, (x.min(), x.max()), (self.MIN_RANGE, self.MAX_RANGE))
        y_ext=y[-1]*self.a[1]+self.b[1] #np.interp(y, (y.min(), y.max()), (self.MIN_RANGE, self.MAX_RANGE))
        z_ext=z[-1]*self.a[2]+self.b[2]
        return x_ext,y_ext,z_ext

    def SetRange(self,min_r=20,max_r=30):

        self.MIN_RANGE = min_r
        self.MAX_RANGE = max_r
        real_min,real_max = self.simulate_long_run()

        self.a_tmp=(self.MAX_RANGE-self.MIN_RANGE)/(real_max[0]-real_min[0])
        self.b_tmp=(self.MAX_RANGE-self.MIN_RANGE)/(real_max[1]-real_min[1])
        self.c_tmp=(self.MAX_RANGE-self.MIN_RANGE)/(real_max[2]-real_min[2])
        self.a=(self.a_tmp,self.b_tmp,self.c_tmp)

        self.b_tmp1=self.MIN_RANGE-self.a[0]*real_min[0]
        self.b_tmp2=self.MIN_RANGE-self.a[1]*real_min[1]
        self.b_tmp3=self.MIN_RANGE-self.a[2]*real_min[2]

        self.b=(self.b_tmp1,self.b_tmp2,self.b_tmp3)#self.MIN_RANGE-self.a*real_min
        

    def simulate_long_run(self):
        ch=LorenzAttr()
        ch.sigma=self.sigma 
        ch.beta=self.beta
        ch.rho =self.rho 
        ch.u0=self.u0 
        ch.v0=self.v0 
        ch.w0=self.w0 
        ch.t=self.t

        vals_x=[]
        vals_y=[]
        vals_z=[]
        for l in range(0,10):
            x,y,z=ch.integrate(tmax=100,n=10000)
            for p in range(0,len(x)):
                vals_x.append(x[p])
                vals_y.append(y[p])
                vals_z.append(z[p])
        ch=None
        return (np.min(np.asarray(vals_x)),np.min(np.asarray(vals_y)),np.min(np.asarray(vals_z))),(np.max(np.asarray(vals_x)),np.max(np.asarray(vals_y)),np.max(np.asarray(vals_z)))



class RTPlotWidget_1(PySide6.QtWidgets.QWidget):

    def __init__(self):

        super().__init__()        

        self.setAttribute(PySide6.QtCore.Qt.WA_DeleteOnClose)
        #the numbers for pressure
        self.label =  PySide6.QtWidgets.QLabel("Pressure values...")
        self.label.setMinimumWidth(130)    
        self.label.setFont(PySide6.QtGui.QFont("Arial", 16))
        self.graphWidget=pg.PlotWidget()
        #self.setCentralWidget(self.graphWidget)
        self.graphWidget.setBackground('w')    
        #data and graph
        self.x1 =[]
        self.pen1=pg.mkPen(color=(255,0,0))
        self.line1=self.graphWidget.plot(self.x1,pen=self.pen1)
                       
        layout=PySide6.QtWidgets.QGridLayout()     
        layout.addWidget(self.label, 1, 0)     
        layout.addWidget(self.graphWidget, 2, 0)                   
        self.setLayout(layout)        
        self.show()
                    
    def flatten(self,xss):
        return [x for xs in xss for x in xs]  
        
    def plot(self,x):
        #CRAZY, but qt does not like cycles, so we do by hands
                
        if(len(x)>0):
            self.x1=x
            self.line1.setData(self.x1)        
        time.sleep(0.1)

    def hex_to_rgb(self,value):
        h = value.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
        

    def updateText(self,text):
        self.label.setText("Measurement "+str(text))


class Optim_Control_1():
    
    def __init__(self, U_initial=[0],
                       U_num=2,
                       U_min=[-4.1],
                       U_max=[4.1],                       
                       max_iter=300,                       
                       f_cost=None,
                       Use_nonl_bounds=True,
                ):

        shp_initial=np.shape(U_initial)        
        self.U0 = np.zeros(shp_initial)
        for i in range(0,len(U_initial)):
            if(U_initial[i]>U_max[i] or U_initial[i]<U_min[i]):
                self.U0[i]=U_min[i]+(U_max[i]-U_min[i])/2
            else:
                self.U0[i]=U_initial[i]
                
        self.Bounds_u = None
        self.f_cost = f_cost
        self.max_iter = max_iter

        self.Solution=None
        self.isRunning=False
        self.Terminate_solver=False
        self.Optimized=False
        
        self.bounds_list=[]
        for i in range(0,U_num):
            #bounds = opt.Bounds(lb=U_min[i], ub=U_max[i])
            self.bounds_list.append([U_min[i], U_max[i]])
        self.Bounds_u=opt.Bounds(lb=np.asarray(self.bounds_list)[:,0], ub=np.asarray(self.bounds_list)[:,1])

        if(len(np.shape(U_initial))==2):
            for k in range(0,np.shape(U_initial)[0]):
                for s in range(0,np.shape(U_initial)[1]):
                    self.U0[k][s] = U_initial[k][s]
        if(len(np.shape(U_initial))==1):
            for k in range(0,np.shape(U_initial)[0]):
                self.U0[k] = U_initial[k]

    def Launch(self):

        self.Optimized=False
        self.isRunning = True
        self.Terminate_solver = False
        self.Solution = opt.minimize(self.f_cost, self.U0,                                 
                                                  method='nelder-mead',#'trust-constr','SLSQP', 'BFGS' cant handle bounds,'nelder-mead'
                                                  bounds=self.Bounds_u,                                 
                                                  options={'maxiter': self.max_iter, 
                                                           'disp': True}
                                        )
        self.isRunning = False
        self.Optimized=True