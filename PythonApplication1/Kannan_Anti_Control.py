import numpy as np
import sys
import threading
import random
import time
import winsound
from datetime import datetime
import pandas as pd
import os
import casadi as cs
from pathlib import Path
from numpy.fft import fft, ifft, rfft
from scipy.signal import savgol_filter
from scipy.signal import stft
import sklearn
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
import collections.abc
import scipy.optimize as opt
from bayes_opt import BayesianOptimization
from bayes_opt  import acquisition

import matplotlib
from   matplotlib.figure import Figure
import matplotlib.pyplot as plt
from   matplotlib.colors import Normalize, LogNorm, NoNorm
import pyqtgraph as pg

import PySide6
from   PySide6 import QtUiTools,QtWidgets
from   PySide6.QtGui import *
from   PySide6.QtWidgets import QApplication,QMainWindow,QVBoxLayout,QWidget,QLabel
from   PySide6 import QtCore
from   PySide6.QtWidgets import QFileDialog

#data prerpocessing from Kannan
import data_preprocessing as dpp
import xaiModel as xm
#additional processing from Sergey
import addit_math_routine as a_math
#QT GUI window
from ui_main_window import Ui_MainWindow
import Additiinal_stuff as add_s #this is for the Lorenz attractor emulation

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self):        
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.init_ui()
        self.InuitGlobals()
        
    def init_ui(self): 
        
        self.ExitButton.clicked.connect(self.Exit_program_click)        
        #REAL TIME TAB
        self.pushButton_start_real_time.clicked.connect(self.StartRealTime)
        self.pushButton_2_stop_real_time.clicked.connect(self.StopRealTime)
        self.UserConfirmationButton.clicked.connect(self.UserConfirmationClick)
        #Summator tab
        self.pushButton_ethalons_load.clicked.connect(self.LoadEthalonFilesClick)
        self.pushButton_ethalons_clear.clicked.connect(self.EraseEthalonsclick)
        self.pushButton_ethalons_setFixedValue.clicked.connect(self.Setfixedvalueasetalon)
        self.pushButton_ethalons_vs_other_data_2.clicked.connect(self.CompareWithEthalonsClick)
        self.pushButton_ethalons_vs_other_data_brows.clicked.connect(self.LoadDataToCompareWithEtalons)
        self.pushButton_real_time_folder_browse.clicked.connect(self.BrowsRealTimeFolderClick)
        #real time background color
        self.label_Real_Time.setStyleSheet("background-color:#d62728")#lightgreen")        
        self.statusbar.setStyleSheet("background-color:#d62728")
        self.statusbar.showMessage("")        
        
    def InuitGlobals(self):
                    
        self.ParamsValuesString=""
        #main processing thread
        self.ProcThread=None #this is for processing routing
        self.UserConfirmationEvent=None #this is user confirms that the values are transfered to the device
        self.StopRealTimeEvenet=None #this is to stop real time
        
        #parameters to tune & starting conditions
        self.ParamNames=[]
        self.RangeMin=[]
        self.RangeMax=[]
        self.StartingParamsVals=[]

        #this is the model for emulation (to substitute the real plant)
        self.VirtualPlant=a_math.Motor_Arm()
        #ethalon files/signals/Patterns
        self.Ethalon_values=None
        self.label_5_ethalons_label.setStyleSheet("background-color:#d62728")
        self.label_5.setText("No Etalons found")
        self.statusbar.setStyleSheet("background-color:#d62728")
        self.statusbar.showMessage("")    

        #where to take the data from - emulation or real data
        self.system_dynamics_user_choice=""        
            
    #TAB REAL TIME
    def StartRealTime(self):
        
        successFlag=self.UpdateGlobaProcSettings()
                
        if(successFlag!=True): return
        self.label_8_action_number_label.setText(str(0))
        self.label_Real_Time.setStyleSheet("background-color:#2ca02c")
        self.UserConfirmationEvent= threading.Event()
        if(self.StopRealTimeEvenet==None):
            self.StopRealTimeEvenet= threading.Event()
        self.StopRealTimeEvenet.clear()
        self.AlternatingBeepExitFlag=False
        #in case we use emulator
        self.VirtualPlant.Reset()
        #start processing thread
        #self.ProcessingRoutine()
        self.ProcThread= threading.Thread(target=self.ProcessingRoutine, args=[])
        self.ProcThread.start()

    def StopRealTime(self):
        
        self.label_Real_Time.setStyleSheet("background-color:#d62728")
        if(self.UserConfirmationEvent!=None):
            self.UserConfirmationEvent = None
        self.StopRealTimeEvenet.set()

        print("")
        print("*********************************************************************************************")
        print("Operation stopped at: ")
        now= datetime.now()
        print(now)
        print("*********************************************************************************************")
        print("")
        self.AlternatingBeepExitFlag=True

    def UserConfirmationClick(self):
        if(self.UserConfirmationEvent!=None):
            self.UserConfirmationEvent.set()
        self.AlternatingBeepExitFlag=True
    
    #functions
    def Exit_program_click(self):        
        self.close()
        sys.exit(app.exec())

    #Tab summator

    def BrowsRealTimeFolderClick(self):
        selected_directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.lineEdit_real_time_folder.setText(selected_directory)       
        self.Summator_real_time_folder_browse=self.lineEdit_real_time_folder.text()
        self.label_5.setText("Loading Etalons ...")
        self.label_5_ethalons_label.setStyleSheet("background-color:#e9d700")
        self.statusbar.setStyleSheet("background-color:#e9d700")
    
    def SummatorFunction(self,x,y,
                         summator_type="Laplacian_kernel",
                         err_type="average", # min,max                     
                        ):
        
        sh_1=np.shape(x)
        
        if(len(sh_1)==1):
            X=x.reshape(1,sh_1[0])
        else:
            X=x

        sh_2=np.shape(y)
        if(len(sh_2)==1):
            Y=y.reshape(1,sh_2[0])
        else:
            Y=y

        summ_type=summator_type
                
        #https://scikit-learn.org/stable/modules/metrics.html
        if(summ_type=="Heat_kernel"):           
            distance=sklearn.metrics.pairwise.rbf_kernel(X, Y=Y, gamma=1.8)
            distance=1-distance            
        if(summ_type == "Polin_kernel"):
            distance=sklearn.metrics.pairwise.polynomial_kernel(X, Y=Y, degree=3, gamma=0.6, coef0=1)
            distance=1-distance
        if(summ_type == "KME"):
            pass
        if(summ_type == "Laplacian_kernel"):
            distance = sklearn.metrics.pairwise.laplacian_kernel(X, Y=Y, gamma=0.6)
            distance=1-distance
        if(summ_type == "ChiSquare_kernel"):
            distance = sklearn.metrics.pairwise.chi2_kernel(X, Y=Y, gamma=0.6)
            distance=1-distance  
        if(summ_type == "Subtract"):
            distance=np.subtract(X,Y)
        
        err=0
        if(err_type=="average"):
            err=np.average(distance)
        if(err_type=="min"):
            err=np.min(distance)
        if(err_type=="max"):
            err=np.max(distance)
        if(err_type=="last"):
            err=distance[len(distance)-1]
            
        return distance,np.absolute(err)
        
    #**********************************************************************************
    #*************** COMPARE DATA USING SUMMATOR***************************************
    #**********************************************************************************
    
    def CompareWithEthalonsClick(self):
        if(self.Ethalon_values is None):
            print("Load etalons first and repeat the oepration")
            return
        if(self.DataToCompareWithEtalons is None):
            print("Load data and repeat the oeration...")
            return       

        summ_type_user = self.comboBox_controlALgorithm_summator_type.currentText()       
        err_type_user=  self.comboBox_controlALgorithm_summator_features_type_3.currentText()
        distance=self.SummatorFunction(self.Ethalon_values,self.DataToCompareWithEtalons,
                                       summator_type=summ_type_user,
                                       err_type = err_type_user
                                      )

        """
        if(summ_type=="Heat kernel"):
            distance=sklearn.metrics.pairwise.rbf_kernel(self.Ethalon_values, Y=self.DataToCompareWithEtalons, gamma=0.2)
        if(summ_type == "Polin.kernel"):
            distance=sklearn.metrics.pairwise.polynomial_kernel(self.Ethalon_values, Y=self.DataToCompareWithEtalons, degree=3, gamma=0.6, coef0=1)
        if(summ_type == "KME"):
            pass
        """
        print(distance)
            
    def LoadDataToCompareWithEtalons(self):
        selected_directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.Summator_feat_type=self.comboBox_controlALgorithm_summator_features_type_3.currentText()
        print("LOADING DATA...")
        self.label_5.setText("Loading Etalons ...")
        self.label_5_ethalons_label.setStyleSheet("background-color:#e9d700")
        self.statusbar.setStyleSheet("background-color:#e9d700")

        self.lineEdit_Beep_Duration_Upon_New_Params_6.setText(selected_directory)
        path=self.lineEdit_Beep_Duration_Upon_New_Params_6.text()
        self.DataToCompareWithEtalons = self.LoadVibrationFiles(path=path,feat=self.Summator_feat_type)
        str_time_of_load=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.label_3_check_dataload.setText(str_time_of_load)
        print("")
        print("#**************************************************")
        print("DATA IS LOADED")
        print("#**************************************************")
        print("")  
        self.label_5.setText("Etalons Loaded")
        self.label_5_ethalons_label.setStyleSheet("background-color:#2ca02c")
        self.statusbar.setStyleSheet("background-color:#2ca02c")
    
    #**********************************************************************************
    #*************** UPLOAD THE ETHALONS AND EXTRACT THE FREQUENCIES*******************
    #**********************************************************************************

    def Setfixedvalueasetalon(self):
        self.Ethalon_values = None
        value=float(self.lineEdit_etalonvalueparameter.text())
        self.Ethalon_values=np.array([value])
        self.label_5_ethalons_label.setStyleSheet("background-color:#2ca02c")
        
    def EraseEthalonsclick(self):
        self.Ethalon_values=None
        self.label_5_ethalons_label.setStyleSheet("background-color:#d62728")
        self.label_5.setText("Etalons cleaned")
        self.statusbar.setStyleSheet("background-color:#d62728")
        self.statusbar.showMessage("")

    def LoadEthalonFilesClick(self):

        print("")
        print("#**************************************************")
        print("LOADING ETHALON VALUES")
        print("#**************************************************")
        print("")
        self.label_5.setText("Loading Etalons ...")
        self.label_5_ethalons_label.setStyleSheet("background-color:#e9d700")
        self.statusbar.setStyleSheet("background-color:#e9d700")
        self.statusbar.showMessage("Processing Etalons ... ")

        self.Ethalon_values=None
        selected_directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.lineEdit_Beep_Duration_Upon_New_Params_5.setText(selected_directory)
        path=self.lineEdit_Beep_Duration_Upon_New_Params_5.text()
        self.Summator_feat_type=self.comboBox_controlALgorithm_summator_features_type_3.currentText()
        start_time = time.time()
        self.Ethalon_values = self.LoadVibrationFiles(path=path,feat=self.Summator_feat_type)
        stop_time=time.time()    
        shp=np.shape(self.Ethalon_values)
        if(np.all(np.asarray(shp))):        
            print("")
            print("#**************************************************")
            print("ETHALON VALUES ARE EXTRACTED")
            print("Shape of ethalon values: "+str(np.shape(self.Ethalon_values)))     
            print("FFT execution time(s): "+str(stop_time-start_time))#print("--- %s seconds ---" % (time.time() - start_time))
            print("")
            self.label_5_ethalons_label.setStyleSheet("background-color:#2ca02c")
            self.label_5.setText("Etalons loaded")
            self.statusbar.setStyleSheet("background-color:#2ca02c")
            self.statusbar.showMessage("Etalons Loaded")
            
        else:
            print("Fail to extract etalon values")
            self.Ethalon_values=None
            self.label_5_ethalons_label.setStyleSheet("background-color:#d62728")
            self.statusbar.setStyleSheet("background-color:#d62728")
            self.statusbar.showMessage("")
    
    #provided by KANNAN
    def LoadVibrationFiles(self,
                           path="",
                           lim=200000,
                           feat="FFT_spectra"
                          ):             

        data_extracted=None
        
        df = dpp.recentRawDataFrame(path, forceSensor=True)

        if(feat=="FFT_spectra"):            
            data_extracted = self.Extract_FFT(df=df,lim=lim)
                   
            #df = xm.dir_to_df(experimentsFolder = path, forceSensor=True)
        if(feat=="DF_features"):
            df_features, fail_freq = xm.failFeatures_freq(df)
            data_extracted=df_features
        if(feat=="JM_frequencies"):
            df_features, fail_freq = xm.failFeatures_freq(df)
            data_extracted=fail_freq

        sz_data = np.shape(data_extracted)
        if(len(sz_data)==1):
            data_extracted=np.expand_dims(data_extracted, axis = 0)
               
        return data_extracted

    #CHECK
    def Extract_FFT(self,df=[],lim=200000):
        
        stft_freq = []
        sfft_val = []
        lim = lim #200000
        df_sensor = df[df.columns[1]]        
        for i in range(0,df.shape[0],lim):
            
            signal = df_sensor[i:i+lim]                  
            signal_mean = signal - np.mean(signal)
            u_freq, v_time, w_val = stft(signal_mean, 10000, nperseg=256)
            w_val_abs = np.abs(w_val).max(axis=1)
            # w_val_abs[:5] = 0
            stft_freq.append(u_freq)            
            sfft_val.append(w_val_abs)
                
        stft_freq = np.asarray(stft_freq)
        sfft_val = np.asarray(sfft_val)
        
        
        sfft_val_n = (sfft_val - sfft_val.mean()) / (sfft_val.max() - sfft_val.min())
        sfft_val_n = np.asarray(sfft_val_n)               
        
        return sfft_val_n #stft_freq

    #**********************************************************************************
    #*************** XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX *******************
    #**********************************************************************************
    
    def UpdateGlobaProcSettings(self):

        print("")
        print("*************************************")
        print("Initialization")
        print("*************************************")
        print("")
        
        #first, read parameters names
        self.ParamNames=[]
        self.RangeMin=[]
        self.RangeMax=[]
        self.StartingParamsVals=[]
        
        for row in range(self.tableWidget_Params.rowCount()):
            for col in range(self.tableWidget_Params.columnCount()):
                item = self.tableWidget_Params.item(row, col)#.text()
                if(item !=None):
                    item=item.text()                
                    if(item!=""):
                        if(col==0):
                            self.ParamNames.append(item)
                        if(col==1):
                            self.RangeMin.append(float(item))
                        if(col==2):
                            self.RangeMax.append(float(item))      
                        if(col==3):                            
                            self.StartingParamsVals.append(float(item))
                            
        if(not len(self.StartingParamsVals)):
            for ip in range(0,len(self.ParamNames)):
                self.StartingParamsVals.append(0)

        print("")
        print("Parameters names: ")
        print(self.ParamNames)
        print("")
        
        #control algorithm
        self.Algorithm=None
        self.control_algotithm_name=self.comboBox_controlALgorithm.currentText()
        #choice of the systen dynamics - i.e. where to take the plant data
        self.system_dynamics_user_choice=self.comboBox_datasource.currentText()
        #if to round the results (U values will be rounded until that value)
        self.RoundUValuesFlag = self.checkBox_UValuesRound_flag.isChecked()
        self.RoundUValuesNumber = int(self.lineEdit_RoundUValuesNumber.text())

        #this algorithm generates only the randm numbers independently of the input
        if(self.control_algotithm_name=="Determenistic"):          
            #lorenz attractor - initialize the parameters            
            self.determenistic_time_period=int(self.lineEdit_Determenistic_time_period_text.text())
            self.determenistic_points_number=int(self.lineEdit_Determenistic_pointsnum_text.text())
            self.LorenzAt=add_s.LorenzAttr()
            self.LorenzAt.SetRange(min_r=self.RangeMin[0],max_r=self.RangeMax[0])            
            print("Determenistic pressure change initialized...")
                        
        if(self.control_algotithm_name=="Stochastic"):
            
            u0=[]
            for m in range(0,len(self.ParamNames)):
                u0.append(self.RangeMin[m]+(self.RangeMax[m]-self.RangeMin[m])/2)
            
            self.Algorithm=add_s.Optim_Control_1( U_initial=np.asarray(u0),
                                                  U_num=len(self.ParamNames),
                                                  U_min=np.asarray(self.RangeMin),
                                                  U_max=np.asarray(self.RangeMax),                       
                                                  max_iter=300,                       
                                                  f_cost=self.cost_function_Optim_Control_1,
                                                  Use_nonl_bounds=True,         
                                                )
            
                          
        print("")
        print("*************************************")
        print("Initialization complete successfully")
        if(self.checkBox_ProgressCharts.isChecked()==True):
            print("Control vizualization is on...")
        print("*************************************")

        #this are the service variables
        #beep
        self.BeepUponNewResults=self.comboBo_New_results_Sound.currentText()#checkBox_Beep_Upon_Mew_Params.isChecked()
        self.BeepDuration=float(self.lineEdit_Beep_Duration_Upon_New_Params.text())   
        self.AlternatingBeepThread=None
        #log info to console
        self.ConsoleLOG=self.checkBox_LOG_info_to_console.isChecked()
        
        #log  file
        self.FileLOG=self.checkBox_LOG_info_file.isChecked()
        #LOG file
        self.LOGFilePath=self.lineEdit_Beep_Duration_Upon_New_Params_2.text()
        #if to write limited number into log
        self.Limit_records_in_LOG_file_flag = self.checkBox_LOG_info_file_limit_flag.isChecked()
        self.Limit_records_in_LOG_file_num = int(self.lineEdit_info_file_limit_value.text())
        head, tail = os.path.split(self.LOGFilePath)
        head=head+"\\"
        self.Limit_records_filename=tail.split(".")[0]
        self.Limit_records_path=head
        
        if(self.FileLOG==True):            
            #prepare the ddata frame
            self.LOGDataFrame=pd.DataFrame()        
            self.LOGDataFrame["Control_input"]=None
            for k in range(0,len(self.ParamNames)):
                self.LOGDataFrame[self.ParamNames[k]]=None
            self.LOGDataFrame["Alg_Update_Time"]=None
            self.LOGDataFrame["User_Update_Time"]=None   
            #check paths and dirs 
            head_tail = os.path.split(self.LOGFilePath)
            if not os.path.exists(head_tail[0]):
                os.makedirs(head_tail[0])

        #prepare in case the file log is chosen to be split among the rest files
        self.cur_file_log_rec_counter=0
        self.cur_file_counter=0
        self.cur_file_path=self.LOGFilePath

        #show additional charts with live data
        self.ShowLiveData = self.checkBox_ProgressCharts.isChecked()
        self.Err_history=[]
        self.U_history=[]
        
        if(self.ShowLiveData==True):
            #plt.close('all')
            #matplotlib.use('TkAgg')
            self.Err_chart_w=add_s.RTPlotWidget_1()            
            #self.Err_chart_w.show()
            #self.U_chart_w=ChartWindow()
            #self.U_chart_w.show()     
            #this are the lists
            self.time_U=[]
            self.time_U_User=[]
            self.err=[]
            self.U=[]
        else:
            self.Err_chart_w=None
            self.U_chart_w=None

        #if to limit the time series
        self.Limit_Time_Series_Flag=self.checkBox_Limit_Time_Series_Flag.isChecked()
        self.Limit_Time_Series_Samples_Num=int(self.lineEdit_Series_samples_number.text())

        #invoke delays after each control action        
        self.Control_waiting_flag=self.comboBox_controlALgorithm_waitingtime_flag.currentText()
        self.Control_waiting_time_value=float(self.lineEdit_Beep_Duration_Upon_New_Params_3.text())
        self.BeepCancel=False

        #if the iperator wil provide the feedback or the machine will act on its own
        self.Feedback_confirmation=self.comboBox_controlALgorithm_feedback_confirmation_flag.currentText()
        
        #summator tab
        self.Summator_dist_measure=self.comboBox_controlALgorithm_summator_type.currentText()
        self.Summator_err_type = self.comboBox_controlALgorithm_summator_err_type_2.currentText()
        self.Summator_feat_type=self.comboBox_controlALgorithm_summator_features_type_3.currentText()
        self.Summator_real_time_folder_browse=self.lineEdit_real_time_folder.text()        

        #this is for optimal control and other condtrol algorithms
        self.xt_hist=[] #here we store the state variables
        self.err_hist=[]# here we store the errors
        self.u_hist=[]  #here we store control U signals
        self.integ_buf=[] #this is smoother for the state variables (polynomial at present)
        #this is event to handle the values in the main thread, passing values from the cost functions in Bayes and optimal control approaches
        self.Cost_function_data_ready=threading.Event()
        self.Cost_function_data_ready_flag=False
          
        return True
        
    def system_dynamics(self, u1, u2=None):
        
        output=[]
                
        if(self.system_dynamics_user_choice=="Rand_data"):     
            
            #generate just random data
            state_var_l = np.shape(self.Ethalon_values)         
            
            if(len(state_var_l)==1):
                state_var_l = state_var_l[0]
            elif(len(state_var_l)==2):
                state_var_l = state_var_l[1]            
                
            for k in range(0,state_var_l):               
                val_=random.uniform(-2, 2)
                output.append(val_)
            
        elif(self.system_dynamics_user_choice=="Linear_response"):
            u_=u1
            if(isinstance(u_, (collections.abc.Sequence, np.ndarray))):
                u_ = u1[0]
            state_var_l = np.shape(self.Ethalon_values)                     
            if(len(state_var_l)==1):
                state_var_l = state_var_l[0]
            elif(len(state_var_l)==2):
                state_var_l = state_var_l[1]    
                
            for k in range(0,state_var_l):
                val_=1.32*u_
                output.append(val_)
        
        elif(self.system_dynamics_user_choice=="Real_data"):
            #KANNAN CODE IS PLACED HERE - please, check this function
            x_t = self.LoadVibrationFiles(path = self.Summator_real_time_folder_browse)
            output=x_t
        return np.asarray(output)    
        
    
    def ALternatingBeep(self,duration):

        if(self.UserConfirmationEvent is None):
            return
        
        while(self.AlternatingBeepExitFlag==False):#not self.UserConfirmationEvent.is_set()):#(True):            
        
            if(self.AlternatingBeepExitFlag==True):                
                self.AlternatingBeepExitFlag==False
                self.UserConfirmationEvent.clear()
                return 
                                        
            if(self.UserConfirmationEvent.is_set()==True):                   
                self.AlternatingBeepExitFlag==False
                self.UserConfirmationEvent.clear()
                return
            
            winsound.Beep(2500, duration)#int(self.BeepDuration*1000))            
                                    
            if(self.UserConfirmationEvent.is_set()==True):  
                self.AlternatingBeepExitFlag==False
                self.UserConfirmationEvent.clear()
                return

            if(self.AlternatingBeepExitFlag==True):                
                self.AlternatingBeepExitFlag==False
                self.UserConfirmationEvent.clear()
                return    
                       
            time.sleep(duration/1000)
            
    
    #***********************************************************************************************************
    #***********************************************************************************************************
    #*********************** CONTROL ALGORITHMS ROUTINE ********************************************************

    # Class Optim_Control_1 cost function
    def cost_function_Optim_Control_1(self,u):
            
        POLY_DEGREE=3
        integ_time=10
        hist_length=20
        u_=u
        x_t = self.system_dynamics(u_)    

        if(isinstance(x_t, (collections.abc.Sequence, np.ndarray))):
            shp=np.shape(np.asarray(x_t))            
            if(len(shp)==1 and shp[0]==1):
                x_t=x_t[0]
                                
        if(self.integ_buf is not None):
            self.integ_buf.append(x_t) #err)
            if(len(self.integ_buf)>integ_time):
                del self.integ_buf[0]                
            #x_t = np.mean(integ_buf)#np.multiply(integ_buf,integ_buf))     
            if (len(self.integ_buf)>5):
                x = np.linspace(0,len(self.integ_buf),len(self.integ_buf))
                y=np.asarray(self.integ_buf)                    
                coeff = np.polyfit(x, y, POLY_DEGREE)
                p= np.poly1d(coeff)
                nex_x_t=p(x)
                x_t=nex_x_t[len(nex_x_t)-1]        
        
        pw_dist,err=self.SummatorFunction(self.Ethalon_values,
                                  x_t,
                                  summator_type = self.Summator_dist_measure,
                                  err_type=self.Summator_err_type
                                 )    
                
        err_result= err*err  # #np.abs(err) #err*err
        
        self.xt_hist.append(x_t)
        self.err_hist.append(err)
        self.u_hist.append(u)

        if(len(self.xt_hist) > hist_length):
            del self.xt_hist[0]
            del self.err_hist[0]
            del self.u_hist[0]
        
        self.Cost_function_data_ready_flag=True
        #self.Cost_function_data_ready.set()    
        
        while (True):#self.Cost_function_data_ready.is_set():
            #self.Cost_function_data_ready.wait(33)
            #if(self.Cost_function_data_ready.is_set()==False):
            if(self.Cost_function_data_ready_flag==False):
                break
            else:
                #print("COST FUNC: WAITING EVENT RESET")
                time.sleep(0.01)
        
        return err_result #alternative - np.abs(err) 

    #THIS FUNCTION HAS TO BE USED WITH HEAT KERNEL
    
    def Target_Bayes_Control_1(self, u1,wait_other_thread=True):
        hist_length=20
        x_t = self.system_dynamics(u1)
        pw_dist,err=self.SummatorFunction(self.Ethalon_values,
                                  x_t,
                                  summator_type = self.Summator_dist_measure,
                                  err_type=self.Summator_err_type
                                 )    
        #target=-1*(err-1)
        target=np.exp(-err*err)
        
        self.xt_hist.append(x_t)
        self.err_hist.append(err)
        self.u_hist.append(u1)

        if(len(self.xt_hist) > hist_length):
            del self.xt_hist[0]
            del self.err_hist[0]
            del self.u_hist[0]

        if(wait_other_thread==True):
            self.Cost_function_data_ready_flag=True    
            while (True):
                if(self.Cost_function_data_ready_flag==False):
                    break
                else:                
                    time.sleep(0.01)
        print("Bayes control target:" +str(target))
        return target

    #************************************************************************************************************
    #************************************************************************************************************
    #************************************************************************************************************
    #*******************ROUTINE FOR CONSOLE OUTPUT***************************************************************
    def ConsoleLOG_TimeOutput(self,time_params_update, cnt=None):
        step_str=""
        if cnt==None:
            step_str="_step num. undefined_"
        else:
            step_str=str(cnt)
        if(self.ConsoleLOG==True):               
            print("--------- Step " +str(step_str)+ " -----------")
            print("Algorithm parameters update: "+str(time_params_update))

    #this is the main window output - the signals U are shown
    def Show_U_Signals_For_Operator(self,u_t):
            self.ParamsValuesString=""         
            operator_string=""
            for i in range (0,len(self.ParamNames)):                                
                operator_string=operator_string + str(self.ParamNames[i])+" : "
                operator_string=operator_string + str(u_t[-1])       
            self.ParamsValuesString=operator_string
            self.lineEdit_Params_Values.setStyleSheet("color: rgb(255, 0, 0);")
            self.lineEdit_Params_Values.setText("")
            self.lineEdit_Params_Values.setText(self.ParamsValuesString)

    #wait for operator if he confirms
    def WaitTheOperatorConfirmation(self):
        
        if(self.Feedback_confirmation == "Operator confirms"):        
                #***************************SOUNDS**************************************                
                if(self.BeepUponNewResults=="Beep_continious"):                
                   winsound.Beep(2500, int(self.BeepDuration*1000))
                if(self.BeepUponNewResults=="Beep_alternating"):   
                   if(self.AlternatingBeepThread is not None):
                       if(self.AlternatingBeepThread.isBuisy()==True):
                           self.AlternatingBeepExitFlag=False
                           self.UserConfirmationEvent.set()
                           while (True):
                               if(self.AlternatingBeepThread.isBuisy()==False):
                                   break                           
                   #self.ALternatingBeep(int(self.BeepDuration*1000))
                   self.AlternatingBeepThread = threading.Thread(target = self.ALternatingBeep, args = (int(self.BeepDuration*1000),))                
                   self.AlternatingBeepThread.start()                                
                if(self.BeepUponNewResults=="No_beep"):
                   pass                               
                #**************************END SOUNDS***********************************
                
                #**************************WAIT OPERATOR INPUT**************************
                #wait user input
                self.UserConfirmationEvent.wait()
                self.UserConfirmationEvent.clear()

        else:
            pass

        self.lineEdit_Params_Values.setStyleSheet("color: rgb(0, 255, 0);")

    #output the string after the operator confirmed the input
    def OperatorConfirmed_U_Input(self,time_user_confirm):
        #time_user_confirm = datetime.now()            
         if(self.ConsoleLOG==True):
             print("User parameters input confirmed: "+str(time_user_confirm))
             print("")

    def Write_File_LOG(self,err_t=0,
                       u_t=[],
                       cnt=1,
                       time_params_update="",
                       time_user_confirm=""):        
            
        if(self.FileLOG==True):   
                
                if(self.Limit_records_in_LOG_file_flag==True):
                    if(self.cur_file_log_rec_counter>=self.Limit_records_in_LOG_file_num):
                        self.cur_file_log_rec_counter=0
                        self.cur_file_counter=self.cur_file_counter+1
                    else:
                        self.cur_file_log_rec_counter=self.cur_file_log_rec_counter+1
                    file_name=self.Limit_records_filename+"_"+str(self.cur_file_counter)+str(".csv")
                    self.cur_file_path=self.Limit_records_path + file_name
                else:
                    self.cur_file_path=self.LOGFilePath
                                      
                FileLOG_df=None
                FileLOG_df = pd.DataFrame()   
                FileLOG_df.loc[1, 'Step']=str(cnt)
                FileLOG_df.loc[1, 'Control_input'] = str(err_t) #FileLOG_df["Control_input"].append(err)               
                for k in range(0,len(self.ParamNames)):
                    FileLOG_df.loc[1, self.ParamNames[k]] = str(u_t[k])         
                FileLOG_df.loc[1, "Alg_Update_Time"] = str(time_params_update)             
                FileLOG_df.loc[1, "User_Update_Time"] = str(time_user_confirm)                             
                #self.LOGDataFrame = pd.concat([self.LOGDataFrame, FileLOG_df], ignore_index=True)                
                #save to file
                try:
                    if(cnt==1):
                        FileLOG_df.to_csv(self.cur_file_path, mode='a',sep='\t', index=False,header=True)
                    else:
                        FileLOG_df.to_csv(self.cur_file_path, mode='a',sep='\t', index=False,header=False)
                except:
                    print("")
                    print("Faild to save data to file: step "+str (cnt))

    #show the data in a separate windows
    def ShowLiveDataInCharts(self, time_params_update="", time_user_confirm=""):
        
        if(self.ShowLiveData==True):
                
                try:                    
                    
                    self.time_U.append(time_params_update)
                    self.time_U_User.append(time_user_confirm)
                    self.err.append(err_t)
                    self.U.append(u_t)
                    #check if we have a limit for the storage of the time series
                    
                    if(self.Limit_Time_Series_Flag==True):
                        if(len(self.time_U)>self.Limit_Time_Series_Samples_Num):
                            del self.time_U[0]
                            del self.time_U_User[0]
                            del self.err[0]
                            del self.U[0]  
                            
                    self.Err_chart_w.Canvas.axes.clear()
                    self.U_chart_w.Canvas.axes.clear()
                                        
                    try:
                        
                        err_line,=self.Err_chart_w.Canvas.axes.plot(self.time_U,self.err)    
                        err_line.set_label('Control errors')
                        self.Err_chart_w.Canvas.axes.set_title(self.Err_chart_w.chart_name)
                        self.Err_chart_w.Canvas.axes.legend(loc = "upper left")
                        self.Err_chart_w.Canvas.fig.canvas.draw()
                        lines_u=[]        
                        
                        for b in range(0,len(self.ParamNames)):
                            series=np.asarray(self.U)[:,b]                            
                            line_u,=self.U_chart_w.Canvas.axes.plot(self.time_U,series)
                            line_u.set_label(self.ParamNames[b])
                            lines_u.append(line_u)
                            
                        self.U_chart_w.Canvas.axes.legend(loc = "upper left")
                        self.U_chart_w.Canvas.axes.set_title(self.U_chart_w.chart_name)
                        self.U_chart_w.Canvas.fig.canvas.draw()
                        
                    except Exception as e:
                        print("exception raised while data visualization: ")    
                        print(e)
                    
                except Exception as es:
                    print("exception raised while data history storage: ")    
                    print(es)

    #waiting time if was selected
    def WaitSomeTime(self):
         if(self.Control_waiting_flag=="Waiting after each control action with duration (s):"):
            if(self.Control_waiting_time_value>0):
                time.sleep(self.Control_waiting_time_value)
         else:
            pass
        

    #************************************************************************************************************
    #************************************************************************************************************
    #************************************************************************************************************
        
    def ProcessingRoutine(self):
        
        if(self.Ethalon_values is None) and (self.control_algotithm_name!="Determenistic"):
            print("Load etalon values and repeat...")
            print("Cant launch real time without etalon values...")
            self.StopRealTime()
            return

        redColor = QColor(255, 0, 0)
        greenColor=QColor(0, 255, 0)

        time_params_update=0
        time_user_confirm=0
        FileLOG_df=None
        cont_response=np.zeros((len(self.ParamNames)))     

        self.time_U=[]
        self.time_U_User=[]
        self.err=[]
        self.U=[]

        #prepare in case the file log is chosen with the 
        self.cur_file_log_rec_counter=0
        self.cur_file_counter=0
        self.cur_file_path=self.LOGFilePath

        print("")
        print("*********************************************************************************************")
        print("Start operation at: ")
        now=datetime.now()
        print(now)
        print("*********************************************************************************************")
        print("")
        
        cnt=1
        #starting conditions and initializations
        u_t=[]
        err_t=0
        pw_dist=0 #pairwise distance
                                    
        while(True):            

            if(self.StopRealTimeEvenet.is_set()):                
                return

            #**************************************************************************************************
            #**************************************************************************************************
            #**************** CONTROL ALGORITHMS **************************************************************
            
            #generation of random U values independently from the input - the choice follows the uniform distribution
            
            if(self.control_algotithm_name=="Determenistic"): #type(self.Algorithm)==EmulationControl):      
                
                x,y,z=self.LorenzAt.integrate_ext(tmax=self.determenistic_time_period,n=self.determenistic_points_number)
                if(self.RoundUValuesFlag == True):
                    x = np.round(x,self.RoundUValuesNumber)    
                u_t.append(x)        
                if(self.Limit_Time_Series_Flag==True):
                    if(len(u_t)>self.Limit_Time_Series_Samples_Num):
                        while(len(u_t)>self.Limit_Time_Series_Samples_Num):
                            del u_t[0]
               
            #**************************************************************************************************
            #**************************************************************************************************                

            # Bayes gradient -classical approach
            if(self.control_algotithm_name=="Determenistic"):#type(self.Algorithm)==Bayes_Control_2):                
                next_point = self.Algorithm.optimizer.suggest()                
                target = self.Target_Bayes_Control_1(**next_point,wait_other_thread=False)          
                try:                    
                    self.Algorithm.optimizer.register(params = next_point, target = target)
                except:
                    pass
                if(self.ConsoleLOG==True):
                    print("Best result: {}; f(x) = {:.3f}.".format(self.Algorithm.optimizer.max["params"], self.Algorithm.optimizer.max["target"]))                
                d_l=len(self.xt_hist)
                x_t =  self.xt_hist[d_l-1]
                err_t = self.err_hist[d_l-1]
                u_t=self.u_hist[d_l-1]    

                if(self.RoundUValuesFlag == True):
                    u_t = np.round(u_t,self.RoundUValuesNumber)    
                
            #****************************************************************************************
            #****************************OUTPUT******************************************************
            #this is console output LOG if user selected
            time_params_update = datetime.now()     
            self.ConsoleLOG_TimeOutput(time_params_update,cnt=cnt)
            #this is the output to the main textbox for operator
            self.Show_U_Signals_For_Operator(u_t)            
            if(self.StopRealTimeEvenet.is_set()):                
                return
            #*******************************************************************************
            #if the operator has to put the new paraeters inside we wait him to conform here
            #*******************************************************************************
            self.WaitTheOperatorConfirmation()                       
            #self.lineEdit_Params_Values.setStyleSheet("color: rgb(0, 255, 0);")           
            
            time_user_confirm = datetime.now()
            self.OperatorConfirmed_U_Input(time_user_confirm)
            
            if(self.StopRealTimeEvenet.is_set()):                
                return

            self.Write_File_LOG(err_t=err_t,
                                u_t=u_t,
                                cnt=cnt,
                                time_params_update=time_params_update,
                                time_user_confirm=time_user_confirm
                               )
                                    
            #***************************************************
            #**************SHOW PLOTS IF NEEDED*****************
            #***************************************************
            #self.ShowLiveDataInCharts(time_params_update=time_params_update, 
            #                          time_user_confirm=time_user_confirm
            #                         )
            
            if(self.ShowLiveData==True):
                
                try:                    
                    
                    self.time_U.append(time_params_update)
                    self.time_U_User.append(time_user_confirm)
                    self.err.append(err_t)
                    self.U.append(u_t[-1])
                    #check if we have a limit for the storage of the time series
                    
                    if(self.Limit_Time_Series_Flag==True):
                        if(len(self.time_U)>self.Limit_Time_Series_Samples_Num):
                            del self.time_U[0]
                            del self.time_U_User[0]
                            del self.err[0]
                            del self.U[0]  
                            
                    self.Err_chart_w.plot(u_t)
                    #self.U_chart_w.Canvas.axes.clear()
                    
                except Exception as e:
                    print("exception raised while data history storage: ")    
                    print(e)
            
            
            if(self.StopRealTimeEvenet.is_set()):                
                return

            #**********************************************************************************
            #**********************WAITING TIME************************************************
            #**********************************************************************************
            self.WaitSomeTime()            
            #**********************************************************************************
            #**********************END OF WAITING**********************************************
            #**********************************************************************************
            
            cnt=cnt+1
            self.label_8_action_number_label.setText(str(cnt))
            
            if(self.StopRealTimeEvenet.is_set()):                
                return

            #clear all events
            if(self.Cost_function_data_ready_flag==True):
                #if(self.Cost_function_data_ready_flag.is_set())):
                    #self.Cost_function_data_ready.clear()
                self.Cost_function_data_ready_flag=False                        

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())