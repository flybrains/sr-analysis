import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import string

def combine_data(PATHSTRING):
    
    #     Input: Path to top level directory for SR data
    #         Currently: r"C:\Users\Patrick\Datasets\sr\SR
    #     Output: List of tuples containting paths in format [Path to Data1, Path to Data2]
    
    #Get all datasets
    all_paths = [x[0] for x in os.walk(PATHSTRING)]
    
    #Segment betwen Dataset1 (Motion and GCaMP) and Dataset 2 (Timing)
    dataset1_paths = []
    dataset2_paths = []
    
    for path in all_paths:
        if "Traces" in path:
            dataset2_paths.append(path)
    
    for path in dataset2_paths:
        splits = path.split('\\')
        new_name = '\\'.join(splits[0:-1])
        new_name = new_name.replace('Data2', 'Data')
        dataset1_paths.append(new_name)
    
    #Generate tuples for both datasets for each trial:
    trial_paths = []
    for i in range(len(dataset1_paths)):
        trial = [dataset1_paths[i], dataset2_paths[i]]
        trial_paths.append(trial)
    
    return trial_paths



def CIRF_conv(FicTrac_Time_Real, timeseries):       
    # ---------------------------------------------------------------------
    # INPUT: Time series kinematic data from experimental data collection, dim([1,N])
    # RETURNS: Time series convolved with CIRF calcium dynamics function
    # ---------------------------------------------------------------------
    CIRF = -1*(np.power(2, (-FicTrac_Time_Real/0.175)) - np.power(2, (-FicTrac_Time_Real/0.55)))
    ts = timeseries
    #Pad the ts Data
    ts_Padded_Front = np.append((np.zeros((1,ts.shape[0]),dtype='float64')), (np.transpose(ts)))
    ts_Padded = np.append(ts_Padded_Front, (np.zeros((1,ts.shape[0]),dtype='float64')))
    #Convolution
    ts_Conv_Padded=np.convolve(ts_Padded, CIRF)
    #UnPad
    ts_Conv=ts_Conv_Padded[(ts.shape[0]+1):(ts.shape[0]*2)]
    return ts_Conv



class Trial(object):
    def __init__(self, name, data1_path, data2_path):
        self.name = name
        self.data1_path = data1_path
        self.data2_path = data2_path
        
        self.time_path = data1_path+r'\GCaMP_Time_Pre.csv'
        self.gcamp_data_path = data1_path + r'\ROI-profiles.txt'
        
        self.kinematic_data_path = data1_path+"\\"+ [x for x in os.listdir(self.data1_path) if x.endswith('.dat')][0]
        
        self.start_stop_data_path = data2_path + r'\StopStart_Times_Exact.csv'
        self.light_times_path = data1_path + r'\Light_Times.xlsx'
        
    def get_GCaMP_data(self):
        #Returns GCaMP data in pandas dataframe
        #Assigns object attributes:
        #   self.Trace_Length
        #   self.GCaMP_data
        #Combines/Processes GCaMP time data
        #------------------------------------------------
        #Open & Load Times
        f = open(self.time_path, 'r')
        times = [float(i) for i in (f.read().split(','))]
        GCaMP_Time_Pre = np.asarray(times)
        f.close()
        
        GCamp_Time = GCaMP_Time_Pre - GCaMP_Time_Pre[0]
        self.Trace_Length = GCamp_Time[-1-9]
        #------------------------------------------------
        #Open and Load GCaMP data into dataframe
        df = pd.read_csv(self.gcamp_data_path, header=None)
        df_trans = df.T
        tag = df_trans.iloc[0][0]
        header = df_trans.iloc[1]
        for idx, item in enumerate(header):
            if idx < 8:
                header[idx] = item.strip() + '_red'
            else:
                header[idx] = item.strip() + '_green'
    
        df_trans.drop(axis=0, index=[0,1], inplace=True)
        df_trans = df_trans.reset_index(drop=True)
        df_trans.columns=header.T
        df_trans['time'] = pd.Series(times)
        self.GCaMP_data = df_trans
        
        return None
        #------------------------------------------------    

    def get_kinematic_data(self):
        #Opens and loads kinematic data
        #Processes kinematic data into meaningful quantities
        #------------------------------------------------    
        #Opens and loads columns of .dat
        FicTracData = np.genfromtxt(self.kinematic_data_path, delimiter=',')
        
        xpos_pre=FicTracData[:,14]
        ypos_pre=FicTracData[:,15]
        Roll_Pre=FicTracData[:,5]
        Pitch_Pre=FicTracData[:,6]
        Ang_Pre=FicTracData[:,7]
        Heading_Pre=FicTracData[:,16]
        Frames_Pre=FicTracData[:,0].astype(int)
        #------------------------------------------------
        #Normalize heading between pi and negative pi
        pi = 3.14159
        Heading1=((Heading_Pre+pi)%(2*pi))-pi;

        #Import The time that the light turns on            
        import openpyxl
        wb=openpyxl.load_workbook(self.light_times_path)

        sheet = wb['Sheet1']
        Lights_On_1 = sheet['A1'].value - 1
        Lights_On_2 = sheet['A2'].value - 1

        #Get Proper Index for start and stop
        Start_frame=np.where(Frames_Pre==Lights_On_1)[0][0]
        End_frame=np.where(Frames_Pre==Lights_On_2)[0][0]
        #------------------------------------------------
        # Cut motion data appropriately
        Roll=Roll_Pre[Start_frame:End_frame]
        Pitch=Pitch_Pre[Start_frame:End_frame]
        Ang=Ang_Pre[Start_frame:End_frame]
        Heading=Heading1[Start_frame:End_frame]
        Frames=Frames_Pre[Start_frame:End_frame]
        RATE=(Frames[-1]-Frames[0])/self.Trace_Length
        FicTrac_Time_Real1=Frames*(1/RATE)
        self.FicTrac_Time_Real=FicTrac_Time_Real1-FicTrac_Time_Real1[0]
        xpos_PRE=xpos_pre[Start_frame:End_frame]
        ypos_PRE=ypos_pre[Start_frame:End_frame]
        xpos=xpos_PRE-xpos_PRE[0];
        ypos=ypos_PRE-ypos_PRE[0];
        #------------------------------------------------
        #Combine Roll, Pitch, and Ang to create Motion
        Motion = np.linalg.norm([Pitch, Ang, Roll], axis=0)
        #------------------------------------------------ 
        self.Roll_data = Roll
        self.Pitch_data = Pitch
        self.Ang_data = Ang
        self.Motion_data = Motion
        
        return None
        #------------------------------------------------
        
    def process_GCaMP_data(self):
        #------------------------------------------------
        #Takes regional mean saturation values
        #------------------------------------------------
        self.G2_R=self.GCaMP_data['G2R_green']/self.GCaMP_data['G2R_red']
        self.G3_R=self.GCaMP_data['G3R_green']/self.GCaMP_data['G3R_red']
        self.G4_R=self.GCaMP_data['G4R_green']/self.GCaMP_data['G4R_red']
        self.G5_R=self.GCaMP_data['G5R_green']/self.GCaMP_data['G5R_red']

        self.G2_L=self.GCaMP_data['G2L_green']/self.GCaMP_data['G2L_red']
        self.G3_L=self.GCaMP_data['G3L_green']/self.GCaMP_data['G3L_red']
        self.G4_L=self.GCaMP_data['G4L_green']/self.GCaMP_data['G4L_red']
        self.G5_L=self.GCaMP_data['G5L_green']/self.GCaMP_data['G5L_red']

        self.G2_AVG= np.mean(np.concatenate((self.G2_L, self.G2_R), axis=0))
        self.G3_AVG=np.mean(np.concatenate((self.G3_L, self.G3_R),axis=0))
        self.G4_AVG=np.mean(np.concatenate((self.G4_L, self.G4_R),axis=0))
        self.G5_AVG=np.mean(np.concatenate((self.G5_L, self.G5_R),axis=0))

    
    def process_kinematic_data(self):
        self.Ang_Conv = CIRF_conv(self.FicTrac_Time_Real, self.Ang_data)
        self.Pitch_Conv = CIRF_conv(self.FicTrac_Time_Real, self.Pitch_data)
        self.Roll_Conv = CIRF_conv(self.FicTrac_Time_Real, self.Roll_data)
        self.Motion_Conv = CIRF_conv(self.FicTrac_Time_Real, self.Motion_data)
        return None



def generate_trial_objects(path_tuples):
    trial_objects = []
    for trial_path in path_tuples:
        name = trial_path[0].split('\\')[-1]
        data1_path = trial_path[0]
        data2_path = trial_path[1]
        
        new_trial = Trial(name, data1_path, data2_path)
        new_trial.get_GCaMP_data()
        new_trial.get_kinematic_data()
        trial_objects.append(new_trial)
    return trial_objects  



def process_trial_data(list_of_trials):
    processed_trials = []
    for trial in list_of_trials:
        trial.process_kinematic_data()
        trial.process_GCaMP_data()
        processed_trials.append(trial)
    return processed_trials