import opensim
from opensim import InverseKinematicsTool
from opensim import Model
from opensim import ScaleTool
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from linecache import getline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from decimal import Decimal

def judge_line_number(path):
    count = 1
    f = open(path,"r")
    line = f.readline()
    while line!="":
        s=line[:4]
        if s.upper()=="time".upper():
          return count
        else:
          count = count + 1
          line = f.readline()

def generate_dict(path):
    def sto_Getline(path):
      return getline(path, judge_line_number(path)).strip('\n').split()
    # for each in dict(zip(sto_Getline(path), range(0,len(sto_Getline(path))))):
    #   print(each, ':', dict(zip(sto_Getline(path), range(0,len(sto_Getline(path)))))[each])

def load_file_1(file_name):
    c = open(file_name)
    for x in range(judge_line_number(file_name)):
        next(c)
    for i in c.readlines():
        m = i.strip('\n').split()
        time.append(Decimal(m[0]))
        ankle_angle_r.append(Decimal(m[11]))
        ankle_angle_l.append(Decimal(m[18]))
        knee_angle_r.append(Decimal(m[10]))
        knee_angle_l.append(Decimal(m[17]))
        hip_flexion_r.append(Decimal(m[7]))
        hip_flexion_l.append(Decimal(m[14]))
    c.flush()
    c.close()

a = Model("C:/Users/giaco/Documents/OpenSim/4.5/Models/Gait2354_Simbody/gait2354_simbody.osim")
print("bodyset:")
for d in a.getBodySet():
    print("  " + d.getName())
print()
print("Jointset:")
for d in a.getJointSet():
    print("  " + d.getName())
print()
print("Forceset:")
for d in a.get_ForceSet():
    print("  " + d.getName())
print()
print("Markerset:")
for d in a.getMarkerSet():
    print("  " + d.getName())
print()
print("Probeset:")
for d in a.get_ProbeSet():
    print("  " + d.getName())
print()
print("FrameList:")
for d in a.getFrameList():
    print(d.getName())
print()

w = Model("C:/Users/giaco/Documents/OpenSim/4.5/Models/Gait2354_Simbody/OutputReference/subject01_simbody.osim")
state = w.initSystem()
count = 0
for d in w.getMuscleList():
    count = count + 1
print(f'muscles: {count} \nbodies: {w.getNumBodies()} \ndegree: {w.getNumCoordinates()}')

start_time = time.time()
ScaleTool("C:/Users/giaco/Documents/OpenSim/4.5/Models/Gait2354_Simbody/subject01_Setup_Scale.xml").run()
print(f'The execution time of ScaleTool is {(time.time() - start_time)} sec')

start_time = time.time()
InverseKinematicsTool("C:/Users/giaco/Documents/OpenSim/4.5/Models/Gait2354_Simbody/subject01_Setup_IK.xml").run()
print(f'The execution time of InverseKinematicsTool is {(time.time() - start_time)} sec')

generate_dict("C:/Users/giaco/Documents/OpenSim/4.5/Models/Gait2354_Simbody/subject01_walk1_ik.mot")

time, ankle_angle_r, ankle_angle_l, knee_angle_r,knee_angle_l= [],[],[],[],[]
hip_flexion_r, hip_flexion_l=[],[]

load_file_1("C:/Users/giaco/Documents/OpenSim/4.5/Models/Gait2354_Simbody/subject01_walk1_ik.mot")

df1 = pd.DataFrame({'ankle_angle_r': ankle_angle_r, 'ankle_angle_l': ankle_angle_l}, index = time)
df2 = pd.DataFrame({'knee_angle_r': knee_angle_r, 'knee_angle_l': knee_angle_l}, index = time)
df3 = pd.DataFrame({'hip_flexion_r': hip_flexion_r, 'hip_flexion_l': hip_flexion_l}, index = time)
df1.index.name = df2.index.name = df3.index.name = 'time'
df1 = df1.apply(pd.to_numeric, errors='coerce')
df2 = df2.apply(pd.to_numeric, errors='coerce')
df3 = df3.apply(pd.to_numeric, errors='coerce')

ankle_angle_l_float = [float(value) for value in ankle_angle_l]
print(ankle_angle_l_float)
ankle_angle_r_float = [float(value) for value in ankle_angle_r]
print(ankle_angle_r_float)
knee_angle_l_float = [float(value) for value in knee_angle_l]
print(knee_angle_l_float)
knee_angle_r_float = [float(value) for value in knee_angle_r]
print(knee_angle_r_float)
hip_flexion_l_float = [float(value) for value in hip_flexion_l]
print(hip_flexion_l_float)
hip_flexion_r_float = [float(value) for value in hip_flexion_r]
print(hip_flexion_r_float)


df1.plot(title='ankle_angle_IK'); plt.show()
df2.plot(title='knee_angle_IK'); plt.show()
df3.plot(title='hip_flexion_IK'); plt.show()