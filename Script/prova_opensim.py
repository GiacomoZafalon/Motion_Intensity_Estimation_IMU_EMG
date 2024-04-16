import opensim as osim
import pandas as pd
import matplotlib.pyplot as plt

# # Create an xsensDataReader and supply the settings file that maps IMUs to your model
# xsens_settings = osim.XsensDataReaderSettings('myIMUMappings.xml')
# xsens_reader = osim.XsensDataReader(xsens_settings)

# # Read the quaternion data and write it to a STO file for in OpenSense workflow
# tables = xsens_reader.read('IMUData/')
# quaternion_table = xsens_reader.getOrientationsTable(tables)

# osim.STOFileAdapterQuaternion.write(quaternion_table, xsens_settings.get_trial_prefix() + '_orientations.sto')

# Setup and run the IMUPlacer tool, with model visualization set to true
imu_placer = osim.IMUPlacer('c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/myIMUPlacer_Setup.xml')
imu_placer.run(True)

# Write the calibrated model to file
calibrated_model = imu_placer.getCalibratedModel()

# Setup and run the IMU IK tool with visualization set to true
imu_ik = osim.IMUInverseKinematicsTool('c:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/myIMUIK_Setup.xml')
imu_ik.run(True)

# Read the motion data file into a DataFrame
motion_data = pd.read_csv('C:/Users/giaco/Documents/OpenSim/4.5/Code/Python/OpenSenseExample/IKResults/ik_lifting_orientations.mot', delimiter='\t', skiprows=6)

# Plot the values of the 'knee_angle_r' column
plt.plot(motion_data['time'], motion_data['elbow_flex_r'])
plt.xlabel('Time')
plt.ylabel('Elbow Angle (degrees)')
plt.title('Right Elbow Angle')
plt.grid(True)
plt.show()