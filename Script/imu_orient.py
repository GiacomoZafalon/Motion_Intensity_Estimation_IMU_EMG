import numpy as np
import math
from scipy.spatial.transform import Rotation
import warnings
from numpy.linalg import norm
import time
import numbers
import pandas as pd
import torch

class Quaternion:
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w_or_q, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion, another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        self._q = np.array([1, 0, 0, 0])

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        elif isinstance(w_or_q, Quaternion):
            q = np.array(w_or_q.q)
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as parameters")

        self.q = q

    # Quaternion specific interfaces

    def conj(self):
        """
        Returns the conjugate of the quaternion
        :rtype : Quaternion
        :return: the conjugate of the quaternion
        """
        return Quaternion(self._q[0], -self._q[1], -self._q[2], -self._q[3])

    def __mul__(self, other):
        """
        multiply the given quaternion with another quaternion or a scalar
        :param other: a Quaternion object or a number
        :return:
        """
        if isinstance(other, Quaternion):
            w = self._q[0]*other._q[0] - self._q[1]*other._q[1] - self._q[2]*other._q[2] - self._q[3]*other._q[3]
            x = self._q[0]*other._q[1] + self._q[1]*other._q[0] + self._q[2]*other._q[3] - self._q[3]*other._q[2]
            y = self._q[0]*other._q[2] + self._q[2]*other._q[0] + self._q[3]*other._q[1] - self._q[1]*other._q[3]
            z = self._q[0]*other._q[3] + self._q[3]*other._q[0] + self._q[1]*other._q[2] - self._q[2]*other._q[1]

            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)

    def __add__(self, other):
        """
        add two quaternions element-wise or add a scalar to each element of the quaternion
        :param other:
        :return:
        """
        if not isinstance(other, Quaternion):
            if len(other) != 4:
                raise TypeError("Quaternions must be added to other quaternions or a 4-element array")
            q = self._q + other
        else:
            q = self._q + other._q

        return Quaternion(q)

    # Implementing other interfaces to ease working with the class

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, q):
        self._q = q

    def __getitem__(self, item):
        return self._q[item]

    def __array__(self):
        return self._q
    
def quaternion_to_euler(quaternion):
    """
    Convert quaternion [w, x, y, z] to Euler angles (roll, pitch, yaw).
    """
    # Extract components
    w, x, y, z = quaternion

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use +-90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.

    Parameters:
        q (list): A list containing the quaternion [q0, q1, q2, q3].

    Returns:
        np.array: A 3x3 rotation matrix representing the rotation.
    """
    q0, q1, q2, q3 = q
    rotation_matrix = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ])
    return rotation_matrix

# class InitialOrientation:
def initial_roll_pitch_yaw_from_acc_mag(accelerometer, magnetometer):
    ax, ay, az = accelerometer
    mx, my, mz = magnetometer
    g = 9.80665
    roll = math.atan2(ay, az) 
    pitch_denom = np.sqrt(ay**2 + az**2)
    pitch = math.atan2(ax, pitch_denom)
    # pitch = math.atan2(-ax, ay*np.sin(roll) + az*np.cos(roll))
    Vx, Vy, Vz = 0, 0, 0
    yaw_num = (my-Vy)*np.cos(roll) + (mz-Vz)*np.sin(roll)
    yaw_den = (mx-Vx)*np.cos(pitch) - (mz-Vz)*np.sin(pitch)
    yaw = math.atan2(yaw_num, yaw_den)

    roll_deg, pitch_deg, yaw_deg = np.degrees([roll, pitch, yaw])

    roll /= 2.0
    pitch /= 2.0
    yaw /= 2.0

    # Calculate trigonometric functions
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Compute quaternion components
    qw = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
    qx = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
    qy = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
    qz = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw

    # qw = cos_yaw * cos_pitch * cos_roll - sin_yaw * sin_pitch * sin_roll
    # qx = sin_yaw * sin_pitch * cos_roll + cos_yaw * cos_pitch * sin_roll
    # qy = cos_yaw * sin_pitch * cos_roll - sin_yaw * cos_pitch * sin_roll
    # qz = sin_yaw * cos_pitch * cos_roll + cos_yaw * sin_pitch * sin_roll

    # Normalize quaternion
    magnitude = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw /= magnitude
    qx /= magnitude
    qy /= magnitude
    qz /= magnitude

    quaternion = np.array([qw, qx, qy, qz])

    return roll_deg, pitch_deg, yaw_deg, quaternion

# Global system variables
# b_x, b_z = 1, 0  # reference direction of flux in earth frame
w_bx, w_by, w_bz = 0, 0, 0  # estimate gyroscope biases error

# Madgwick Filter
# Function to compute one filter iteration
def filterUpdate(acc, gyro, mag, init_quat=None, Ts=None, beta=None, zeta=None, gyroError=None, gyroDrift=None):
    global w_bx, w_by, w_bz

    if init_quat is None:
        init_quat = np.array([1, 0, 0, 0])
    if Ts is None:
        Ts = 0.01
    if gyroError is None:
        gyroMeasError =  math.pi * (5.0 / 180.0)  # gyroscope measurement error in rad/s (shown as 5 deg/s)
    if gyroDrift is None:
        gyroMeasDrift = 0 # math.pi * (0.1 / 180.0)  # gyroscope measurement error in rad/s^2 (shown as 0.2 deg/s^2)
    if beta is None:
        beta = math.sqrt(3.0 / 4.0) * gyroMeasError  # compute beta
    if zeta is None:
        zeta = math.sqrt(3.0 / 4.0) * gyroMeasDrift  # compute zeta
        
    SEq_1, SEq_2, SEq_3, SEq_4 = init_quat
    deltat = Ts
    
    w_x, w_y, w_z = gyro
    a_x, a_y, a_z = acc
    m_x, m_y, m_z = mag

    # local system variables
    # vector norm
    acc_norm = math.sqrt(a_x * a_x + a_y * a_y + a_z * a_z)
    a_x /= acc_norm
    a_y /= acc_norm
    a_z /= acc_norm

    # normalise the magnetometer measurement
    mag_norm = math.sqrt(m_x * m_x + m_y * m_y + m_z * m_z)
    m_x /= mag_norm
    m_y /= mag_norm
    m_z /= mag_norm
    
    # compute flux in the earth frame
    h_x = 2.0 * (m_x * (0.5 - SEq_3**2 - SEq_4**2) + m_y * (SEq_2 * SEq_3 - SEq_1 * SEq_4) + m_z * (SEq_2 * SEq_4 + SEq_1 * SEq_3))
    h_y = 2.0 * (m_x * (SEq_2 * SEq_3 + SEq_1 * SEq_4) + m_y * (0.5 - SEq_2**2 - SEq_4**2) + m_z * (SEq_3 * SEq_4 - SEq_1 * SEq_2))
    h_z = 2.0 * (m_x * (SEq_2 * SEq_4 - SEq_1 * SEq_3) + m_y * (SEq_3 * SEq_4 + SEq_1 * SEq_2) + m_z * (0.5 - SEq_2**2 - SEq_3**2))
    
    # normalise the flux vector to have only components in the x and z
    b_x = math.sqrt(h_x**2 + h_y**2)
    b_z = h_z

    # compute the objective function and Jacobian
    f_1 = 2.0 * SEq_2 * SEq_4 - 2.0 * SEq_1 * SEq_3 - a_x
    f_2 = 2.0 * SEq_1 * SEq_2 + SEq_3 * SEq_4 - a_y
    f_3 = 1.0 - 2.0 * SEq_2 * SEq_2 - 2.0 * SEq_3 * SEq_3 - a_z
    f_4 = 2.0 * b_x * (0.5 - SEq_3**2 - SEq_4**2) + 2.0 * b_z * (SEq_2 * SEq_4 - SEq_1 * SEq_3) - m_x
    f_5 = 2.0 * b_x * (SEq_2 * SEq_3 - SEq_1 * SEq_4) + 2.0 * b_z * (SEq_1 * SEq_2 + SEq_3 * SEq_4) - m_y
    f_6 = 2.0 * b_x * (SEq_1 * SEq_3 + SEq_2 * SEq_4) + 2.0 * b_z * (0.5 - SEq_2**2 - SEq_3**2) - m_z

    J_11or24 = 2.0 * SEq_3
    J_12or23 = 2.0 * SEq_4
    J_13or22 = 2.0 * SEq_1
    J_14or21 = 2.0 * SEq_2
    J_32 = 2.0 * J_14or21
    J_33 = 2.0 * J_11or24
    J_41 = 2.0 * b_z * SEq_3
    J_42 = 2.0 * b_z * SEq_4
    J_43 = 2.0 * 2.0 * b_x * SEq_3 + 2.0 * b_z * SEq_1
    J_44 = 2.0 * 2.0 * b_x * SEq_4 - 2.0 * b_z * SEq_2
    J_51 = 2.0 * b_x * SEq_4 - 2.0 * b_z * SEq_2
    J_52 = 2.0 * b_x * SEq_3 + 2.0 * b_z * SEq_1
    J_53 = 2.0 * b_x * SEq_2 + 2.0 * b_z * SEq_4
    J_54 = 2.0 * b_x * SEq_1 - 2.0 * b_z * SEq_3
    J_61 = 2.0 * b_x * SEq_3
    J_62 = 2.0 * b_x * SEq_4 - 2.0 * 2.0 * b_z * SEq_2
    J_63 = 2.0 * b_x * SEq_1 - 2.0 * 2.0 * b_z * SEq_3
    J_64 = 2.0 * b_x * SEq_2
    
    # compute the gradient (matrix multiplication)
    SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1 - J_41 * f_4 - J_51 * f_5 + J_61 * f_6
    SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3 + J_42 * f_4 + J_52 * f_5 + J_62 * f_6
    SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1 - J_43 * f_4 + J_53 * f_5 + J_63 * f_6
    SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2 - J_44 * f_4 - J_54 * f_5 + J_64 * f_6
    
    # normalise the gradient to estimate direction of the gyroscope error
    grad_norm = math.sqrt(SEqHatDot_1**2 + SEqHatDot_2**2 + SEqHatDot_3**2 + SEqHatDot_4**2)
    if grad_norm != 0:
        SEqHatDot_1 /= grad_norm
        SEqHatDot_2 /= grad_norm
        SEqHatDot_3 /= grad_norm
        SEqHatDot_4 /= grad_norm

    # compute angular estimated direction of the gyroscope error
    w_err_x = 2.0 * (SEq_1 * SEqHatDot_2 - SEq_2 * SEqHatDot_1 - SEq_3 * SEqHatDot_4 + SEq_4 * SEqHatDot_3)
    w_err_y = 2.0 * (SEq_1 * SEqHatDot_3 + SEq_2 * SEqHatDot_4 - SEq_3 * SEqHatDot_1 - SEq_4 * SEqHatDot_2)
    w_err_z = 2.0 * (SEq_1 * SEqHatDot_4 - SEq_2 * SEqHatDot_3 + SEq_3 * SEqHatDot_2 - SEq_4 * SEqHatDot_1)

    # compute and remove the gyroscope baises
    w_bx += w_err_x * deltat * zeta
    w_by += w_err_y * deltat * zeta
    w_bz += w_err_z * deltat * zeta
    w_x -= w_bx
    w_y -= w_by
    w_z -= w_bz

    # compute the quaternion rate measured by gyroscopes
    SEqDot_omega_1 = -0.5 * (SEq_2 * w_x + SEq_3 * w_y + SEq_4 * w_z)
    SEqDot_omega_2 = 0.5 * (SEq_1 * w_x + SEq_3 * w_z - SEq_4 * w_y)
    SEqDot_omega_3 = 0.5 * (SEq_1 * w_y - SEq_2 * w_z + SEq_4 * w_x)
    SEqDot_omega_4 = 0.5 * (SEq_1 * w_z + SEq_2 * w_y - SEq_3 * w_x)
    
    # compute then integrate the estimated quaternion rate
    SEq_1 += (SEqDot_omega_1 - beta * SEqHatDot_1) * deltat
    SEq_2 += (SEqDot_omega_2 - beta * SEqHatDot_2) * deltat
    SEq_3 += (SEqDot_omega_3 - beta * SEqHatDot_3) * deltat
    SEq_4 += (SEqDot_omega_4 - beta * SEqHatDot_4) * deltat

    # normalise quaternion
    quat_norm = math.sqrt(SEq_1**2 + SEq_2**2 + SEq_3**2 + SEq_4**2)
    SEq_1 /= quat_norm
    SEq_2 /= quat_norm
    SEq_3 /= quat_norm
    SEq_4 /= quat_norm
    quaternion = np.array([SEq_1, SEq_2, SEq_3, SEq_4])
    
    return quaternion

gyroscope = np.array([0.0, 0.0, 0.0])  # Angular rates in rad/s
accelerometer = np.array([-0.0011, 0.0068, 10.04])  # Acceleration in m/s^2
magnetometer = np.array([30.0, 60.0, 0.0])  # Magnetic field in µT

start = time.time()
quat = filterUpdate(accelerometer, gyroscope, magnetometer, init_quat=np.array([1, 0, 0, 0]), Ts=0.01)
roll, pitch, yaw = quaternion_to_euler(quat)
end = time.time()
rot_mat = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
rot_mat_frame = rot_mat.T
acc_world = np.matmul(rot_mat_frame.T, accelerometer) - [0, 0, 9.80665]

print("Madgwick Filter")
print("Acceleration wrt the world (no g):", acc_world)
print("Roll:", np.degrees(roll), "Pitch:", np.degrees(pitch), "Yaw:", np.degrees(yaw))
print("Quaternion:", quat)
print("Total time:", end-start, "seconds")
print("--------------------------------")