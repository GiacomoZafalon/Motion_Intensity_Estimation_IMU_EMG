# Master's Thesis on Motion Intensity Estimation using IMUs and sEMG

## Introduction

My Master's Thesis will be conducted at **Ercura Industrial Exoskeletons** from February to July 2024. Ercura focuses on designing and realizing upper limb exoskeletons primarily targeting workers engaged in overhead activities and heavy load handling.

## Background

Numerous studies have addressed shoulder injuries during physical work, emphasizing a moderate relationship between arm elevation and shoulder injury, particularly for angles exceeding 90°. Other factors contributing to injuries include handling tools or heavy loads, working with handheld vibrating tools, and repetitive movements. Lifting and carrying heavy objects, coupled with incorrect posture, are considered actions with the highest risk of muscle activation and subsequent injuries. Currently, Ercura's exoskeleton provides passive assistance using an air piston to push the arms up and maintain position. The aim is to enhance it to a semi-active state, offering varied assistance based on the user's task.

## Objectives

The main objective of my Master's Thesis is to determine the effort required for specific tasks, such as lifting objects, and provide dynamic assistance using the exoskeleton. I plan to employ surface electromyogram (sEMG) electrodes and inertial measurement units (IMUs) in conjunction with a neural network comprising a convolutional neural network (CNN) and long short-term memory (LSTM) cells. This approach utilizes the sEMG signal as ground truth, while IMU data serves as network input. CNN and LSTM are preferred for their superior performance over other deep learning methods.

## Methodology

### Muscle Activation Estimation
- **Sensor Placement**: Electrodes will be placed on deltoid muscles and the bicep to retrieve upper limb muscle activation during lifting.
- **Signal Preprocessing**: This includes band-pass filtering to remove noise, notch filtering to eliminate powerline interferences, rectification for amplitude normalization, and subsequent smoothing via root mean square (RMS) calculations.
- **Normalization**: The signal will be normalized over maximum voluntary contraction (MVC) to obtain a label between 0 and 1 for the neural network.

### IMU Data Processing
- **Sensor Placement**: IMUs will be placed on the lower back, shoulder, upper arm, and lower arm to assess movement dynamics and kinematics.
- **Orientation Estimation**: Initial orientation will be obtained in a static position using gravity and Earth's magnetic field, followed by continuous updates using the Madgwick Filter.
- **Feature Extraction**: Angular data at shoulder and elbow joints, joint angular rate, and vertical velocity during lifting will be extracted and provided as input to the neural network.

## Conclusion

Upon obtaining estimated effort from the neural network, parameters of the exoskeleton, such as the point of force application of the piston, will be defined to provide appropriate assistance, optimizing power consumption and operator safety.
