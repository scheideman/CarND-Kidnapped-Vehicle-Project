# Overview

This project implements a 2 dimensional particle filter in C++. Your particle filter is given a map and some initial localization information (analogous to what a GPS would provide). At each time step the filter will gets observation and control data. 

# Implementation

A particle filter is a Monte Carlo localization method that represents the posterior distribution as set of particles, each estimating the state of the robot. During the prediction step noise is added to the motion model to take into account uncertainty. Then in the update step a weight is calculated for each particle that represents the likelihood that the last set of measurements matches the particle. After that the particles are resampled with replacement with probability proportional to their weight. So over time unlikely particles are culled and more likely particles stick around. A benefit of this method is its easy to implement and it can represent multimodal posteriors.  

## Running the Code
This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and intall uWebSocketIO for either Linux or Mac systems. For windows you can use either Docker, VMware, or even Windows 10 Bash on Ubuntu to install uWebSocketIO.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

mkdir build
cd build
cmake ..
make
./particle_filter


#### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

### All other data the simulator provides, such as observations and controls.

> * Map data provided by 3D Mapping Solutions GmbH.
