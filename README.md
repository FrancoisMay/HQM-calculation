# HQM-calculation
Compute the HQM of the bicycle from the matrices A and B constructed from the Whipple bicycle model

The matrices A end b are given in the beginnig of the code

This code does two things : 
  - First compute the steer and roll rate closed loop gains from the characteristics defined in [1], then compute the roll open loop gain using the characteristic given by [1].
  - Then compute the HQM associated to this model with the gains defined precedently

Finally plot the HQM with on a linear scale.

[1] : Moore, Jason & Hubbard, Mont & Hess, Ronald. (2016). An Optimal Handling Bicycle. 10.6084/M9.FIGSHARE.3806310.V1. 
