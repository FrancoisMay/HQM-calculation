import control
import numpy as np
from math import *
from sympy import Symbol, Matrix, MatrixSymbol, Eq, simplify, Poly, det, latex, S, solve, cse
from scipy.optimize import fsolve,minimize_scalar
from IPython.display import display, Math
import matplotlib.pyplot as plt

#####################################################################################
# Compute the HQM of any bicycle defined by Matrices A & B
#####################################################################################

# Function definition for computing gains, using the same method as Moore in HumanControl
# 
# the five next functions are translated from HumanControl Moore's Matlab code 

def compute_inner_gains(A, B, omega_nm, zeta_nm, zeta_delta, zeta_phidot):
    """Returns the steer and roll rate gains given the state and input matrices
    of the bicycle, the neuromuscular block's natural frequency and damping
    ratio, and the desired closed loop damping ratio of the signature
    neuromuscular peak in each of the closed loops.

    Parameters
    ==========

    A : array_like, shape(4, 4)
        The state matrix for a linear Whipple bicycle model, where states are
        [roll angle, steer angle, roll angular rate, steer angular rate].
    B : array_like, shape(4, 2)
        The input matrix for a linear Whipple bicycle model, where
        the 2nd input is [steer torque].
    omega_nm : float
        The natural frequency of the neuromuscular model.
    zeta_nm : float
        The damping ratio of the neuromuscular model.
    zeta_delta : float
        The damping ratio of the desired closed loop pole.
    zeta_phidot : float
        The damping ratio of the desired closed loop pole.

    Returns
    =======
    k_delta : float
        The steer angle feedback gain.
    k_phi_dot : float
        The roll rate feedback gain.

    """

    knowns = [omega_nm, zeta_delta, zeta_phidot, zeta_nm, A[2,0],A[2,1],A[2,2],A[2,3],A[3,0],A[3,1],A[3,2],A[3,3],B[2,1],B[3,1]]
    
    # TODO : This guess comes from the known solution for the Benchmark bike
    # at 5 m/s, this should be made more robust.
    guess = np.asarray([-8.71798244e3, 1.58227411e3, 1.23258658e3, 5.47761600e1,
                      5.59477325e1, 1.28748836e3, 2.57663234e3, -8.43896178e3,
                      4.65520391e1, -5.21602768e-2, 1.39479436e1, 1.41766516e1])
    
    eval_zero(guess,knowns)
    

    res = fsolve(eval_zero, guess, fprime=eval_jac, args=(knowns))
    
    k_delta = res[8]
    k_phidot = res[9]

    return k_delta, k_phidot
    

def eval_sub_exprs(unknowns, knowns, sexprs):
    # one of the functions used in compute_inner_gain
    c0 = unknowns[0]
    c1 = unknowns[1]
    c2 = unknowns[2]
    c3 = unknowns[3]
    c4 = unknowns[4]
    c5 = unknowns[5]
    c6 = unknowns[6]
    c7 = unknowns[7]
    k_delta = unknowns[8]
    k_dot_phi = unknowns[9]
    omega_delta = unknowns[10]
    omega_dot_phi = unknowns[11]
    
    omega_nm = knowns[0]
    zeta_delta = knowns[1]
    zeta_dot_phi = knowns[2]
    zeta_nm = knowns[3]
    a_20 = knowns[4]
    a_21 = knowns[5]
    a_22 = knowns[6]
    a_23 = knowns[7]
    a_30 = knowns[8]
    a_31 = knowns[9]
    a_32 = knowns[10]
    a_33 = knowns[11]
    b_21 = knowns[12]
    b_31 = knowns[13]

    sexprs[0] = 2*omega_delta
    sexprs[1] = zeta_delta*sexprs[0]
    sexprs[2] = -sexprs[1]
    sexprs[3] = 2*omega_nm*zeta_nm
    sexprs[4] = -a_22 - a_33 + sexprs[3]
    sexprs[5] = omega_delta**2
    sexprs[6] = -sexprs[5]
    sexprs[7] = omega_nm**2
    sexprs[8] = a_22*a_33
    sexprs[9] = a_23*a_32
    sexprs[10] = -a_20 - a_22*sexprs[3] - a_31 - a_33*sexprs[3] - sexprs[9] + sexprs[7] + sexprs[8]
    sexprs[11] = a_20*a_33
    sexprs[12] = a_22*a_31
    sexprs[13] = a_21*a_32
    sexprs[14] = a_23*a_30
    sexprs[15] = a_22*sexprs[7]
    sexprs[16] = -a_20*sexprs[3] - a_31*sexprs[3] - a_33*sexprs[7] - sexprs[9]*sexprs[3] + sexprs[11] + sexprs[12] - sexprs[13] - sexprs[14] - sexprs[15] + sexprs[3]*sexprs[8]
    sexprs[17] = a_20*a_31
    sexprs[18] = a_21*a_30
    sexprs[19] = a_20*sexprs[7]
    sexprs[20] = b_31*sexprs[7]
    sexprs[21] = k_delta*sexprs[20]
    sexprs[22] = -a_31*sexprs[7] - sexprs[9]*sexprs[7] + sexprs[11]*sexprs[3] + sexprs[12]*sexprs[3] - sexprs[13]*sexprs[3] - sexprs[14]*sexprs[3] + sexprs[17] - sexprs[18] - sexprs[19] + sexprs[21] + sexprs[7]*sexprs[8]
    sexprs[23] = b_21*sexprs[7]
    sexprs[24] = a_32*sexprs[23]
    sexprs[25] = b_31*sexprs[15]
    sexprs[26] = k_delta*sexprs[24] - k_delta*sexprs[25] + sexprs[11]*sexprs[7] + sexprs[12]*sexprs[7] - sexprs[13]*sexprs[7] - sexprs[14]*sexprs[7] + sexprs[17]*sexprs[3] - sexprs[18]*sexprs[3]
    sexprs[27] = a_30*sexprs[23]
    sexprs[28] = b_31*sexprs[19]
    sexprs[29] = k_delta*sexprs[27] - k_delta*sexprs[28] + sexprs[17]*sexprs[7] - sexprs[18]*sexprs[7]
    sexprs[30] = 2*omega_dot_phi
    sexprs[31] = zeta_dot_phi*sexprs[30]
    sexprs[32] = -sexprs[31]
    sexprs[33] = omega_dot_phi**2
    sexprs[34] = -sexprs[33]
    sexprs[35] = k_delta*sexprs[23]
    sexprs[36] = a_23*sexprs[21]
    sexprs[37] = a_33*b_21*sexprs[7]
    sexprs[38] = k_delta*sexprs[37]
    sexprs[39] = a_21*sexprs[21]
    sexprs[40] = a_31*b_21*sexprs[7]
    sexprs[41] = k_delta*sexprs[40]
    sexprs[42] = 2*zeta_delta
    sexprs[43] = sexprs[24] - sexprs[25]
    sexprs[44] = sexprs[27] - sexprs[28]
    sexprs[45] = 2*zeta_dot_phi
    sexprs[46] = b_31*k_dot_phi*sexprs[7]
    return sexprs

def eval_zero(unknowns, knowns):
    # one of the functions used in compute_inner_gain
    
    c0 = unknowns[0]
    c1 = unknowns[1]
    c2 = unknowns[2]
    c3 = unknowns[3]
    c4 = unknowns[4]
    c5 = unknowns[5]
    c6 = unknowns[6]
    c7 = unknowns[7]
    k_delta = unknowns[8]
    k_dot_phi = unknowns[9]
    omega_delta = unknowns[10]
    omega_dot_phi = unknowns[11]
    
    omega_nm = knowns[0]
    zeta_delta = knowns[1]
    zeta_dot_phi = knowns[2]
    zeta_nm = knowns[3]
    a_20 = knowns[4]
    a_21 = knowns[5]
    a_22 = knowns[6]
    a_23 = knowns[7]
    a_30 = knowns[8]
    a_31 = knowns[9]
    a_32 = knowns[10]
    a_33 = knowns[11]
    b_21 = knowns[12]
    b_31 = knowns[13]

    sexprs = np.zeros(47)
    sexprs = eval_sub_exprs(unknowns, knowns, sexprs)

    zero = [-c3 + sexprs[2] + sexprs[4],
            -c2 - c3*sexprs[1] + sexprs[10] + sexprs[6],
            -c1 - c2*sexprs[1] - c3*sexprs[5] + sexprs[16],
            -c0 - c1*sexprs[1] - c2*sexprs[5] + sexprs[22],
            -c0*sexprs[1] - c1*sexprs[5] + sexprs[26],
            -c0*sexprs[5] + sexprs[29],
            -c4 + sexprs[32] + sexprs[4],
            -c4*sexprs[31] - c5 + sexprs[10] + sexprs[34],
            -c4*sexprs[33] - c5*sexprs[31] - c6 + k_dot_phi*sexprs[35] + sexprs[16],
            -c5*sexprs[33] - c6*sexprs[31] - c7 + k_dot_phi*sexprs[36] - k_dot_phi*sexprs[38] + sexprs[22],
            -c6*sexprs[33] - c7*sexprs[31] + k_dot_phi*sexprs[39] - k_dot_phi*sexprs[41] + sexprs[26],
            -c7*sexprs[33] + sexprs[29]]

    return zero

def eval_jac(unknowns, knowns):
    # one of the functions used in compute_inner_gain

    c0 = unknowns[0]
    c1 = unknowns[1]
    c2 = unknowns[2]
    c3 = unknowns[3]
    c4 = unknowns[4]
    c5 = unknowns[5]
    c6 = unknowns[6]
    c7 = unknowns[7]
    k_delta = unknowns[8]
    k_dot_phi = unknowns[9]
    omega_delta = unknowns[10]
    omega_dot_phi = unknowns[11]
    
    omega_nm = knowns[0]
    zeta_delta = knowns[1]
    zeta_dot_phi = knowns[2]
    zeta_nm = knowns[3]
    a_20 = knowns[4]
    a_21 = knowns[5]
    a_22 = knowns[6]
    a_23 = knowns[7]
    a_30 = knowns[8]
    a_31 = knowns[9]
    a_32 = knowns[10]
    a_33 = knowns[11]
    b_21 = knowns[12]
    b_31 = knowns[13]

    sexprs = np.zeros(47)
    sexprs = eval_sub_exprs(unknowns, knowns, sexprs)

    zero_jac = [[0, 0, 0,-1,0,0,0,0, 0, 0 , -sexprs[42]  ,  0],
                [0,0,-1,sexprs[2], 0 ,0, 0,0, 0, 0,-c3*sexprs[42] - sexprs[0],0],
                [0,-1, sexprs[2], sexprs[6],0, 0 , 0 ,0, 0,0, -c2*sexprs[42] - c3*sexprs[0] ,0],
                [-1, sexprs[2], sexprs[6],0,0,0,0, 0 ,sexprs[20],0, -c1*sexprs[42] - c2*sexprs[0] ,0],
                [sexprs[2], sexprs[6],0, 0,0, 0 ,0,0, sexprs[43],0, -c0*sexprs[42] - c1*sexprs[0], 0],
                [sexprs[6],0, 0, 0, 0,0, 0,0 ,sexprs[44],0,-c0*sexprs[0],0],
                [0,0,0, 0,-1, 0,0, 0, 0 , 0,0,-sexprs[45]],
                [0,0,0, 0, sexprs[32] ,-1,  0, 0, 0, 0,0 ,-c4*sexprs[45] - sexprs[30]],
                [0,0,0, 0, sexprs[34], sexprs[32],-1, 0, k_dot_phi*sexprs[23],sexprs[35],0, -c4*sexprs[30] - c5*sexprs[45]],
                [0,0,0, 0,0, sexprs[34], sexprs[32], -1, a_23*sexprs[46] - k_dot_phi*sexprs[37] + sexprs[20], sexprs[36] - sexprs[38],0, -c5*sexprs[30] - c6*sexprs[45]],
                [0,0,0,0,0,0, sexprs[34], sexprs[32],a_21*sexprs[46] - k_dot_phi*sexprs[40] + sexprs[43], sexprs[39] - sexprs[41], 0, -c6*sexprs[30] - c7*sexprs[45]],
                [0,0,0,0, 0,0,0, sexprs[34], sexprs[44],0, 0,-c7*sexprs[30]]]
    return zero_jac


def phi_loop_fun(A, B, omega_nm, k_delta, k_phi_dot, zeta_nm,plot):
    """ 
    Returns the gain needed to set the crossover frequency of the roll loop at a desired value.

    Parameters
    ==========

    A : array_like, shape(4, 4)
        The state matrix for a linear Whipple bicycle model, where states are
        [roll angle, steer angle, roll angular rate, steer angular rate].
    B : array_like, shape(4, 2)
        The input matrix for a linear Whipple bicycle model, where
        the 2nd input is [steer torque].
    omega_nm : float
        The natural frequency of the neuromuscular model.
    zeta_nm : float
        The damping ratio of the neuromuscular model.
    k_delta : float 
        The steer angle feedback gain.
    k_phi_dot : float
        the roll rate angle feedback gain.

    Returns
    =======
    k_phi : float
        The roll angle open loop gain.
    """
    
    k_phi = 1
    w = np.linspace(1e-1,20,100)
    A_phi = np.hstack((A,np.zeros((4,2))))
    A_phi = np.vstack((A_phi,np.array([[0,0,0,0,0,1],[-omega_nm**2*k_delta*k_phi_dot*k_phi , -omega_nm**2*k_delta , -omega_nm**2*k_delta*k_phi_dot , 0 , -omega_nm**2 , -2*omega_nm*zeta_nm]])))
    A_phi[2,4] = B[2,1]
    A_phi[3,4] = B[3,1]
    B_phi =  [0,0,0,0,0,omega_nm**2*k_delta*k_phi_dot*k_phi]
    C_phi = np.zeros((1,6))
    C_phi = C_phi[0]
    C_phi[0] = 1
    D_phi = [0] 
    FTBF_phi = control.ss2tf(A_phi,B_phi,C_phi,D_phi)
    FTBO_phi = FTBF_phi/(1-FTBF_phi)
    mag_phi,phase,w = control.bode(FTBO_phi,w,dB=1,plot=0)
    mag_phi_dB = 20*np.log(mag_phi)/np.log(10)
    #index = np.argmin(np.abs(w-2))
    MagC0 = np.interp(2,w,mag_phi)
    k_phi = 1/MagC0
    
    if plot == 1 : 
        ax=plt.figure().add_subplot()
        plt.plot(w,mag_phi_dB,label='phi')
        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.set_xlim(np.min(w),np.max(w))
        ax.set_ylim(-45,20)
        ax.grid()
    return k_phi

##############
# Matrices A and B from the Whipple model
##############
bicycle = 'Benchmark'
A = np.array([[  0.0,          0.0,          1.0,          0.0,       ],
     [  0.0,          0.0,          0.0,          1.0,       ],
     [  9.48977445, -22.85146663,  -0.52761225,  -1.65257699],
     [ 11.71947687, -18.38412373,  18.38402617, -15.42432764]])
B = np.array([[0.0, 0.0        ],
     [0.0, 0.0        ],
     [0.0, -0.12409203],
     [0.0,  4.32384018]])

##############
# Parameter values of the open and closed loops 
##############
omega_nm = 30 # rad/s
zeta_nm = 0.707
omega_phi_dot = 0.1289
zeta_phi_dot = 0.0855

# Compute the roll rate angle and steer angle gains
k_delta,k_phi_dot = compute_inner_gains(A,B,omega_nm,zeta_nm,omega_phi_dot,zeta_phi_dot)

# Compute the roll angle gain
k_phi = phi_loop_fun(A, B, omega_nm, k_delta, k_phi_dot, zeta_nm,plot=1)

print("________________________________________________________________________ \n")
print("Gains computed using the same method as in Moore's HumanControl Matlab codes\n")
print(bicycle + " bicycle : ")
print("Kdelta = ",k_delta)
print("Kphidot = ",k_phi_dot)
print("Kphi = ",k_phi)
print("________________________________________________________________________ \n")

#############
# Compute the HQM from the gicen parameters
#############
A_hqm = np.hstack((A,np.zeros((4,2))))
A_hqm = np.vstack((A_hqm,np.array([[0,0,0,0,0,1],[-omega_nm**2*k_delta*k_phi_dot*k_phi , -omega_nm**2*k_delta , -omega_nm**2*k_delta*k_phi_dot , 0 , -omega_nm**2 , -2*omega_nm*zeta_nm]])))
A_hqm[2,4] = B[2,1]
A_hqm[3,4] = B[3,1]
B_hqm =  [0,0,0,0,0,omega_nm**2*k_phi*k_delta*k_phi_dot]
C_hqm = np.zeros((1,6))
C_hqm = C_hqm[0]
C_hqm[2] = -1 / k_phi
C_hqm[1] = -1 / k_phi * 1 / k_phi_dot
D_hqm = [0] 
hqm = control.ss2tf(A_hqm,B_hqm,C_hqm,D_hqm)
sys_filter = control.tf([20**2],[1,2*20,20**2])
hqm = hqm*sys_filter
mag, phase, omega = control.bode(hqm,np.linspace(0,40,250),dB=0,plot = 0)
plt.close(('all'))
plt.plot(omega,mag,label = 'HQM for the '+ bicycle + ' bicycle')
plt.legend()
plt.grid()
plt.xlabel('Omega (rad/s)')
plt.ylabel('HQM')

