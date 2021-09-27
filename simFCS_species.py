import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
Simulation of fluorescence correlation spectroscopy data
The simulation mimics multiple fractions of freely diffusing molecules with individual diffusion constant(s) and molecular brightness.
The diffusion is simulated in a box of size L with periodic boundary conditions. The intensity is calculated using an optical transfer function, 
which accounts for the geometry of the confocal volume (kappa and Veff). Important parameters are the overal concentration (C=1000pM by default), 
the time step size (DT=1µs by default) and the number of time steps (N_STEPS=1e5 by default). For optimal correlation results, the concentration 
should be 1000pM and the trace length (N_STEPS*DT) should be 100 times bigger than the slowest diffusion time. In order to increase statistics, 
multiple iterations can be computed using N_ITER > 1.

27.09.2021 | Andreas Hartmann
"""

# parameters
#######################################################################################################
# experimental parameters
C = 1000 # (pM) molecule concentration -> 1000 pM is a reliable concentration for FCS
D = np.array([426, 200, 50]) # (µm^2/s) diffusion constant of each molecule species
FRAC = np.array([0.2, 0.3, 0.5]) # molecule fraction of each diffusion species -> Should sum to 1!!!
Q = np.array([30, 50, 60]) # (kHz) molecular brightness of each diffusion species

# setup parameters
V_EFF = 2.24 # (fL=µm^3) effective confocal volume
KAPPA = 5 # wz/wxy axis ratio of the confocal volume elipsoid

# simulation parameters
L = 5 # (µm) simulation box size
DT = 1 # (µs) time step size
N_STEPS = int(1e5)  # number of time steps
N_ITER = 1 # number of repeating iterations

NA = 6.02214076e23 # (mol^-1) avogadro constant
N_MOL = int(np.round(C*NA*1e-27*(L**3))) # total number of molecules in the box
D_MICRO = D*1e-6 # (µm^2/µs) diffusion constant of the molecule species
SIGMA = np.sqrt(2*D_MICRO*DT) # (µm) mean displacement per coordinate
#######################################################################################################

# optical transfer function (OTF)
V_conf = V_EFF*(0.5**1.5) # (fL)
wz = (V_conf*(KAPPA**2)/((np.pi/2)**1.5))**(1/3) # (µm) z-length at which the OTF decays by 1/exp(2) 
wxy = wz/KAPPA # (µm) x/y-length at which the OTF decays by 1/exp(2) 

OTF = lambda x,y,z: np.exp(-2*((x%L - L/2)/wxy)**2)*np.exp(-2*((y%L - L/2)/wxy)**2)*np.exp(-2*((z%L - L/2)/wz)**2)

# theoretical correlation function
lags = np.logspace(np.log10(DT/1000), np.log10(N_STEPS*DT/1000), num=200, endpoint=True, base=10.0, dtype=None, axis=0)
G_theo = lambda tau, tauD: 1/((1 + tau/tauD)*np.sqrt(1+tau/((KAPPA**2)*tauD)))/(V_EFF*1e-15*C*1e-12*NA)

print('Time trace length: ' + str(N_STEPS*DT/1000) + ' ms')

preQF = (Q**2)*FRAC/(np.sum(FRAC*Q)**2) # brightness weighted prefactors

for iterD in range(len(D)):

    tauD = (wxy**2)/(4*D_MICRO[iterD]*1000)
    print('tauD' + str(iterD) + '=' + str(tauD) + ' ms')
    
    if iterD == 0:
        
        G_all = preQF[iterD]*G_theo(lags, tauD)
    else:
        G_all = G_all + preQF[iterD]*G_theo(lags, tauD)

# drawing diffusion fraction
if len(D) > 1:

    idxSpecies = np.arange(0,len(D),1)
    boolFraction = np.dot(np.random.multinomial(1,FRAC, N_MOL), idxSpecies.T)

else:
    boolFraction = np.zeros(N_MOL, dtype=int)

# initialize molecule positions
x = L*np.random.rand(N_MOL)
y = L*np.random.rand(N_MOL)
z = L*np.random.rand(N_MOL)

tracks = {}
record = np.zeros(len(D))

for i in range(N_ITER):

    I = np.zeros(N_STEPS)

    for iter in range(N_MOL):
        
        # drawing displacements
        dx = np.multiply(np.random.randn(N_STEPS), SIGMA[boolFraction[iter]]) # (µm)
        dy = np.multiply(np.random.randn(N_STEPS), SIGMA[boolFraction[iter]]) # (µm)
        dz = np.multiply(np.random.randn(N_STEPS), SIGMA[boolFraction[iter]]) # (µm)

        # molecule coordinates
        xtrack = x[iter] + np.cumsum(dx)
        ytrack = y[iter] + np.cumsum(dy)
        ztrack = z[iter] + np.cumsum(dz)

        # calculation of the molecule intensity time trace and signal accumulation
        I = I + np.multiply(OTF(xtrack, ytrack, ztrack), Q[boolFraction[iter]])

        # recording 1 track for each diffusion species
        if  record[boolFraction[iter]] == 0:

            tracks['x' + str(boolFraction[iter])] = xtrack
            tracks['y' + str(boolFraction[iter])] = ytrack
            tracks['z' + str(boolFraction[iter])] = ztrack

            record[boolFraction[iter]] = 1

        # starting position for the next iteration
        x[iter] = xtrack[-1]%L
        y[iter] = ytrack[-1]%L
        z[iter] = ztrack[-1]%L    
    
    # calculation of the auto-correlation and correlation accumulation
    muI = np.mean(I)
    I_ctr = (I - muI)

    if i == 0:

        gII_lin = signal.correlate(I_ctr, I_ctr, mode='full', method='fft')/(muI**2)/N_STEPS

    else:
        
        gII_lin = gII_lin + signal.correlate(I_ctr, I_ctr, mode='full', method='fft')/(muI**2)/N_STEPS

    print("Progress: {p: 4.1f} %".format(p = (i + 1)/(N_ITER)*100))

# average correlation function with linear lag times
gII_lin /= N_ITER
lags_lin = signal.correlation_lags(len(I_ctr), len(I_ctr), mode='full')
gII_lin = gII_lin[lags_lin >= 0]
lags_lin = lags_lin[lags_lin >= 0]*DT/1000 # (ms)

# average correlation function with logarithmic lag times
lags = np.logspace(np.log10(DT/1000), np.log10(N_STEPS*DT/1000), num=200, endpoint=True, base=10.0, dtype=None, axis=0)
gII = np.interp(lags, lags_lin, gII_lin)

# angles for polar coordinates of the confocal volume
phi = np.linspace(0, 2*np.pi, 100)
psi = np.linspace(0, np.pi, 100)

x_ellip = wxy*np.outer(np.cos(phi), np.sin(psi)) + L/2
y_ellip = wxy*np.outer(np.sin(phi), np.sin(psi)) + L/2
z_ellip = wz*np.outer(np.ones_like(phi), np.cos(psi)) + L/2

# time for intensity time trace
tt = np.arange(1, N_STEPS+1)*DT/1000 # (ms)

# plot of simulation results
fig = plt.figure(figsize=(12, 9))

# 3D plot of the confocal volume
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot_surface(x_ellip, y_ellip, z_ellip, alpha=0.25, edgecolor='none')

arrC = {}
lgd=[]

for j in range(len(D)):

    lgd.append('$\itN_{' + str(j) + '}$=' + str(int(np.round(N_MOL*FRAC[j]))))

    ax.plot(x[boolFraction == j], y[boolFraction == j], z[boolFraction == j],'.')
    arrC['Color' + str(j)] = ax.lines[j].get_color()

ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_zlim(0, L)
ax.set_xlabel('$\itx$ (µm)')
ax.set_ylabel('$\ity$ (µm)')
ax.set_zlabel('$\itz$ (µm)')
ax.legend(lgd, loc = 2)

# plot of representative diffusion tracks
ax = fig.add_subplot(2, 2, 2)

lgd=[]

for j in range(len(D)):

    lgd.append('$\itD_{' + str(j) + '}$=' + str(D[j]) + ' µm$^2$/s')

    lenx = len(tracks['x' + str(j)])
    leny = len(tracks['y' + str(j)])
    lenz = len(tracks['z' + str(j)])

    mean_x = np.mean(tracks['x' + str(j)][1:np.min([lenx, 10000])]) 
    mean_y = np.mean(tracks['y' + str(j)][1:np.min([lenx, 10000])])

    ax.plot(tracks['x' + str(j)][1:np.min([lenx, 10000])] - mean_x, tracks['y' + str(j)][1:np.min([leny, 10000])] - mean_y)
    ax.lines[j].set_color(arrC['Color' + str(j)])

ax.set_xlabel('$\\Delta\itx$ (µm)')
ax.set_ylabel('$\\Delta\ity$ (µm)')
ax.axis('equal')
ax.legend(lgd, loc = 2)

# plot of intensity time trace
ax = fig.add_subplot(2, 2, 3)
ax.plot(tt[1:np.min([300000, N_STEPS])], I[1:np.min([300000, N_STEPS])])
ax.set_xlabel('$\itt$ (ms)')
ax.set_ylabel('$\itI$($\itt$) (kHz)')

# plot of the correlation signal
ax = fig.add_subplot(2, 2, 4)

lgd=[]

for j in range(len(D)):

    tauD = (wxy**2)/(4*D_MICRO[j]*1000) # (ms) diffusion time of species j   

    strD = '{p: 4.2f}'.format(p = tauD)
    lgd.append('$\\tau_{D' + str(j) + '}$=' + strD + ' ms')

    '$\\tau_D_ {p: 4.1f} %'.format(p = tauD)

    ax.semilogx(lags, G_theo(lags, tauD)/np.max(G_theo(lags, tauD))*np.max(G_all), '--', color=(1, 0.6, 0.6))
    ax.lines[j].set_color(arrC['Color' + str(j)])

lgd.append('Simulation')

# ax.semilogx(lags, G_all,'-', color=(1, 0.6, 0.6))
ax.semilogx(lags, gII/np.max(gII)*np.max(G_all),'.-k')
ax.set_xlim(1e-3, N_STEPS*DT/1000)
ax.set_xlabel('$\\tau$ (ms)')
ax.set_ylabel('$\itG\mathrm{(\\tau)}$')
ax.legend(lgd, loc = 1)

plt.show()
