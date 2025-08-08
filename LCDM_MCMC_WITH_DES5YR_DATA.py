#Python code for mensure best fits of cosmological parameters with MCMC using emcee bib

import numpy as np
import pandas as pd #For read data 
import emcee  #For maximization of likelihood
import tqdm   #for progress bar 
import scipy.integrate as si #For numerical integration
import matplotlib.pyplot as plt  #For plots
from sklearn.linear_model import LinearRegression  #For linear regression
from sklearn.metrics import r2_score  #For R² of linear regression
from getdist import plots, MCSamples #For corner plots


#Let's load the data of DES  

data = pd.read_csv('path/DES-SN5YR_HD+MetaData.csv', sep=',', header = 0)
display(data)


z = data['zCMB']
mu = data['MU']
mu_err = data['MUERR_FINAL']
mb = data['mB']
mb_err = data['mBERR']
x1 = data['x1']
c = data['c']

# Convert z and mu to numpy arrays
x = np.log10(z.values).reshape(-1, 1)  # log(z) as input feature for the model
y = mu.values 

# Fit linear regression model
model = LinearRegression()
model.fit(x, y)

a = model.coef_[0]
b = model.intercept_
print(f"Slope (a): {a:.3f}")
print(f"Intercept (b): {b:.3f}")

# Generate fitted curve points
z_fit = np.linspace(min(z), max(z), 200)
x_fit = np.log10(z_fit).reshape(-1, 1)
y_fit = model.predict(x_fit)

# Calculate R² correctly using the original data
y_pred_train = model.predict(x)  # prediction on training data points
r2 = r2_score(y, y_pred_train)
print(f"R²: {r2:.4f}")

# Plotting
plt.figure(figsize=(8, 6), dpi=120)
plt.scatter(z, mu, color='green', s=10, alpha=0.6, label='DES5Yr data')
plt.plot(z_fit, y_fit, color='black', label=f'Linear Regression (R²={r2:.3f})')
plt.xscale('log')
plt.xlabel('Redshift (log scale)', fontdict={'family': 'serif', 'size': 12, 'weight': 'normal'})
plt.ylabel('Distance modulus μ', fontdict={'family': 'serif', 'size': 12, 'weight': 'normal'})
plt.legend()
plt.show()


################################################################################## MCMC Analysis #############################################################


#Let's load the systematic erros   

data_sys_errors = np.loadtxt('path/covsys_000.txt', skiprows=1) #load txt and skip the first term 

with open('path/covsys_000.txt') as f: n = int(f.readline().strip())  #read the first term of txt and save as 'n'

sys_errors = data_sys_errors.reshape(n, n) #construct the matrix of n x n for sys errors 

#print(sys_errors)



#To make the covariance matrix we have to sum the systematic errors with statistical errors



# First we have to take a column of statistical errors in data

est_errors = data['MUERR_FINAL'].to_numpy() #Transform in array

est_errors = np.diag(est_errors**2) #Transform the array in diagonal matrix

#print(est_errors)


#The covariance matrix

Cij = sys_errors + est_errors


#Inverse with linear algebra inverse of numpy

InvCovTot = np.linalg.inv(Cij)


print(InvCovTot)





zcmb = data['zCMB']
mu = data['MU']

velc = 299792.4580  # speed of light in [km/s] 

# Hubble parameter of the cosmological model
def H(z, om, H0):
    Ez = np.sqrt(om * (1.0 + z)**3 + (1.0 - om))
    return H0 * Ez

# Luminosity distance 
def dL(z, om, H0):

    def integrand(zz, om, H0):
        return 1.0 / H(zz, om, H0)
    
    integral = si.quad(integrand, 0, z, args=(om, H0))[0]
    return (1.0 + z) * velc * integral

# Distance modulus 
def muth(z, om, H0):
    return 5 * np.log10(dL(z, om, H0)) + 25

# Log-likelihood including color and stretch coefficients
def log_likelihood_sn_ab(om, H0, α, β, M):

    def mbth(z, om, H0, α, β, M, x11, cc):
        return muth(z, om, H0) + M - α * x11 + β * cc

    chi2 = sum(((mb[i] - mbth(zcmb[i], om, H0, α, β, M, x1[i], c[i])) / mb_err[i])**2 
               for i in range(len(zcmb)))

    return -0.5 * chi2

# Log-likelihood using only the distance modulus
def log_likelihood_sn(om, H0):
    mu_vec = np.vectorize(lambda z: muth(z, om, H0))(zcmb)
    diff = mu - mu_vec
    chi2 = np.dot(diff, InvCovTot @ diff)
    return -0.5 * chi2





# Priors
def log_prior(om, H0):
    if not 0.1 < om < 0.5:
        return -np.inf
    if not 60.0 < H0 < 80.0:
        return -np.inf
#    if not -1.0 < α < 1.0:
#        return -np.inf
#    if not 0.0 < β < 5.0:
#        return -np.inf
#    if not -25.0 < M < -10.0:
#        return -np.inf
    
    return 0

# Total log-likelihood (posterior)
def log_probability(listparams):

    # om, H0, α, β, M = listparams
    om, H0 = listparams

    lp = log_prior(om, H0) 
    if not np.isfinite(lp):  # if lp is not finite, reject parameters
        return -np.inf       
    lpprior = lp            

    log_probability_tot = log_likelihood_sn(om, H0)

    return log_probability_tot

# Initial guess for our parameters
# om_ini, H0_ini, α_ini, β_ini, M_ini = 0.3, 70.0, 0.0, 1.0, -19.0
om_ini, H0_ini = 0.3, 70.0

# List of initial parameter values
# param_ini = [om_ini, H0_ini, α_ini, β_ini, M_ini]
param_ini = [om_ini, H0_ini]


########################################################################### Run MCMC with emcee ###############################################################

dim = len(param_ini) 

pos = param_ini + 1e-4 * np.random.randn(10,dim)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

totalsteps = 100000
coords = pos
# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(totalsteps)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(coords, iterations=totalsteps, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau



########################################################## Plots and Results ##############################################################################


fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()

#labels = [r"\Omega_m", "H_0", "α", "β", "M"]
labels = [r"\Omega_m", "H_0"]

ndim = len(labels)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")



import arviz as az

# This code belongs to the MCMC convergence diagnostic step after sampling with emcee.

# Convert the emcee sampler object into an ArviZ InferenceData object.
# ArviZ provides tools for analyzing and visualizing Bayesian inference results.
data = az.from_emcee(sampler)

# Compute and print the R-hat statistic (Gelman-Rubin diagnostic) for each parameter.

# R-hat measures the convergence of MCMC chains:
# Values close to 1 indicate good convergence (chains mixing well).
# Values significantly greater than 1 indicate poor convergence (chains may not have mixed).

print(az.rhat(data))




tau = sampler.get_autocorr_time()
ndim = dim

burnin = int(3 * np.max(tau))  # discarding 3 × tau from the initial steps
thin = int(0.5 * np.min(tau))  # thinning to about half of tau

samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)

print("Autocorrelation time: {0}".format(tau))
print("Burn-in: {0}".format(burnin))
print("Thin: {0}".format(thin))
print("Flat chain shape: {0}".format(samples.shape))
print("Flat log prob shape: {0}".format(log_prob_samples.shape))




np.savetxt("MCMC_param_FLCDM.txt", np.array(samples))
np.savetxt("MCMC_log_prob_FLCDM.txt", np.array(log_prob_samples))



# Reading MCMC output files - Flat-LCDM model
MCMC_full = pd.read_fwf("MCMC_param_FLCDM.txt", sep=' ', names=['om', 'H0'])
MCMC_log_prob_full = pd.read_fwf("MCMC_log_prob_FLCDM.txt", sep=' ', names=['log_prob'])


############################################################## Best Fits of parameters ######################################################################

from IPython.display import display, Math

sample = np.array(MCMC_full)
labels = [r"\Omega_m", "H0"]
ndim = sample.shape[1]

print("Mean and statistical confidence interval:")
alpha = (100 - 68.3) / 2  # 1 sigma confidence level
# alpha = (100 - 95.4) / 2  # 2 sigma confidence level
# alpha = (100 - 99.7) / 2  # 3 sigma confidence level

MeanParam = []
for i in range(ndim):
    mcmc1s = np.percentile(sample[:, i], [alpha, 50, 100 - alpha])  # percentiles for 1-sigma interval (68.3%)
    q1s = np.diff(mcmc1s)
    
    txt = r"\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
    txt = txt.format(mcmc1s[1], q1s[0], q1s[1], labels[i])
    
    MeanParam.append(np.around(mcmc1s[1], 3))
    display(Math(txt))

print("Best fit and minimum chi-square:")

# If MCMC_log_prob_full is not a DataFrame, convert it
if not isinstance(MCMC_log_prob_full, pd.DataFrame):
    MCMC_log_prob_full = pd.DataFrame(MCMC_log_prob_full, columns=["log_prob"])

list_chi2_df = -2 * MCMC_log_prob_full  # chi-squared values

indxmin = list_chi2_df["log_prob"].idxmin()  # index of minimum chi²
display(MCMC_full.loc[indxmin])  # parameters corresponding to minimum chi²
print("chi2min = ", np.around(list_chi2_df.loc[indxmin, 'log_prob'], 3))




################################################################### Corner Plot ######################################################

#names = ['x0','x1','x2','x3','x4']
#labels = [r"\Omega_m", "H_0", "α", "β", "M"]

names = ['x0','x1']
labels = [r"\Omega_m", "H_0"]

# Creating MCSamples object with parameter names, labels and ranges
sample_FXCDM = MCSamples(
    samples=np.array(MCMC_full), 
    names=names, 
    labels=labels,
    label='Full',
    #ranges={'x0':[0.20,0.5],'x1':[60,80], 'x2':[0.0,5.0], 'x3':[0.0,1.0], 'x4':[-30.0,-10.0]}
    ranges={'x0':[0.20,0.5],'x1':[60,80]}
)


## Plot settings #######
g = plots.get_subplot_plotter(width_inch=6)
g.settings.title_limit_fontsize = 12
g.settings.figure_legend_frame = True
# g.settings.alpha_filled_add = 0.7
g.settings.axis_tick_x_rotation = 45
g.settings.axis_tick_y_rotation = 45
g.settings.axis_tick_max_labels = 20




# Plot the triangle plot (corner plot)
g.triangle_plot(
    [sample_FXCDM], ['x0', 'x1'], 
    shaded=False,
    filled=[True], 
    line_args=[{'lw':1.5, 'color': "Green"}],
    contour_lws=[1.5], 
    contour_ls=['-'],
    contour_colors=["Green"], 
    alphas=[1.0],  
    legend_labels=["DES5YR Sn"], 
    legend_loc=[0.72, 0.84],)
