import numpy as np
import pandas as pd
import scipy as sc
import scipy.io as io
from scipy.stats import multivariate_normal
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsc
from jax import grad, jit, vmap, random, lax, value_and_grad, tree_multimap, tree_map

from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
 

@jit
def AlpEnvelope(Xarr, TRise, TauDiff, Lag):
  TRise = TRise ** 2.0
  TDecay = TRise + TauDiff ** 2 + 1e-8
  TMax = (jnp.log(TRise / TDecay) * TDecay * TRise) / (TRise - TDecay)
  Max =  jnp.exp(- TMax / TDecay) - jnp.exp(- TMax / TRise)
  Delayed = Xarr - Lag
  NewTime = jnp.where(Delayed < 0.0, 1000, Delayed)
  return (jnp.exp(- NewTime / TDecay) - jnp.exp(- NewTime / TRise)) / Max

@jit
def Squared_exp(I, J, Sigma_f, Ell):
  return Sigma_f**2.0*jnp.exp(-(I-J)**2/(2.0*Ell**2))

key = random.PRNGKey(1)
key, *subkeys = random.split(key, 4)

data = pd.read_hdf('./Data/finger_posdata.h5')
k = 10000
num_filt = 130
xtest = data.spikes.to_numpy()[:k, :num_filt]
print(xtest.shape)
data.index = data.index - data.index[0]
ytime = (data.index / np.timedelta64(1, 's')).to_numpy()[:k]
ycoord = data.finger_pos.y.to_numpy()[:k]
xcoord = data.finger_pos.x.to_numpy()[:k]
k, num_filt = np.shape(xtest)
k -= 1
ytest = ycoord.reshape(k + 1, 1)
xtest = xtest.reshape(num_filt, k + 1, 1)
data.spikes

ytest -= ytest[0, 0]

subkeyf = subkeys[-2]

# Time bin size
time_bin = ytime[1] - ytime[0]

# Filter horizon
f_maxt = 1.0

# Maximum index of filter data
n = np.floor(f_maxt/time_bin).astype(np.int64)
ftime = np.linspace(0.0, f_maxt, n + 1).reshape((n + 1, 1))


xpad = np.hstack((xtest, np.zeros((num_filt, n, 1))))
x_fft = np.fft.rfft(xpad, axis=1)

@jit
def KL_Term(L_Matrix, V_Vector, Num_Filt, M):
  return  0.5 * (- jnp.sum(jnp.log(jnp.diagonal(L_Matrix, axis1 = 1, axis2 = 2) ** 2)) + \
                jnp.sum(L_Matrix ** 2) + jnp.sum(V_Vector ** 2) - Num_Filt * M)

@jit
def Solver(Kmm, Diff):
  return jsc.linalg.solve(Kmm, Diff, sym_pos=True, check_finite=True)

V_Solver = vmap(Solver, in_axes = [None, 0])

@partial(jit, static_argnums = range(9,15))
def Expected_Term(Sigma_f, Ell, Sigma_n, Z_Vector, V_Vector, L_Matrix, TRise, TauDiff, Lag, Number_F, Num_Base, M, Number_Filt, N, K, Y_Test, X_Fft, F_Time, Subkeys):

  Sigma_f_d = Sigma_f[:, :, None]
  Ell_f_d = Ell[:, :, None]
  TRise_d = TRise[:, :, None]
  TauDiff_d = TauDiff[:, :, None]

  Thetas = random.normal(Subkeys[0], (Number_F, Number_Filt, 1, Num_Base)) * (1.0 / Ell_f_d)

  Taus = random.uniform(Subkeys[1], (Number_F, Number_Filt, 1, Num_Base)) * 2.0 * jnp.pi

  Omegas = random.normal(Subkeys[2], (Number_F, Number_Filt, Num_Base, 1))

  Constant = (Sigma_f_d * jnp.sqrt(2.0 / Num_Base))

  ZT = Z_Vector.transpose(0, 2, 1)
 
  Phi1 = Constant * jnp.cos(F_Time * Thetas + Taus)
  Phi2 = Constant * jnp.cos(Z_Vector * Thetas + Taus)

  Kmm = Squared_exp(Z_Vector, ZT, Sigma_f_d, Ell_f_d)
  Knm = Squared_exp(F_Time, ZT, Sigma_f_d, Ell_f_d)

  C = jnp.linalg.cholesky(Kmm + jnp.eye(M) * 1e-6)

  V_u = C @ L_Matrix @ L_Matrix.transpose(0, 2, 1) @ C.transpose(0, 2, 1)

  Mu_u = C @ V_Vector

  V_uChol = jnp.linalg.cholesky(V_u + 1e-6 * jnp.eye(M))
  U_Samples = Mu_u + V_uChol @ random.normal(Subkeys[3], (Number_F, Number_Filt, M, 1))

  Vu = V_Solver(Kmm + 1e-6 * jnp.eye(M), U_Samples - Phi2 @ Omegas)

  F_Samples = (Phi1 @ Omegas + Knm @ Vu) * AlpEnvelope(F_Time, TRise_d, TauDiff_d, Lag)

  F_Samples = jnp.dstack((F_Samples, jnp.zeros((Number_F, Number_Filt, K, 1))))
  
  F_Fft = jnp.fft.rfft(F_Samples, axis = -2)

  Filter_Out = jnp.fft.irfft(F_Fft * X_Fft, N + K + 1, axis = -2)[:, :, : K + 1]

  Pred = jnp.sum(Filter_Out, axis = 1)
  
  Likelihoods = - 0.5 * ((K + 1) * jnp.log(2 * jnp.pi * Sigma_n ** 2) + \
                    jnp.sum((Y_Test - Pred)**2, axis = 1)/(Sigma_n ** 2))
  
  return jnp.mean(Likelihoods)

test_number_f = 150
test_num_base = 100
test_m = int(1000)
test_key = random.PRNGKey(0)
test_key, *test_subkeys = random.split(test_key, 5)
test_sigma_f = 5.0 * np.ones((num_filt, 1))# np.arange(num_filt).reshape(1, num_filt, 1, 1)
test_ell_f = 2.0 * np.ones((num_filt, 1))
test_sigma_n = 0.05
test_z_m = np.tile(np.linspace(0.0, ftime[-1], test_m).reshape((test_m, 1)), (num_filt, 1, 1))
test_v_m = np.zeros(test_m * num_filt).reshape((num_filt, test_m, 1))
test_l_mm = np.tile(np.eye(test_m), (num_filt, 1, 1))
test_trise = 0.02 * np.ones((num_filt, 1))
test_tdiff = 0.03 * np.ones((num_filt, 1))
test_lag = 0.01 * np.ones((num_filt, 1))
test_lag = test_lag[:,:, None]
xtest = xtest.reshape(1, (k + 1) * num_filt, 1)
ftime = ftime.reshape(n + 1, 1)

samps = Expected_Term(test_sigma_f, test_ell_f, test_sigma_n, test_z_m, test_v_m, test_l_mm, test_trise, test_tdiff, test_lag, test_number_f, test_num_base, test_m, num_filt, n, k, ytest, x_fft, ftime, test_subkeys)