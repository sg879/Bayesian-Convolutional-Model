# %%
import numpy as np
import pandas as pd
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsc
from jax import jit, vmap, pmap, random, lax, value_and_grad, tree_map

from tqdm import trange
import pickle
import os.path as pt
# %%
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

# %%
attemptno = '8.1.1'
attemptadd = 1
n = 99999
datapath = pt.abspath(pt.join(pt.dirname(__file__), '..', 'Data/Allfinger_veldata.h5'))
loadpath = pt.abspath(pt.join(pt.dirname(__file__), '..', 'Data/Hyperparameters/LinearSum/' + str(n) + '_' + attemptno + 'yvel' + '.pkl'))

# %%
with open(loadpath, 'rb') as f:
    ydict = pickle.load(f)

sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag = ydict["Parameters"]
elbo_history = ydict["ELBO History"]
trainind = ydict["Training Sets"]
steparr = ydict["Steps"]
steplab = ydict["Step Labels"]
isigma_f, iell, isigma_n, iz_fracs, iv_vector, il_diag, il_odiag, itrise, itaudiff, ilag = ydict["Initial Parameters"]
num_f, num_b, num_ind, num_filt, num_tbin, num_mach = ydict["Training Parameters"]
f_maxt, time_bin = ydict["Filter Form"]
mad = ydict["mad"]
vad = ydict["vad"]

# %%
@jit
def AlpEnvelope(Xarr, TRise, TauDiff, Lag):
  TRise = TRise ** 2.0
  TDecay = TRise + TauDiff ** 2 + 1e-8
  TMax = (jnp.log(TRise / TDecay) * TDecay * TRise) / (TRise - TDecay)
  Max =  jnp.exp(- TMax / TDecay) - jnp.exp(- TMax / TRise)
  Delayed = Xarr - Lag
  NewTime = jnp.where(Delayed < 0.0, 1000, Delayed)
  return (jnp.exp(- NewTime / TDecay) - jnp.exp(- NewTime / TRise)) / Max

# %%
@jit
def Squared_exp(I, J, Sigma_f, Ell):
  return Sigma_f**2.0*jnp.exp(-(I-J)**2/(2.0*Ell**2))

# %% [markdown]
# # Importing Spike Trains and Finger Movement

# %%
alldata = pd.read_hdf(datapath) # Import DataFrame

numdat = len(trainind)

# %%
data = [alldata.loc[i] for i in trainind]

# %%
datlen = [len(i.index) for i in data]

# %%
# Set number of time bins (k) and number of filters to use, num_tbin cannot exceed the minimum length in the training set!
batch_size = num_filt // num_mach
subsamp = int(time_bin/0.001)

# %%
spikedat = [data[i].spikes.to_numpy()[:num_tbin, :num_filt].T[:, ::subsamp, None] for i in trainind]

# %%
ytime = [(data[i].index / np.timedelta64(1, 's')).to_numpy()[:num_tbin:subsamp] for i in trainind] # Get spikes/output time array

# %%
# Get x velocities
yraw = [data[i].finger_vel.y.to_numpy()[:num_tbin:subsamp].reshape(n + 1, 1) for i in trainind]

# %% [markdown]
# # Standardising Data

# %%
# Set variance to 1.0
yvel = [yraw[i] / np.std(yraw[i]) for i in trainind]

# %%
del data # Clear data from memory
del alldata

# %%
k = np.floor(f_maxt/time_bin).astype(np.int16) # Maximum index of filter data

ftime = np.linspace(0.0, f_maxt, k + 1).reshape((k + 1, 1)) # Filter corresponding time array

# %%
# FFT of spike train
spikepad = [np.hstack((spikedat[i], np.zeros((num_filt, k, 1)))) for i in trainind]
spikefft = [np.fft.rfft(spikepad[i], axis=1) for i in trainind]
fftlen = [np.shape(spikefft[i])[1] for i in trainind]
spikefft = [spikefft[i].reshape(num_mach, batch_size, fftlen[i], 1) for i in trainind]
spikefft = [[spikefft[j][i] for j in range(len(trainind))] for i in range(num_mach)]
spikefft = jnp.asarray(spikefft)
yvel = jnp.asarray(yvel)

# %% [markdown]
# # ELBO 

# %%
Diag = vmap(jnp.diag)

# %%
@jit
def Solver(Kmm, Diff):
  return jsc.linalg.solve(Kmm, Diff, sym_pos=True, check_finite=True)

# %%
V_Solver = vmap(Solver, in_axes = [None, 0])

# %%
@jit
def Likelihoods(Predictions, Velocity, Sigma_n, N):
  return - 0.5 * ((N + 1) * jnp.log(2 * jnp.pi * Sigma_n ** 2) + \
                    jnp.sum((Velocity - Predictions)**2, axis = -2)/(Sigma_n ** 2))

# %%
Likely = vmap(Likelihoods, in_axes=(0, 0, None, None,))

# %%
@partial(jit, static_argnums = (2, 3,))
def Irfft(Fft, Array, K, N):
  return jnp.fft.irfft(Fft * Array, N + K + 1, axis = -2)[:, :, : N + 1].sum(axis = 1)

# %%
VIrfft = vmap(Irfft, in_axes=(None, 0, None, None,))

# %%
@partial(jit, static_argnums = range(10, 18))
def Neg_ELBO(Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag, Number_F, Num_Base, M, Num_Filt, Batch_Size, K, N, Num_Vel,
              Indices, Spike_Fft, Velocity, F_Time, Subkeys):

  # Creating lag and z-vector         
  Lag = Lag ** 2.0
  
  Z_Vector = (0.5 * jnp.sin( jnp.pi * (Z_Fractions - 0.5)) + 0.5) * (F_Time[-1, 0] - Lag) + Lag

  # Creating the L-matrix
  D = Diag(jnp.exp(L_Diag))

  L_Matrix = D.at[:, Indices[0], Indices[1]].set(L_ODiag)

  # KL term
  KL = 0.5 * (- jnp.sum(jnp.log(jnp.diagonal(L_Matrix, axis1 = 1, axis2 = 2) ** 2)) + \
                jnp.sum(L_Matrix ** 2) + jnp.sum(V_Vector ** 2) - Batch_Size * M)

  # Expectation term
  Thetas = random.normal(Subkeys[0], (1, Batch_Size, 1, Num_Base)) * (1.0 / Ell)

  Taus = random.uniform(Subkeys[1], (1, Batch_Size, 1, Num_Base)) * 2.0 * jnp.pi

  Omegas = random.normal(Subkeys[2], (Number_F, Batch_Size, Num_Base, 1))

  Constant = (Sigma_f * jnp.sqrt(2.0 / Num_Base))

  ZT = Z_Vector.transpose(0, 2, 1)
 
  Phi1 = Constant * jnp.cos(F_Time * Thetas + Taus)
  Phi2 = Constant * jnp.cos(Z_Vector * Thetas + Taus)

  Kmm = Squared_exp(Z_Vector, ZT, Sigma_f, Ell)
  Knm = Squared_exp(F_Time, ZT, Sigma_f, Ell)

  C = jnp.linalg.cholesky(Kmm + jnp.eye(M) * 1e-6)

  V_u = C @ L_Matrix @ L_Matrix.transpose(0, 2, 1) @ C.transpose(0, 2, 1)

  Mu_u = C @ V_Vector

  V_uChol = jnp.linalg.cholesky(V_u + 1e-6 * jnp.eye(M))

  U_Samples = Mu_u + V_uChol @ random.normal(Subkeys[3], (Number_F, Batch_Size, M, 1))

  Vu = V_Solver(Kmm + 1e-6 * jnp.eye(M), U_Samples - Phi2 @ Omegas)

  F_Samples = (Phi1 @ Omegas + Knm @ Vu) * AlpEnvelope(F_Time, TRise, TauDiff, Lag)

  F_Fft = jnp.fft.rfft(F_Samples, n = N + K + 1, axis = -2)

  Filter_Out = VIrfft(F_Fft, Spike_Fft, K, N)

  Pred = lax.psum(Filter_Out, axis_name="machs")

  Likelihood = Likely(Pred, Velocity, Sigma_n, N).mean(axis = 1)

  KL = lax.psum(KL, axis_name="machs")
  Exp = jnp.sum(Likelihood)
                  
  return (KL - Exp)/(Num_Filt * Num_Vel * (N + 1))

# %%
PNeg = pmap(Neg_ELBO, axis_name = "machs", in_axes=(0, 0, None, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, None, None, 0, None, None, 0,),
            static_broadcasted_argnums = range(10, 18))

# %%
Grad_Bound = value_and_grad(Neg_ELBO, argnums = range(10))

# %% [markdown]
# # Training the Model

# %%
@jit
def MCalc(Grad, M, B1 = 0.9):
  return (1 - B1) * Grad + B1 * M

# %%
@jit
def MBias(M, Step, B1 = 0.9):
  return M / (1 - B1 ** (Step + 1))

# %%
@jit
def VCalc(Grad, V, B2 = 0.99):
  return (1 - B2) * jnp.square(Grad) + B2 * V

# %%
@jit
def VBias(V, Step, B2 = 0.99):
  return V / (1 - B2 ** (Step + 1))

# %%
@jit
def CFinState(X, Mhat, Vhat, Step_Size = 1e-2, Eps = 1e-8):
  return X - Step_Size * Mhat / (jnp.sqrt(Vhat) + Eps)

# %%
@jit
def MFinState(X, Mhat, Vhat, Step_Size = 1e-3, Eps = 1e-8):
  return X - Step_Size * Mhat / (jnp.sqrt(Vhat) + Eps)

# %%
@jit
def FFinState(X, Mhat, Vhat, Step_Size = 1e-5, Eps = 1e-8):
  return X - Step_Size * Mhat / (jnp.sqrt(Vhat) + Eps)

# %%
@jit
def CAdam(Step, X, Grad, M, V):

  M = tree_map(MCalc, Grad, M) # First  moment estimate.
  V = tree_map(VCalc, Grad, V)  # Second moment estimate.
  Step = tuple(Step * jnp.ones(10))
  Mhat = tree_map(MBias, M, Step) # Bias correction.
  Vhat = tree_map(VBias, V, Step) # Bias correction.

  X = tree_map(CFinState, X, Mhat, Vhat)

  return X, M, V

# %%
@jit
def MAdam(Step, X, Grad, M, V):

  M = tree_map(MCalc, Grad, M) # First  moment estimate.
  V = tree_map(VCalc, Grad, V)  # Second moment estimate.
  Step = tuple(Step * jnp.ones(10))
  Mhat = tree_map(MBias, M, Step) # Bias correction.
  Vhat = tree_map(VBias, V, Step) # Bias correction.

  X = tree_map(MFinState, X, Mhat, Vhat)

  return X, M, V

# %%
@jit
def FAdam(Step, X, Grad, M, V):

  M = tree_map(MCalc, Grad, M) # First  moment estimate.
  V = tree_map(VCalc, Grad, V)  # Second moment estimate.
  Step = tuple(Step * jnp.ones(10))
  Mhat = tree_map(MBias, M, Step) # Bias correction.
  Vhat = tree_map(VBias, V, Step) # Bias correction.

  X = tree_map(FFinState, X, Mhat, Vhat)

  return X, M, V

# %%
@partial(pmap, axis_name = "machs", in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, None, 
                                    None, None, None, None, None, None, None, None, 
                                    0, None, None, 0, 0, 0, 0,),
                static_broadcasted_argnums = range(11, 19))

def _CUpdate(Iter, Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag, Num_Coarse_Fs, Num_Base, M, Num_Filt, Batch_Size, K, N, Num_Vel,
              Indices, Spike_Fft, Velocity, F_Time, Subkeys, Key, Mad, Vad):
  
  Value, Grads = Grad_Bound(Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag, Num_Coarse_Fs, Num_Base, M, Num_Filt, Batch_Size, K, N, Num_Vel,
                  Indices, Spike_Fft, Velocity, F_Time, Subkeys)
  
  X, Mad, Vad = CAdam(Iter, (Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag), Grads, Mad, Vad)

  Key, *Subkeys = random.split(Key, 5)
  Subkeys = jnp.asarray(Subkeys).astype(jnp.uint32).reshape(4, 2)

  return *X, Subkeys, Key, Mad, Vad, Value

# %%
@partial(pmap, axis_name = "machs", in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, None, 
                                    None, None, None, None, None, None, None, None,
                                    0, None, None, 0, 0, 0, 0,),
                static_broadcasted_argnums = range(11, 19))

def _MUpdate(Iter, Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag, Num_Mid_Fs, Num_Base, M, Num_Filt, Batch_Size, K, N, Num_Vel,
              Indices, Spike_Fft, Velocity, F_Time, Subkeys, Key, Mad, Vad):
  
  Value, Grads = Grad_Bound(Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag, Num_Mid_Fs, Num_Base, M, Num_Filt, Batch_Size, K, N, Num_Vel,
                  Indices, Spike_Fft, Velocity, F_Time, Subkeys)
  
  X, Mad, Vad = MAdam(Iter, (Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag), Grads, Mad, Vad)

  Key, *Subkeys = random.split(Key, 5)
  Subkeys = jnp.asarray(Subkeys).astype(jnp.uint32).reshape(4, 2)

  return *X, Subkeys, Key, Mad, Vad, Value

# %%
@partial(pmap, axis_name = "machs", in_axes=(None, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, None, 
                                    None, None, None, None, None, None, None, None,
                                    0, None, None, 0, 0, 0, 0,),
                static_broadcasted_argnums = range(11, 19))

def _FUpdate(Iter, Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag, Num_Fine_Fs, Num_Base, M, Num_Filt, Batch_Size, K, N, Num_Vel, 
              Indices, Spike_Fft, Velocity, F_Time, Subkeys, Key, Mad, Vad):
  
  Value, Grads = Grad_Bound(Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag, Num_Fine_Fs, Num_Base, M, Num_Filt, Batch_Size, K, N, Num_Vel, 
                  Indices, Spike_Fft, Velocity, F_Time, Subkeys)
  
  X, Mad, Vad = FAdam(Iter, (Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag), Grads, Mad, Vad)

  Key, *Subkeys = random.split(Key, 5)
  Subkeys = jnp.asarray(Subkeys).astype(jnp.uint32).reshape(4, 2)

  return *X, Subkeys, Key, Mad, Vad, Value

# %%
@partial(pmap, in_axes = (0, 0, None, 0, 0, 0, 0, 0, 0, 0,))
def Init_Adam(Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag):
  X = (Sigma_f, Ell, Sigma_n, Z_Fractions, V_Vector, L_Diag, L_ODiag, TRise,
              TauDiff, Lag)

  Mad = tree_map(jnp.zeros_like, X)

  return Mad, Mad


# %% [markdown]
# # Y Velocity

# %%

indices = jnp.asarray(jnp.tril_indices(num_ind, -1))

# Random number generator
iopt_key = np.asarray([random.PRNGKey(i) for i in range(num_mach)]).astype(np.uint32)
itest = vmap(random.split, in_axes=(0, None))(iopt_key, 1 + 4)
iopt_key = np.asarray([itest[i][0] for i in range(num_mach)]).astype(np.uint32)
iopt_subkey = np.asarray([itest[i][1:] for i in range(num_mach)]).astype(np.uint32)

opt_key = iopt_key
opt_subkey = iopt_subkey

# %%
extra_steps = 3000
steps_label = "Fine"
step = len(elbo_history)

elbo_history = np.hstack((elbo_history, np.zeros(extra_steps)))
savepath = pt.abspath(pt.join(pt.dirname(__file__), '..', 'Data/Hyperparameters/LinearSum/' + str(n) + '_' + attemptno + '.' + str(attemptadd) + 'yvel' + '.pkl'))

# %%
print(PNeg(sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag,
                l_odiag, trise, taudiff, lag, num_f, num_b,
                num_ind, num_filt, batch_size, k, n, numdat, indices, spikefft, yvel, 
                ftime, opt_subkey))

# %%

if steps_label == "Coarse":
  for i in trange(extra_steps):  
    sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag, opt_subkey, opt_key, mad, vad, value = _CUpdate(i, sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag, num_f, num_b, num_ind, num_filt, batch_size, k, n, numdat, indices, spikefft, yvel, ftime, opt_subkey, opt_key, mad, vad)
    sigma_n = sigma_n[0]
    elbo_history[step] = -value[0]
    step += 1

elif steps_label == "Mid":
  for i in trange(extra_steps): 
    sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag, opt_subkey, opt_key, mad, vad, value = _MUpdate(i, sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag, num_f, num_b, num_ind, num_filt, batch_size, k, n, numdat, indices, spikefft, yvel, ftime, opt_subkey, opt_key, mad, vad)
    sigma_n = sigma_n[0]
    elbo_history[step] = -value[0]
    step += 1

elif steps_label == "Fine":
  for i in trange(extra_steps):  
    sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag, opt_subkey, opt_key, mad, vad, value = _FUpdate(i, sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag, num_f, num_b, num_ind, num_filt, batch_size, k, n, numdat, indices, spikefft, yvel, ftime, opt_subkey, opt_key, mad, vad)
    sigma_n = sigma_n[0]
    elbo_history[step] = -value[0]
    step += 1

# %%
print(PNeg(isigma_f, iell, isigma_n, iz_fracs, iv_vector, il_diag,
               il_odiag, itrise, itaudiff, ilag, num_f, num_b,
                num_ind, num_filt, batch_size, k, n, numdat, indices, spikefft, yvel, 
                ftime, iopt_subkey))

# %%
print(PNeg(sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag,
                l_odiag, trise, taudiff, lag, num_f, num_b,
                num_ind, num_filt, batch_size, k, n, numdat, indices, spikefft, yvel, 
                ftime, opt_subkey))

# %%
outdict = {"Parameters": (sigma_f, ell, sigma_n, z_fracs, v_vector, l_diag, l_odiag, trise, taudiff, lag),
          "ELBO History": elbo_history,
          "Training Sets": trainind,
          "Steps": steparr,# steparr.append(extra_steps),
          "Step Labels":  steplab,# steplab.append(steps_label),
          "Initial Parameters": (isigma_f, iell, isigma_n, iz_fracs, iv_vector, il_diag, il_odiag, itrise, itaudiff, ilag),
          "Training Parameters": (num_f, num_b, num_ind, num_filt, num_tbin, num_mach),
          "Filter Form": (f_maxt, time_bin),
          "mad": mad,
          "vad": vad,
          "Velocity Trained": "y"}
with open(savepath, 'wb') as f:
    pickle.dump(outdict, f)