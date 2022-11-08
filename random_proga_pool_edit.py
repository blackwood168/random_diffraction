#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import iotbx.pdb
import iotbx.mrcfile
from scitbx.array_family import flex
from cctbx.development import random_structure
from cctbx import sgtbx
from cctbx import maptbx
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from multiprocessing import Pool
import time
from scipy.stats import truncnorm

import matplotlib.pyplot as plt

np.random.seed(5)

lambd = 1.540596

def fwhm(peak, lh = 10*0.437, Lam = 1.540596):
    Rad2 = 360 / np.pi
    return lh / 10**3 / Lam * np.tan(peak / Rad2) * Rad2

def LP_Factor(Th2, CeV = 0):
    Deg = np.pi / 180
    A = np.cos(CeV*Deg)**2
    return (1 + A * np.cos(Th2*Deg) ** 2) / (1 + A) / np.sin(Th2*Deg)

def lorenz(Th2, peak, peak_i):
    return (2 / np.pi / fwhm(peak_i)) / (1 + 4 * (Th2 - peak)**2 / fwhm(peak_i)**2)

def h(phi, peak):
    return L*np.sqrt(np.cos(phi*np.pi/180)**2/np.cos(peak*np.pi/180)**2 - 1)
def phi_min(peak):
    return 180/np.pi*np.arccos(np.cos(peak*np.pi/180)*np.sqrt( ((H+S)/L)**2 + 1 ))
def phi_infl(peak):
    return 180/np.pi*np.arccos(np.cos(peak*np.pi/180)*np.sqrt( ((H-S)/L)**2 + 1 ))
def W(phi, peak):
    if phi < phi_min(peak):
        return 0
    if phi_min(peak) <= phi <= phi_infl(peak):
        return H + S - h(phi, peak)
    if phi_infl(peak) <= phi <= peak:
        return 2*min(H, S)
    if phi > peak:
        return 0

def W2(phis, peak):
    result = np.zeros(len(phis))
    cond1 = (phi_min(peak) <= phis) & (phis <= phi_infl(peak))
    result[cond1] = H + S - h(phis[cond1], peak)
    cond2 = (phis > phi_infl(peak)) & (phis <= peak)
    result[cond2] = 2 * min(H, S)
    return result

def pool_peaks(peak_i):
    peak = theta_peaks[peak_i]
    a, b = np.where(peak - 3 <= theta2)[0][0], np.where(theta2 <= peak + 3)[0][-1]
    peak_index = np.where(theta2 <= peak)[0][-1]
    #tmp = tmp / np.sum(tmp) / step * factors[peak_i]
    if peak < 10:
        N_gauss = 20
    elif peak < 30:
        N_gauss = 14
    elif peak < 70:
        N_gauss = 7
    else:
        N_gauss = 4
    xn, wn = np.polynomial.legendre.leggauss(N_gauss)
    deltan = (peak+phi_min(peak))/2 + (peak-phi_min(peak))*xn/2
    tmp_assy = np.zeros(len(theta2[a:b]))
    i = 0
    for phi in theta2[a:b]:
    #    print(deltan)
        if phi == theta2[peak_index]:
            xn, wn = np.polynomial.legendre.leggauss(20)
            deltan = (peak+phi_min(peak))/2 + (peak-phi_min(peak))*xn/2
        sum1 = np.sum(wn*W2(deltan, peak)*lorenz(phi, deltan, peak)/h(deltan, peak)/np.cos(deltan*np.pi/180))
        sum2 = np.sum(wn*W2(deltan, peak)/h(deltan, peak)/np.cos(deltan*np.pi/180))
        tmp_assy[i] = sum1/sum2
        i = i+1
    tmp_assy = tmp_assy / np.sum(tmp_assy) / step * factors[peak_i]
    #y += y_tmp
    #print(y)
    return (a, b, tmp_assy)

def pool_peaks2(peak_i):
    peak = theta_peaks[peak_i]
    a, b = np.where(peak - 3 <= theta2)[0][0], np.where(theta2 <= peak + 3)[0][-1]
    peak_index = np.where(theta2 <= peak)[0][-1]
    #tmp = tmp / np.sum(tmp) / step * factors[peak_i]
    if peak < 10:
        N_gauss = 30
    elif peak < 30:
        N_gauss = 20
    elif peak < 70:
        N_gauss = 20
    else:
        N_gauss = 15
    xn, wn = np.polynomial.legendre.leggauss(N_gauss)
    deltan = (peak+phi_min(peak))/2 + (peak-phi_min(peak))*xn/2
    tmp_assy = np.zeros(len(theta2[a:b]))
    i = 0
    sum2 = np.sum(wn*W2(deltan, peak)/h(deltan, peak)/np.cos(deltan*np.pi/180))
    arr1 = wn*W2(deltan, peak)/h(deltan, peak)/np.cos(deltan*np.pi/180)
    for phi in theta2[a:b]:
    #    print(deltan)
        sum1 = np.sum(arr1 * lorenz(phi, deltan, peak))
        tmp_assy[i] = sum1/sum2
        i = i+1
    tmp_assy = tmp_assy / np.sum(tmp_assy) / step * factors[peak_i]
    #y += y_tmp
    #print(y)
    return (a, b, tmp_assy)
    #y_none[a:b] += tmp


def truncated_normal(mean, stddev, minval, maxval, n):
    a, b = (minval - mean) / stddev, (maxval - mean) / stddev
    r = truncnorm(a,b, loc = mean, scale = stddev)
    return(r.rvs(n))


def dmin (angle = 90):
    return lambd / np.sin(angle/180*np.pi) / 2

def XRS(groups, cell, elemental):
    xrs = random_structure.xray_structure(
      space_group_info = sgtbx.space_group_info(groups),
      elements         = elemental,
      unit_cell        = cell)
    a = xrs.structure_factors(d_min= dmin()).f_calc().sort()
    I = a.as_intensity_array().data().as_numpy_array()
    m = a.multiplicities().data().as_numpy_array()
    for i in range(len(m)):
        I[i] *= m[i]
    Ind = list(a.indices())
    D = a.d_spacings().data().as_numpy_array()
    T2 = a.two_theta(lambd, deg = True).data().as_numpy_array()
    return I, Ind, D, T2


if __name__ == '__main__':
    N = 3    #number of pictures

    cell_a =  truncated_normal(10.05493, 2.792331, 2, 10000, N)
    cell_b = truncated_normal(12.18931, 3.201324, 2, 10000, N)
    cell_c =  truncated_normal(15.10612, 4.623489, 2, 10000, N)
    angle_a = truncated_normal(90, 13.83713, 40, 140, N)
    angle_b = truncated_normal(90, 11.86436, 40, 140, N)
    angle_c = truncated_normal(90, 14.70701, 40, 140, N)



    #setting parameters for assymetry
    L, H, S = 720, 7.5, 10.7
    cells = list(zip(cell_a, cell_b, cell_c, angle_a, angle_b, angle_c))
    groups = "P-1"  #setting for groups
    elemental = [["C"]*randrange(6, 15) for i in range(N)] #setting for elemets
    for i in range(N):
        factors, index, d_s, theta_peaks = XRS(groups, cells[i], elemental[i])
        theta2 = np.arange(1, 90, 0.005)
        theta_peaks = theta_peaks[theta_peaks < 89] # берем только нужные пики
        #print(range(len(theta_peaks)))
        step = theta2[1] - theta2[0]
        y = np.zeros(len(theta2))
        #y_none = np.zeros(len(theta2))

        with Pool(processes = 8) as p:
            z = p.map(pool_peaks2, range(len(theta_peaks)))  #ассиметрия пиков
        #print(z)
        for j in z:
            y[j[0]:j[1]] += j[2]
        #print(y_end)
        #y.wait()
        y = np.multiply(y, LP_Factor(theta2))
        #print(y_end)
        #y_none = np.multiply(y_none, LP_Factor(theta2))
        coeffs = np.random.normal(loc = 0, scale = 1, size = 14)
        xx = np.linspace(-1, 1, len(theta2))
        yy = np.polynomial.chebyshev.chebval(xx, coeffs)
        a, b = 0.2, 90
        x1, x2 = -1, 1                                          #background
        xx = (a - b)/(x1 - x2)*xx + (b*x1-a*x2)/(x1-x2)
        a, b = 0, 25000
        y1, y2 = np.min(yy), np.max(yy)
        yy = (a - b)/(y1 - y2)*yy + (b*y1-a*y2)/(y1-y2)
        y += yy
        #y_none += yy
        file = open('./cryst_edit'+str(i)+'.txt', 'w')
        for j in range(0, len(theta2), 1):
            file.write(str(theta2[j])+ ' ' + str(y[j]))
            file.write('\n')
        file.close()
