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

def fwhm(peak, lh = 0.437, Lam = 1.540596):
    Rad2 = 360 / np.pi
    return lh / 10**3 / Lam * np.tan(peak / Rad2) * Rad2

def fwhmL(peak, X = 410/10**4, Y = 181/10**4):
    peak = peak / 180 * np.pi
    return (X * np.tan(peak/2) + Y/np.cos(peak/2))

def fwhmG(peak, U = 74/10**4, V = -94/10**4, W = 37/10**4, Z = 0):
    peak = peak / 180 * np.pi
    return np.sqrt(U * np.tan(peak/2) ** 2 + V * np.tan(peak/2) + W + Z / np.cos(peak/2) ** 2)

def gauss(Th2, peak, g):
    return (2 * (np.log(2)/np.pi) ** 0.5 / g) * np.exp(-4 * np.log(2) * (Th2 - peak)**2 / g**2)

def n_for_tchz(l, g):
    G = g ** 5 + 2.69269*g ** 4 * l + 2.42843 * g ** 3 * l ** 2 + 4.47163 * g ** 2 * l ** 3
    G += 0.07842 * g * l ** 4 + l ** 5
    G = l / (G ** 0.2)
    n = 1.36603 * G - 0.47719 * G ** 2 + 0.11116 * G ** 3
    return n

def tchz(Th2, peak, l, g, n):
    return n*lorenz(Th2, peak, l) + (1 - n)*gauss(Th2, peak, g)

def LP_Factor(Th2, CeV = 0):
    Deg = np.pi / 180
    A = np.cos(CeV*Deg)**2
    return (1 + A * np.cos(Th2*Deg) ** 2) / (1 + A) / np.sin(Th2*Deg)

def lorenz(Th2, peak, l):
    return (2 / np.pi / l) / (1 + 4 * (Th2 - peak)**2 / l**2)

def h(phi, peak):
    return L*np.sqrt(np.cos(phi*np.pi/180)**2/np.cos(peak*np.pi/180)**2 - 1)
def phi_min(peak):
    return 180/np.pi*np.arccos(np.cos(peak*np.pi/180)*np.sqrt( ((H+S)/L)**2 + 1 ))
def phi_infl(peak):
    return 180/np.pi*np.arccos(np.cos(peak*np.pi/180)*np.sqrt( ((H-S)/L)**2 + 1 ))

def W2(phis, peak):
    result = np.zeros(len(phis))
    cond1 = (phi_min(peak) <= phis) & (phis <= phi_infl(peak))
    result[cond1] = H + S - h(phis[cond1], peak)
    cond2 = (phis > phi_infl(peak)) & (phis <= peak)
    result[cond2] = 2 * min(H, S)
    return result

def pool_peaks2(peak):
    global L, H, S
    L, H, S = 720, 7.5, 10.7
    l, g = fwhmL(peak), fwhmG(peak)
    n = n_for_tchz(l, g)
    a, b = np.where(peak - 3 <= theta2)[0][0], np.where(theta2 <= peak + 3)[0][-1]
    if peak < 10:
        N_gauss = 30
    elif peak < 30:
        N_gauss = 30
    elif peak < 50:
        N_gauss = 20
    elif peak < 70:
        N_gauss = 10
    else:
        N_gauss = 7
    xn, wn = np.polynomial.legendre.leggauss(N_gauss)
    deltan = (peak+phi_min(peak))/2 + (peak-phi_min(peak))*xn/2
    tmp_assy = np.zeros(len(theta2[a:b]))
    arr1 = wn*W2(deltan, peak)/h(deltan, peak)/np.cos(deltan*np.pi/180)
    for dn in range(len(deltan)):
        tmp_assy += arr1[dn] * tchz(theta2[a:b], deltan[dn], l, g, n)
    tmp_assy = tmp_assy / np.sum(tmp_assy) / step * intensity[np.where(peak == theta_peaks_binned)[0]]
    return (a, b, tmp_assy)


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
    I *= m
    #Ind = list(a.indices())
    #D = a.d_spacings().data().as_numpy_array()
    T2 = a.two_theta(lambd, deg = True).data().as_numpy_array()
    return I, T2


if __name__ == '__main__':
    N = 1 #number of pictures
    cell_a =  truncated_normal(10.05493, 2.792331, 2, 10000, N)
    cell_b = truncated_normal(12.18931, 3.201324, 2, 10000, N)
    cell_c =  truncated_normal(15.10612, 4.623489, 2, 10000, N)
    angle_a = truncated_normal(90, 13.83713, 40, 140, N)
    angle_b = truncated_normal(90, 11.86436, 40, 140, N)
    angle_c = truncated_normal(90, 14.70701, 40, 140, N)



    #setting parameters for assymetry
    cells = list(zip(cell_a, cell_b, cell_c, angle_a, angle_b, angle_c))
    groups = "P-1"  #setting for groups
    elemental = [["C"]*np.random.randint(6, 12) for i in range(N)] #setting for elements
    theta2 = np.round(np.arange(1, 90, 0.01), 2)
    step = theta2[1] - theta2[0]
    for i in range(N):
        factors, theta_peaks = XRS(groups, cells[i], elemental[i])
        theta_peaks = np.round(theta_peaks[theta_peaks < 87], 2) # берем только нужные пики
        peaks_indices = np.where(np.in1d(theta2, theta_peaks))[0] #биним пики
        intensity = np.zeros(len(peaks_indices))
        theta_peaks_binned = theta2[peaks_indices]
        for index in range(len(peaks_indices)):
            a = np.where(theta_peaks_binned[index] == theta_peaks)[0][0]   #суммируем интенсивности совпавших пиков, не придумал как это сделать красивее
            b = np.where(theta_peaks_binned[index] == theta_peaks)[0][-1]
            intensity[index] = np.sum(factors[a:b+1])

        I_peaks = np.zeros(len(theta2))

        #with Pool(processes = 8) as p:
        #    z = p.map(pool_peaks2, theta_peaks_binned)  #ассиметрия пиков
        z = map(pool_peaks2, theta_peaks_binned)
        for j in z:
            I_peaks[j[0]:j[1]] += j[2]


        I_peaks *= LP_Factor(theta2)


        coeffs = np.random.normal(loc = 0, scale = 1, size = np.random.randint(5, 15))
        xx = np.linspace(-1, 1, len(theta2))
        bkg = np.polynomial.chebyshev.chebval(xx, coeffs)   #background
        xx1, xx2 = 1, 90
        x1, x2 = -1, 1
        xx = (xx1 - xx2)/(x1 - x2)*xx + (xx2*x1-xx1*x2)/(x1-x2)     #[-1, 1] -> [1, 90]

        a, b = 0, 1
        y1, y2 = np.min(bkg), np.max(bkg)
        bkg_norm = (a - b)/(y1 - y2)*bkg + (b*y1-a*y2)/(y1-y2)       #bkg -> [0, 1]

        a = np.random.uniform(0, 0.5)
        b = np.random.uniform(a, 0.5)
        I_max = np.random.uniform(1000, 100000)             #случайное соотношение
        I = a*I_max + bkg_norm*I_max*(b-a) + I_peaks*(1 - b)

        I = np.random.poisson(I)           #шум


        file = open('./cryst_edit'+str(i)+'.txt', 'w')
        for j in range(0, len(theta2), 1):
            file.write(str(theta2[j])+ ' ' + str(I[j]))
            file.write('\n')
        file.close()
