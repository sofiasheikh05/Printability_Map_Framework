#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#libraries
import numpy as np
import pandas as pd
from itertools import compress
import time
import concurrent.futures
from pickle import dump
from pickle import load
import psutil
import os.path as path

import time
import logging
import datetime
from scipy.constants import R
import os
import sys
from scipy.integrate import *
import scipy.version
import ctypes
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
from mendeleev import element 
from itertools import product

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.pipeline import Pipeline
import concurrent.futures


# In[ ]:


def ROM_THERMO(results_df):
    
    element_list_joined = []
    mol_weight = []
    density_list= []
    T_m = []
    T_b = []
    T_sint = []



    for i in range(results_df.shape[0]):
        element_list = results_df.loc[i,'Elements_active']
        atomic_fraction_arr = (results_df.loc[i,'atomic_per'])*0.01
        element_list_joined = ''.join(element_list)


        comp_obj = Composition(element_list_joined)
        element_obj_list = comp_obj.elements
        #need to convert at% to wt% before inserting into comp_obj because it only captures atomic fraction 
        elemental_mass_arr = np.array([element_obj.atomic_mass for element_obj in element_obj_list]) #mass of each element
        #moles = wt_per/elemental_mass_arr
        #total_moles = sum(moles)
        #atomic_fraction_arr = np.array(moles/total_moles)

        #atomic_fraction_arr = np.array(atomic_arr[i])


        #calculate molecular weight
        molecular_weight = np.sum(atomic_fraction_arr * elemental_mass_arr)

        #Calculate Density
        elemental_density_arr = np.array([element_obj.density_of_solid for element_obj in element_obj_list]) # kg/m^3
        try:
            density = np.sum(atomic_fraction_arr*elemental_density_arr)
        except TypeError:
            try:
                index=[k for k in range(len(elemental_density_arr)) if elemental_density_arr[k] == None] 
                for idx in index:
                    elem=str(element_list[ele][idx])
                    ele_dens = (element(elem).density) *1000 
                    if ele_dens == None:
                        density = np.nan
                    else:
                        elemental_density_arr[idx] = ele_dens 
                density = np.sum(atomic_fraction_arr*elemental_density_arr)
            except TypeError:
                index=[k for k in range(len(elemental_density_arr)) if elemental_density_arr[k] == 'Md']
                elemental_density_arr[idx] = 10.3*1000 
            density = np.sum(atomic_fraction_arr*elemental_density_arr)

        #melting temperature 
        elemental_melting_point_arr = np.array([element_obj.melting_point for element_obj in element_obj_list]) #kelvins
        melting_temperature = np.sum(atomic_fraction_arr*elemental_melting_point_arr)

        #boiling temperature 
        elemental_boiling_point_arr = np.array([element_obj.boiling_point for element_obj in element_obj_list]) #kelvins
        print(elemental_boiling_point_arr)
        
        try:
            boiling_temperature = np.sum(atomic_fraction_arr*elemental_boiling_point_arr)
        except Exception as e2:
            #if "Md" in element_list[ele]:
            boiling_temperature = np.nan


        #Sintering Temperature
        sintering_temp = 0.3*melting_temperature

        mol_weight.append(molecular_weight)
        density_list.append(density)
        T_m.append(melting_temperature)
        T_b.append(boiling_temperature)
        T_sint.append(sintering_temp)


    results_df['ROM_Molecular_Weight [g/mol]'] = mol_weight
    results_df['ROM_Density_ROM kg/m3'] = density_list
    results_df['ROM_Melting_Temp_[K]'] = T_m
    results_df['ROM_Boiling_Temp_[K]'] = T_b
    results_df['ROM_Sintering_Temp_[K]'] = T_sint

    return results_df


# In[ ]:


def melt_pool_dimensionless (data):
    params = data['par']
    results_df = data['mat']
    
    v_mm = params['Velocity_m/s']*1000 
    t_mm = params['Powder_thickness_um']*0.001
    h_mm = params['hatch_spacing_um']*0.001
    t_m = params['Powder_thickness_um']*0.001
    h_m = params['hatch_spacing_um']*0.001
    d_laser_mm = params['d_laser_um'] *0.001
    r_laser_mm = d_laser_mm/2
    d_laser_m = params['d_laser_um']*1e-6
    r_laser_m = d_laser_m/2
    
    r_laser_um = params['d_laser_um']/2
    mass_kg = results_df['ROM_Molecular_Weight [g/mol]']*0.001 #kg/mol
    
    
    T_liquidus_298 = results_df['PROP LT (K)'] - 298
    results_df['equil_SR_K'] = results_df['PROP LT (K)'] - results_df["PROP ST (K)"]
    results_df['H_total_J'] = results_df['EQUIL Liquidus H (J)'] - results_df['EQUIL RT H (J)']
    results_df['H_total_Jmol'] = results_df['EQUIL Liquidus H (J/mol)'] - results_df['EQUIL RT H (J/mol)']    
    results_df['eff_Cp_(J/molK)'] = results_df['H_total_Jmol']/T_liquidus_298
    results_df['H_melting_J'] = results_df['EQUIL Liquidus H (J)'] - results_df['EQUIL Solidus H (J)']
    results_df['H_boiling_(J/mol)'] = 10*R*results_df['PROP LT (K)']  
    results_df['ROM_H_boiling_(J/mol)'] = 10*R*results_df['ROM_Melting_Temp_[K]']
    results_df['H_at_boiling_(J/mol)'] = results_df['EQUIL Liquidus H (J/mol)'] + results_df['Prop Liquidus Heat capacity (J/(mol K))'] * (results_df['ROM_Boiling_Temp_[K]'] - results_df['PROP LT (K)'])
    results_df['H_after_boiling_(J/mol)'] = results_df['H_at_boiling_(J/mol)'] + results_df['H_boiling_(J/mol)']
    
    #Energy Densities 
    params['LED_J/mm'] = params['Power']/v_mm
    params['SED_J/mm2'] = params['Power']/(v_mm*h_mm)
    params['VED_J/mm3'] = params['Power']/(v_mm*t_mm*h_mm)
    params['generic_VED_J/mm'] = (params['powder_grain_size_um']/params['d_laser_um'])*params['VED_J/mm3']
    
    density = results_df['Prop RT Density (g/cm3)']*1000
    density_liq = results_df['Prop Liquidus Density (g/cm3)'] * 1000
    
    
    #Dimensionless values 
    results_df['eff_Cp_JkgK'] = (results_df['eff_Cp_(J/molK)']/(mass_kg)) #J/kgK
    eff_Cp_JkgK = results_df['eff_Cp_JkgK']
    #results_df['h_s_J/m3'] = (results_df['H_total_Jmol'])*(density)*(1/mass_kg)
    results_df['h_s_J/m3'] = eff_Cp_JkgK*results_df['PROP LT (K)']*(density) #J/m3
    
    
    #All unique power and beam size radius values from params database
    power = params['Power'].unique()
    r_laser = r_laser_m.unique()
    vel_ms = params['Velocity_m/s'].unique()
    t_m = (params['Powder_thickness_um'].unique())*1e-6
    h_m = (params['hatch_spacing_um'].unique())*1e-6
    laser_wavelength_m = (params['laser_wavelength_nm'].unique())*1e-9
    
    h_s_list = []

    
    H_specific_Jm3 = []

    
    H_normalized = []
    
    
    
    H_after_boiling  = []
    H_at_boiling = []
    thermal_diffusion_time = []
    B = []
    p = []
    dwell_time = []
    T_surface = []
    composition = []
    atomic = []
    Power = []
    velocity = []
    beam_radius_m = []
    wavelength_m = []
    powder_t_m = []
    hatch_spacing_m = []
    absorp_drude = []
    P_dimensionless = []
    v_dimensionless = []
    t_dimensionless = []
    h_dimensionless = []
    T_liq = []
    density = []
    mass_kg= []
    T_b = []
    
    cp_eff = []
    cp_liq = []
    
    T_solid= []
    H_latent = []
    density_liq = [] # convert to kg/m^3 for units to cancel
    
    surf_tens_liq = []
    thermal_cond_RT = []
    thermal_diff_liq = []
    thermal_cond_liq = [] 
    H_boiling = []
    
    
    VED_dimensionless = []
    
    SED_dimensionless = []
    
    LED_dimensionless = []
    
    Ma_number = []
    
    
    for P in power: 
        for v in vel_ms:
            for beam_rad in r_laser:
                for wave in laser_wavelength_m:
                    for t in t_m:
                        for h in h_m:
                            for idx in results_df.index:
                                h_s = results_df.loc[idx,'h_s_J/m3']
                                
                                absorp_drude_calc = (0.365*np.sqrt((results_df.loc[idx,'Prop Liquidus Electric resistivity (Ohm m)']/wave))) #0.75
                                H_specific = (absorp_drude_calc * P)/((np.pi)*(np.sqrt(results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']*v*(beam_rad**3)))) #J/m3
                                delta_H = ((2**(3/4))*absorp_drude_calc*P)/(np.sqrt(np.pi*results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']*v*(beam_rad**3)))
    
                                beam_rad_Gaussian = np.sqrt(2*beam_rad)
    
                                B_list = delta_H/((2**(3/4)*np.pi*h_s))
                                dwell_time_list = beam_rad/v
                                thermal_diffusion_time_list = np.sqrt((results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']*beam_rad)/(v**2)) 
                                p_list = results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']/(v*beam_rad)
                                T_surface_list = (absorp_drude_calc*P)/(np.pi*(results_df.loc[idx,'Prop RT Density (g/cm3)']*1000)*results_df.loc[idx,'eff_Cp_JkgK']*np.sqrt(results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']*v*(beam_rad**3))) #check formula
    
    
                                P_dimensionless_list = (absorp_drude_calc*P)/(beam_rad*results_df.loc[idx,'Prop Liquidus Thermal conductivity (W/(mK))'] *(results_df.loc[idx,'PROP LT (K)']-298))
    
                                v_dimensionless_list = (v*beam_rad)/results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']
    
                                t_dimensionless_list = (2*t)/beam_rad
    
                                h_dimensionless_list = h/beam_rad
    
                                VED_dimensionless_list = P_dimensionless_list/(v_dimensionless_list*t_dimensionless_list*h_dimensionless_list)
    
                                SED_dimensionless_list = P_dimensionless_list/(v_dimensionless_list*h_dimensionless_list)
    
                                LED_dimensionless_list = P_dimensionless_list/(v_dimensionless_list)
    
                                Ma_number_list = - (results_df.loc[idx,'EQUIL Liquidus Surface Tension (N/m)'] - results_df.loc[idx,'EQUIL RT Surface Tension (N/m)']) *((t*(results_df.loc[idx,'PROP LT (K)']-298))/(results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']*results_df.loc[idx, 'EQUIL Liquidus DVIS (Pa-s)']))
    
    
                                #Put everything in the empty dataframe
                                T_solid.append(results_df.loc[idx,"PROP ST (K)"])
                                H_latent.append(results_df.loc[idx,'Latent Heat Fusion (J)'])
                                density_liq.append(results_df.loc[idx,'Prop Liquidus Density (g/cm3)'] * 1000)  # convert to kg/m^3 for units to cancel
                                cp_liq.append(results_df.loc[idx,'Prop Liquidus Heat capacity (J/(mol K))'])
                                thermal_diff_liq.append(results_df.loc[idx,'Prop Liquidus Thermal diffusivity (m2/s)']) #should be LT
                                thermal_cond_liq.append(results_df.loc[idx,'Prop Liquidus Thermal conductivity (W/(mK))'])
                                surf_tens_liq.append(results_df.loc[idx,'EQUIL Liquidus Surface Tension (N/m)'])
                                thermal_cond_RT.append(results_df.loc[idx,'EQUIL RT Thermal Conductivity (W/mK)'])
    
    
                                cp_eff.append(results_df.loc[idx,'eff_Cp_JkgK'])
    
                                h_s_list.append(h_s)
    
                                H_specific_Jm3.append(H_specific)

                                H_normalized.append(H_specific/h_s)
    
    
                                T_liq.append(results_df.loc[idx,'PROP LT (K)'])
                                
                                print(element_list[idx])
                                composition.append(element_list[idx])
                                atomic.append(atomic_arr[idx])
    
                                Power.append(P)
                                velocity.append(v)
                                beam_radius_m.append(beam_rad)
                                wavelength_m.append(wave)
                                powder_t_m.append(t)
                                hatch_spacing_m.append(h)
                                density.append(results_df.loc[idx,'Prop RT Density (g/cm3)']*1000)
                                H_after_boiling.append(results_df.loc[idx,'H_after_boiling_(J/mol)'])
                                H_at_boiling.append(results_df.loc[idx,'H_at_boiling_(J/mol)'])                            
                                mass_kg.append(results_df.loc[idx,'ROM_Molecular_Weight [g/mol]']*0.001)
                                T_b.append(results_df.loc[idx,'ROM_Boiling_Temp_[K]'])
                                absorp_drude.append(absorp_drude_calc)
                                H_boiling.append(results_df.loc[idx,'H_boiling_(J/mol)'])
                                #H_specific_Jm3.append(H_specific)
                                #H_normalized.append(H_specific/h_s) #recheck formula for normalized enthalpy  
                                B.append(B_list)
                                dwell_time.append(dwell_time_list)
                                thermal_diffusion_time.append(thermal_diffusion_time_list)
                                p.append(p_list)
                                T_surface.append(T_surface_list)
                                P_dimensionless.append(P_dimensionless_list)
                                v_dimensionless.append(v_dimensionless_list)
                                t_dimensionless.append(t_dimensionless_list)
                                h_dimensionless.append(h_dimensionless_list)
                                VED_dimensionless.append(VED_dimensionless_list)
                                SED_dimensionless.append(SED_dimensionless_list)
                                LED_dimensionless.append(LED_dimensionless_list)
                                
                                Ma_number.append(Ma_number_list)
    #currently all prints into one dataframe --- think about printing multiple dataframe based on composition 
    
    dimensionless_df = pd.DataFrame()
    dimensionless_df['Elements'] = composition
    dimensionless_df['Atomic_frac'] = atomic
    dimensionless_df['Power'] = Power
    dimensionless_df['Velocity_m/s'] = velocity
    dimensionless_df['Hatch_spacing_m'] = hatch_spacing_m
    dimensionless_df['Powder_thick_m'] = powder_t_m
    dimensionless_df['Beam_radium_m'] = beam_radius_m
    dimensionless_df["Beam_diameter_m"] = ((np.array(beam_radius_m))*2 )
    dimensionless_df['thermal_cond_liq'] = thermal_cond_liq
    dimensionless_df['Cp_J/kg'] = cp_eff
    dimensionless_df['Laser_WaveLength_m'] = wavelength_m
    
    
    dimensionless_df['T_solidus'] = T_solid
    dimensionless_df['Latent_Heat'] = H_latent
    dimensionless_df['Density_liq_kg/m3'] = density_liq 
    dimensionless_df['Cp_liq'] = cp_liq
    dimensionless_df['thermal_diff_liq'] = thermal_diff_liq
    dimensionless_df['thermal_cond_liq']=thermal_cond_liq
    dimensionless_df['surf_tens_liq'] = surf_tens_liq 
    dimensionless_df['thermal_cond_RT'] = thermal_cond_RT
    
    
    
    dimensionless_df['T_liquidus'] = T_liq
    dimensionless_df["Density_kg/m3"] = density
    dimensionless_df['Absorptivity'] = absorp_drude
    
    dimensionless_df['H_specific_J/m3'] = H_specific_Jm3

    
    dimensionless_df['h_s_J/m3'] = h_s_list

    
    
    dimensionless_df['H_normalized'] = H_normalized
    
    dimensionless_df['Marangoni Number'] = Ma_number

    dimensionless_df['H_boiling_(J/mol)'] = H_boiling
    dimensionless_df['H_after_boiling'] = H_after_boiling
    dimensionless_df['H_at_boiling'] = H_at_boiling
    dimensionless_df['mass_kg'] = mass_kg
    dimensionless_df['T_b'] = T_b
    dimensionless_df['B'] = B
    dimensionless_df['p'] = p
    dimensionless_df['dwell_time_s'] = dwell_time
    dimensionless_df['Thermal_diffusion_time_s'] = thermal_diffusion_time
    dimensionless_df['T_surface_K'] = T_surface
    dimensionless_df['dimensioness_P'] = P_dimensionless
    dimensionless_df['dimensioness_v'] = v_dimensionless
    dimensionless_df['dimensionless_t'] = t_dimensionless
    dimensionless_df['dimensionless_h'] = h_dimensionless
    dimensionless_df['LED_dimensionless'] = LED_dimensionless
    dimensionless_df['SED_dimensionless'] = SED_dimensionless
    dimensionless_df['VED_dimensionless'] = VED_dimensionless
    
    
    #Energy Densities 
    dimensionless_df['LED_J/mm'] = dimensionless_df['Power']/(dimensionless_df['Velocity_m/s']*1000)
    dimensionless_df['SED_J/mm2'] = dimensionless_df['Power']/((dimensionless_df['Hatch_spacing_m']*1000)*(dimensionless_df['Velocity_m/s']*1000))
    dimensionless_df['VED_J/mm3'] = dimensionless_df['Power']/((dimensionless_df['Hatch_spacing_m']*1000)*(dimensionless_df['Velocity_m/s']*1000)*(dimensionless_df['Powder_thick_m']*1000))
    params['generic_VED_J/mm'] = (params['powder_grain_size_um']/params['d_laser_um'])*params['VED_J/mm3']

    return params,dimensionless_df 



def scaled_ET(dimensionless_df):
    dimensionless_df['V_pool'] = np.nan
    dimensionless_df['M_pool'] = np.nan
    dimensionless_df['Eff_Cp_melt_pool'] = np.nan
    dimensionless_df['width'] = np.nan
    dimensionless_df['length'] = np.nan
    dimensionless_df['depth'] = np.nan

    for i in dimensionless_df.index:
        if dimensionless_df.loc[i,'p'] <= 1:
            T_max_scaled = dimensionless_df.loc[i,'T_liquidus']*(2.1-1.9*dimensionless_df.loc[i,'p']+0.67*(dimensionless_df.loc[i,'p']**2))*dimensionless_df.loc[i,'B']    # Peak Temoerature (K): Valid just for p<1
        else:
            T_max_scaled = np.NAN

        dimensionless_df.at[i,'T_max'] = T_max_scaled
        
        B = dimensionless_df.loc[i,'B']
        p = dimensionless_df.loc[i,'p']
        a = dimensionless_df.loc[i,'Beam_radium_m']

        s = (a/np.sqrt(p))*(0.008-0.0048*B-0.047*p-0.099*B*p+(0.32+0.015*B)*p*np.log(p)
         +np.log(B)*(0.0056-0.89*p+0.29*p*np.log(p)))*-1  
        
        
        dimensionless_df.at[i,'depth'] = s

        l = (a/(p**2))*(0.0053-0.21*p+1.3*(p**2)+(-0.11-0.17*B)*(p**2)*np.log(p)
         +B*(-0.0062+0.23*p+0.75*(p**2)))     #length in m 
                     
        dimensionless_df.at[i,'length'] = l

        w = (a/(B*(p**3)))*(0.0021-0.047*p+0.34*(p**2)-1.9*(p**3)-0.33*(p**4)
         +B*(0.00066-0.0070*p-0.00059*(p**2)+2.8*(p**3)-0.12*(p**4))
         +(B**2)*(-0.00070+0.015*p-0.12*(p**2)+0.59*(p**3)-0.023*(p**4))
         +(B**3)*(0.00001-0.00022*p+0.0020*(p**2)-0.0085*(p**3)+0.0014*(p**4)))  #width in m
             
        dimensionless_df.at[i,'width'] = w

        H_after_boiling =dimensionless_df.loc[i,'H_after_boiling']
        H_at_boiling = dimensionless_df.loc[i,'H_at_boiling']
        mass_kg = dimensionless_df.loc[i,'mass_kg']
        T_b = dimensionless_df.loc[i,'T_b']
        absorp_drude =  dimensionless_df.loc[i,'Absorptivity']                         
        v = dimensionless_df.loc[i,'Velocity_m/s']
        P = dimensionless_df.loc[i,'Power']






        V_pool = (np.pi/6)*s*l*w                     # Melt pool volume (m3)
        dimensionless_df.at[i,'V_pool'] = V_pool

        density = dimensionless_df.loc[i,"Density_kg/m3"]

        M_pool = density*V_pool
        dimensionless_df.at[i,'M_pool'] = M_pool           # Melt pool mass (kg)




        Eff_Cp_melt_pool = (H_after_boiling*M_pool)/(mass_kg*(T_b-298))# Effective Cp obtained by the enthalpy after boiling for the melt pool
        dimensionless_df.at[i,'Eff_Cp_melt_pool'] = Eff_Cp_melt_pool

        T_max_est = (absorp_drude*P*(l/v))/Eff_Cp_melt_pool+298              # Estimated maximum temperature (K) using the deposited energy as the laser beam passes along the melt pool length                
        dimensionless_df.at[i,'T_max_est'] = T_max_est


        Res_t = l/v                                  # Residence time (s)
        dimensionless_df.at[i,'Residence_time'] = Res_t

        Q = absorp_drude*P*Res_t                                # Deposited energy (J) by the laser beam as it passes along the melt pool length
        dimensionless_df.at[i,'Q_dep_energy_J'] = Q
        #delta_H_b = delta_H_pb-10*8.314*T_m    # Molar enthalpy at boiling (J/mole)



        Q_b = (H_at_boiling*M_pool)/mass_kg                  # Melt pool enthalpy at boiling (J)
        dimensionless_df.at[i,'H_MP_at_boiling_J'] = Q_b

        Q_pb = (H_after_boiling*M_pool)/mass_kg                 # Melt pool enthalpy after boiling (J)
        dimensionless_df.at[i,'H_MP_after_boiling_J'] = Q_pb

    return dimensionless_df     
#ET Model NN


def ET_NN(dimensionless_df):

    def load_pipeline_keras(model, folder_name="model"):
        #standardize = pickle.load(open(folder_name+'/'+standard,'rb'))
        build_model = lambda: load_model(folder_name+'/'+model)
        reg = KerasRegressor(build_fn=build_model, epochs=50, batch_size=5, verbose=0)
        reg.model = build_model()
        return Pipeline([
            ('lstm', reg)
        ])



    #data=pd.read_csv('input.csv')
    #create input file

    
    data = pd.DataFrame()
    data['Elements'] = dimensionless_df['Elements']
    data['Atomic_frac'] = dimensionless_df['Atomic_frac']
    
    
    
    data['v']=dimensionless_df['Velocity_m/s']
    data['P'] = dimensionless_df['Power']
    data['twoSigma'] = dimensionless_df["Beam_diameter_m"] #m
    data['A'] = dimensionless_df['Absorptivity']
    data['tMelt'] = dimensionless_df['T_liquidus']
    data['rho'] = dimensionless_df["Density_kg/m3"]
    data['k'] = dimensionless_df['thermal_cond_liq']
    data['cp'] = dimensionless_df['Cp_J/kg']
    #dimensionless_df['tMelt'] =dimensionless_df['T_liquidus'] 
    rows_with_nan = [index for index, row in data.iterrows() if row.isnull().any()]
    data = data.dropna(axis=0,how='any',inplace=False)
    elements = data['Elements']
    atomic = data['Atomic_frac']
    data = data.drop(['Elements','Atomic_frac'],axis=1)

    
    
    
    # load the model
    Classifier_ = load(open('classifier_4_classes_119.pkl', 'rb'))
    # load the scaler
    scaler = load(open('scaler_x_all_119.pkl', 'rb'))
    X = scaler.transform(data)

    #print(X[0,:])

    Classes = Classifier_.predict(X)
    Classes_in_model=set(Classes)

    data_df = pd.DataFrame(X)


    data_df['8']=Classes

    #print(data_df.head())

    # load the scaler

    Tmin_Tmax=load_pipeline_keras('Regression_Tmin_Tmax.h5', folder_name="Tmin_Tmax")
    #Result
    Tmin_Tmax_values = Tmin_Tmax.predict(X)
    #print(Tmin_Tmax_values[0:5,:])
    #Divide datasets for different calculations
    Class_0=data_df[data_df['8']==0]
    Class_1=data_df[data_df['8']==1]
    Class_2=data_df[data_df['8']==2]
    Class_3=data_df[data_df['8']==3]
    #Drop Class Column
    Class_0=Class_0.drop(columns=['8'])
    Class_1=Class_1.drop(columns=['8'])
    Class_2=Class_2.drop(columns=['8'])
    Class_3=Class_3.drop(columns=['8'])

    Classes=pd.DataFrame()
    if 0 in Classes_in_model:
        Class_0['8']=0
        Class_0['9']=0
        Class_0['10']=0
        Classes=Classes.append(Class_0)



    if 1 in Classes_in_model:
        Class_1_regression=load_pipeline_keras('Class_1_regression.h5', folder_name="Class_1_regression")
        y_1=Class_1_regression.predict(Class_1.to_numpy())
        scaler_y1 = load(open('Y_1_tranformation_1110.pkl', 'rb'))
        y_1=scaler_y1.inverse_transform(y_1)
        try:
            Class_1['8']=y_1[:,0]
            Class_1['9']=y_1[:,1]
        except:
            Class_1['8']=y_1[0]
            Class_1['9']=y_1[1]

        Class_1['10']=0
        Classes=Classes.append(Class_1)

    if 2 in Classes_in_model:
        Class_2_regression=load_pipeline_keras('Class_2_regression.h5', folder_name="Class_2_regression")
        y_2=Class_2_regression.predict(Class_2.to_numpy())
        scaler_y2 = load(open('Y_2_tranformation_1110.pkl', 'rb'))
        y_2=scaler_y2.inverse_transform(y_2)
        try:
            Class_2['8']=y_2[:,0]
            Class_2['9']=y_2[:,1]
            Class_2['10']=y_2[:,2]
        except:
            Class_2['8']=y_2[0]
            Class_2['9']=y_2[1]
            Class_2['10']=y_2[2]
        Classes=Classes.append(Class_2)


    if 3 in Classes_in_model:
        Class_3_regression=load_pipeline_keras('Class_3_regression.h5', folder_name="Class_3_regression")
        y_3=Class_3_regression.predict(Class_3.to_numpy())
        scaler_y3 = load(open('Y_3_tranformation_1110.pkl', 'rb'))
        y_3=scaler_y3.inverse_transform(y_3)
        try:
            Class_3['8']=y_3[:,0]
            Class_3['9']=y_3[:,1]
            Class_3['10']=y_3[:,2]
        except:
            Class_3['8']=y_3[0]
            Class_3['9']=y_3[1]
            Class_3['10']=y_3[2]
        Classes=Classes.append(Class_3)

    Classes=Classes.rename(columns={"8": "length", "9": "width","10": "depth"})
    try:
        Tmin_Tmax_values=pd.DataFrame(Tmin_Tmax_values,columns=['Tmax', 'Tmin'])
    except:
        Tmin_Tmax_values = list(Tmin_Tmax_values)
        Tmin_Tmax_values = pd.DataFrame([Tmin_Tmax_values])
        Tmin_Tmax_values.columns =['Tmax','Tmin']
    Results_ET_NN=pd.DataFrame(data)
    Results_ET_NN=Results_ET_NN.join(Classes.iloc[:,8:],how='outer')
    Results_ET_NN=Results_ET_NN.join(Tmin_Tmax_values,how='outer')
    Results_ET_NN=Results_ET_NN.abs()
    #Results_ET_NN.to_csv('Results_NN.csv', index=False) #change name of output file
    
    return Results_ET_NN


# In[ ]:



#Analytical ET Model 

def analytical_ET(dimensionless_df):
    """ Main section of the script file """
    #  inputfile = os.getcwd() + '/ET_input' + sys.argv[1] + '.csv'
    #inputfile = os.getcwd() + '/ET_input_1.csv'
    # set process parameters and material properties
    
        
    data = pd.DataFrame()
    data['Elements'] = dimensionless_df['Elements']
    data['Atomic_frac'] = dimensionless_df['Atomic_frac']
    
    
    
    data['v']=dimensionless_df['Velocity_m/s']
    data['P'] = dimensionless_df['Power']
    data['twoSigma'] = dimensionless_df["Beam_diameter_m"]  #m
    data['A'] = dimensionless_df['Absorptivity']
    data['tMelt'] = dimensionless_df['T_liquidus']
    data['k'] = dimensionless_df['thermal_cond_liq']
    data['rho'] = dimensionless_df["Density_kg/m3"]
    data['cp'] = dimensionless_df['Cp_J/kg']
    #dimensionless_df['tMelt'] =dimensionless_df['T_liquidus'] 
    rows_with_nan = [index for index, row in data.iterrows() if row.isnull().any()]
    data = data.dropna(axis=0,how='any',inplace=False)
    elements = data['Elements']
    atomic = data['Atomic_frac']
    data = data.drop(['Elements','Atomic_frac'],axis=1)
    
    inputfile = data
    
    
    outputfile = os.getcwd() + '/ET_Cantor_package.csv'   # sys.argv[2]   ###Change name
    
    # ========== SIMULATION ====================================================
    # Define some of the simulation parameters
    # How big is the simulation domain?
    # Specify in microns [x, y, z]
    domain = np.array([1200.0, 1200.0, 1000.0])   # This will get increased if its not big enough
    # What spatial resolution is required?
    # This will determine the number of bins in the domain
    spatialRes = 1.0   # Microns - about 1 micron is as fine as you will need to go
    # ==========================================================================

    # ========== BEAM and MATERIAL =============================================
    # Define the beam and material parameters
    # These we will iterate through
    beam1, mat1 = beamFromCSV(inputfile)
    # ==========================================================================


    # Warn the user about which version of python/scipy they are using
    # It has implications on the type of integration used and the resulting speed
    integrationWarning()

    # Output section
    # Define the csv save options
    print('Saving to file ' + outputfile)
    
    # Set up the variables to save into
    runSize = np.size(beam1.v, axis=0)
    melt_length = np.zeros(runSize, dtype='f8')
    melt_width = np.zeros(runSize, dtype='f8')
    melt_depth = np.zeros(runSize, dtype='f8')
    peakT = np.zeros(runSize, dtype='f8')
    minT = np.zeros(runSize, dtype='f8')

    # Loop through the various beam parameters and simulate melt pool
    for i in np.arange(runSize):
        # Create the simulation parameter object
        sim1 = simParam(domain,spatialRes)
        
        melt_length[i], melt_width[i], melt_depth[i], peakT[i], minT[i] = eagarTsaiParam(beam1,mat1,sim1,i)
        print ('Completed run ' + str(i+1) + ' of ' + str(runSize))

    # Write Output File
    results = pd.DataFrame({'v': beam1.v,
                           'P': beam1.P,
                           'twoSigma': beam1.twoSigma,
                           'A': beam1.A,
                           'tMelt': mat1.tMelt,
                           'k': mat1.k,
                           'rho': mat1.rho,
                           'cp': mat1.cp,
                           'length': melt_length,
                           'width': melt_width,
                           'depth': melt_depth,
                           'Tmax': peakT,
                           'Tmin': minT})

    colnames = ['v','P','twoSigma','A','tMelt','k','rho','cp','length','width','depth','Tmax','Tmin']
#    if 'ET_input.csv' in inputfile:
#        results.to_csv(outputfile,index=False,header=True,mode='w',columns=colnames,float_format='%.6E')
#    else:
    #results.to_csv(outputfile,index=False,header=False,mode='a',columns=colnames,float_format='%.6E')
    return results

def beamFromCSV(inputfile):
    """ Import beam parameters from csv file """
    data = inputfile

	# These we will iterate through
    v = data['v'] # m/s
    P = data['P'] # Watts
    twoSigma = data['twoSigma'] # meters
    A = data['A'] # >0, <1
    tMelt = data['tMelt']
    k = data['k']
    rho = data['rho']
    cp = data['cp']
    
    # Create the beam object
    return beam(twoSigma,P,v,A), material(tMelt,k,rho,cp)

def eagarTsaiParam(beam,material,simParam,i):
    """ Function that runs each iteration of the EagarTsai Simulation """
    # This version of the code uses a re-formulation of EagarTsai
    # By Sasha Rubenchik - LLNL 2015
    
    # Unpack a few variables
    # Material
    tMelt = material.tMelt[i]
    k = material.k[i]
    rho = material.rho[i]
    cp = material.cp[i]
    alpha = k/(rho*cp)
    
    # Beam (this varies each run)
    P = beam.P[i]
    A = beam.A[i]
    v = beam.v[i]
    sigma = beam.sigma[i]
    
    # Simulation params
    delta = simParam.spatialRes
    
    # Now lets define the domain we are going to evaluate temperature over
    # Define the minimums and maximums
    xMin = round(-1.0 * beam.twoSigma[i] * 1.5, 5)
    xMax = simParam.domain[0]
    yMin = 0.0
    yMax = simParam.domain[1]
    zMin = -1.0 * simParam.domain[2]
    zMax = 0.0
    # Find the number of intervals
    nx = int(np.round(abs(xMax-xMin)/delta))+1
    ny = int(np.round(abs(yMax-yMin)/delta))+1
    nz = int(np.round(abs(zMax-zMin)/delta))+1
    # Create the range arrays
    nxrange = np.linspace(xMin,xMax,nx)
    nyrange = np.linspace(yMin,yMax,ny)
    nzrange = np.linspace(zMin,zMax,nz)
    
    # We need to do some checking about which method to use
    # We need to see which version of scipy we are running
    # The compiled integral will only work on scipy 0.15.1 and later
    sciVer = scipy.version.version.split(".")
    # Find the platform type
    osType = sys.platform
    
    if int(sciVer[1]) >= 1:
        # Then we can use compiled code to run the integration
        if osType == 'darwin':   # Then you're cool because that's a Mac!
            libpath = 'libsasha.dylib'
    
        elif osType in 'linux2':   # It's linux, I guess that's OK
            libpath = './libsasha.so'
        
        elif (osType == 'win32') or (osType == 'cygwin'):
            # Sorry, windows sucks, run the integration using normal interpreted code
            libpath = 0

    else:
        # Old version of SciPy and possibly python
        libpath = 0
        
    # Run integral
    tplanexy, tplanexz = runIntegrate(nxrange,nyrange,nzrange,nx,ny,nz,alpha,sigma,k,v,A,P,libpath)

    # Find the peak temperature
    peakT = np.amax(tplanexy)
    minT = np.amin(tplanexy)
    # Now check to see if the peak temperature is hotter than Tmelt
    if peakT > tMelt:
        # Then there is a melt pool, find the size
        # Section to actually extract the metrics regarding melt pool
        # From the XY Plane
        # Want to find the length
        meltXInd = np.squeeze(np.where(tplanexy[0,:] > tMelt))
        melt_length = np.amax(nxrange[meltXInd])-np.amin(nxrange[meltXInd])
        melt_trail_length = abs(np.amin(nxrange[meltXInd]))
        
        # Now want to find the width + depth (can do it in same outer loop)
        yLength = 0
        zLength = 0
        for i1 in np.arange(np.size(meltXInd,axis=0)):
            meltYInd = np.squeeze(np.where(tplanexy[:,meltXInd[i1]] > tMelt))
            tmpYLength = np.amax(nyrange[meltYInd]) - np.amin(nyrange[meltYInd])
            if tmpYLength > yLength:
                yLength = tmpYLength
            
            meltZInd = np.squeeze(np.where(tplanexz[:,meltXInd[i1]] > tMelt))
            tmpZLength = np.amax(nzrange[meltZInd]) - np.amin(nzrange[meltZInd])
            if tmpZLength > zLength:
                zLength = tmpZLength
		
        # Test to see if the domain is the correct size
        if np.isclose(np.amax(nxrange[meltXInd]),xMax):
            # Then the x domain is not long enough
            print ("The x domain (length) is not large enough, increasing size and re-running")
            simParam.domain[0] += sigma
            # Re-run the analysis
            # melt_width, melt_depth, melt_length = eagarTsaiParam(beam,material,simParam,i)
            melt_length, melt_width, melt_depth = eagarTsaiParam(beam,material,simParam,i)
        elif np.isclose(yLength,abs(yMax - yMin)):
            # Then the y domain is not large enough
            print ("The y domain (width) is not large enough, increasing size and re-running")
            simParam.domain[1] += sigma
            # Re-run the analysis
            melt_length, melt_width, melt_depth = eagarTsaiParam(beam,material,simParam,i)
        elif np.isclose(zLength,abs(zMax-zMin)):
            print ("The z domain (depth) is not large enough, increasing size and re-running")
            simParam.domain[2] += sigma
            # Re-run the analysis
            melt_length, melt_width, melt_depth = eagarTsaiParam(beam,material,simParam,i)
        else:
            # All is good, the melt pool wasnt clipped in the domain
            # Return the values
            melt_width = yLength * 2
            melt_depth = zLength
    else:
        # Then there is no melt pool, return 0 lengths
        melt_width = 0.0
        melt_depth = 0.0
        melt_length = 0.0

    del tplanexy, tplanexz
    return melt_length, melt_width, melt_depth, peakT, minT

def runIntegrate(nxrange,nyrange,nzrange,nx,ny,nz,alpha,sigma,k,v,A,P,libpath):
    """ Function to run the integration """
    # Check if using compiled or interpreted code
    if libpath == 0:
        # using interpreted code
        func = sasha_int
    else:
        # using compiled code
        lib = ctypes.CDLL(libpath) # Use absolute path to shared library
        func = lib.f # Assign specific function to name func (for simplicity)
        func.restype = ctypes.c_double
        func.argtypes = (ctypes.c_int, ctypes.c_double)

    # Set up invariant parameters
    # Define the starting temperature
    t0 = 300.0 # Kelvin
    Ts = (A*P)/(np.pi*(k/alpha)*np.sqrt(np.pi*alpha*v*(sigma**3)))
    p = alpha/(v*sigma)
    
    # Create the two temperature planes
    tplanexy = np.zeros(nx*ny,dtype='f8').reshape(ny,nx)
    tplanexz = np.zeros(nx*nz,dtype='f8').reshape(nz,nx)
    
    # Run the integration
    for i1 in np.arange(nx):
        x = nxrange[i1]/sigma # make dimensional
        for i2 in np.arange(ny):
            y = nyrange[i2]/sigma # make dimensional
            tmpTemp = quad(func,0.,np.inf,args=(x, y, 0.0, p))
            tplanexy[i2,i1] = t0 + Ts*tmpTemp[0]
        for i3 in np.arange(nz):
            z = nzrange[i3]/np.sqrt((alpha*sigma)/v) # make dimensional
            tmpTemp = quad(func,0.,np.inf,args=(x, 0.0, z, p))
            tplanexz[i3,i1] = t0 + Ts*tmpTemp[0]

    # Return the temperature planes
    return tplanexy, tplanexz

def sasha_int(t, x, y, z, p):
    # This is Sasha's formulation
    intpre = 1.0/((4*p*t + 1)*np.sqrt(t))
    intexp = (-(z**2)/(4*t))-(((y**2)+(x-t)**2)/(4*p*t + 1))
    return intpre * np.exp(intexp)

class simParam():
    def __init__(self,domain,spatialRes):
        """ Define a simulation parameter object """
        self.domain = domain / 1.e6 # Convert into meters
        self.spatialRes = spatialRes / 1.e6 # Convert into meters

class beam():
    def __init__(self,twoSigma,P,v,A):
        """ Define a beam object """
        self.twoSigma = twoSigma
        # The next sigma is an altered version for Sasha's ET re-interp
        self.sigma = np.sqrt(2.0) * (self.twoSigma / 2.0)
        self.P = P
        self.v = v
        self.A = A

class material():
    def __init__(self,tMelt,k,rho,cp):
        """ Define a material object """
        self.tMelt = tMelt
        self.k = k
        self.rho = rho
        self.cp = cp

def integrationWarning():
    """ Function that warns the user about the integration type """
    # We need to do some checking about which method to use
    # We need to see which version of scipy we are running
    # The compiled integral will only work on scipy 0.15.1 and later
    sciVer = scipy.version.version.split(".")
    
    # Find the platform type
    osType = sys.platform
    if int(sciVer[1]) >= 1:
        # Then we can use compiled code to run the integration
        if osType == 'darwin': # Then you're cool because that's a Mac!
            print ("Using compiled integration code, should be faster")
        elif osType in 'linux2': # It's linux, I guess that's OK
            print ("Using compiled integration code, should be faster")
        elif (osType=='win32') or (osType=='cygwin'):
            # Sorry, windows sucks, run the integration using normal interpreted code
            print ("Using interpreted integration code, will be slow")
            print ("Consider switching to a Mac or Linux")
    else:
        print ("Using old version of Python and SciPy")
        print ("Please consider switching to Python 2.7.10 and SciPy 0.15.1")
        print ("Using interpreted integration code, will be slow")


# In[ ]:


def GS_depth (dimensionless_df_row):
    A = dimensionless_df_row["Absorptivity"]
    P = dimensionless_df_row["Power"]
    v = dimensionless_df_row['Velocity_m/s']
    k = dimensionless_df_row['thermal_cond_liq']
    T_b = dimensionless_df_row['T_b']
    thermal_diff = dimensionless_df_row['thermal_diff_liq']
    beam_size = dimensionless_df_row['Beam_radium_m']

    gs_d = ((A*P)/(2*np.pi*T_b))*(np.log((beam_size + (thermal_diff/v))/(beam_size)))
    
    return gs_d


def keyholing_normalized(dimensionless_df, T_0 = [298]):


    for i in dimensionless_df.index: 
        P = dimensionless_df.loc[i,'Power']
        absorp = dimensionless_df.loc[i,'Absorptivity']
        T_liquidus = dimensionless_df.loc[i,'T_liquidus']
        T_sub = T_0
        Cp = dimensionless_df.loc[i,'Cp_J/kg']
        thermal_diff = dimensionless_df.loc[i,'thermal_diff_liq']
        density = dimensionless_df.loc[i,'Density_liq_kg/m3']
        vel = dimensionless_df.loc[i,'Velocity_m/s']
        beam_rad = dimensionless_df.loc[i,'Beam_radium_m']
        
        
        Ke_num = absorp*P
        Ke_denom = (T_liquidus - T_sub)* np.pi*density*Cp*(np.sqrt(thermal_diff*vel*beam_rad**3))
        Ke = Ke_num/Ke_denom
        dimensionless_df.at[i,'Ke'] = Ke
        
        dimensionless_df.at[i,'norm_key_depth'] = 0.4*(Ke-1.4)
        
        var_key_depth  = 0.36*(Ke**0.86)
        dimensionless_df.at[i,'Var_Key_deph'] = var_key_depth
        
        thermal_diffusion_length = np.sqrt((thermal_diff*beam_rad)/vel)
        dimensionless_df.at[i,'Thermal_Diffusion_Length'] = thermal_diffusion_length
        
        norm_diffusion_length = thermal_diffusion_length/beam_rad
        dimensionless_df.at[i,'Norm_Diffusion_Length'] = norm_diffusion_length
        
    return dimensionless_df    

def cooling_rate(dimensionless_df,T_0=[298]):
    
    for i in dimensionless_df.index:
        Power = dimensionless_df.loc[i,'Power']
        absorptivity = dimensionless_df.loc[i,'Absorptivity']
        thermal_cond = dimensionless_df.loc[i,'thermal_cond_liq']
        T_solidus = dimensionless_df.loc[i,'T_solidus']
        T_sub = T_0
        T_liquidus = dimensionless_df.loc[i,'T_liquidus']
        vel = dimensionless_df.loc[i,'Velocity_m/s']
        
        Qp = Power * absorptivity
        
        cooling_rate = 2*np.pi*thermal_cond*(T_solidus-T_sub)*(T_liquidus-T_sub)*(vel/Qp)
        dimensionless_df.at[i,'Cooling_rate'] = cooling_rate
    return dimensionless_df



def keyholing_criteria(dimensionless_df):
    for i in dimensionless_df.index:
        T_boil = dimensionless_df.loc[i,'T_b']
        T_m = dimensionless_df.loc[i,'T_liquidus']
        Ke = dimensionless_df.loc[i,'Ke']
        d = dimensionless_df.loc[i,'depth']
        w = dimensionless_df.loc[i,'width']
        
        H_normalized = dimensionless_df.loc[i,'H_normalized']
    
        criteria = (np.pi*T_boil)/T_m
        
        if H_normalized > criteria:
            dimensionless_df.at[i,'Keyholing_KH2'] = 1.0
        else:
            dimensionless_df.at[i,'Keyholing_KH2'] = 0.0
            
        
        if Ke >= 6.0:
            dimensionless_df.at[i,'Keyholing_KH3'] = 1.0
        else:
            dimensionless_df.at[i,'Keyholing_KH3'] = 0.0   
            
            
        for key_value in dim_key_value:
            print(key_value)
            if d >= (w / key_value):
                dimensionless_df[f'Keyholing_{key_value}'] = 1.0
            else:
                dimensionless_df[f'Keyholing_{key_value}'] = 0.0
            
    return dimensionless_df


def lof_criteria(dimensionless_df): #using depth after normalized keyholing
    for idx in dimensionless_df.index:
        d = dimensionless_df.loc[idx,'depth_KH2_corrected']
        
        w = dimensionless_df.loc[idx,'width']
        h = dimensionless_df.loc[idx,'Hatch_spacing_m']
        t = dimensionless_df.loc[idx,'Powder_thick_m']
        
        try: 
            criteria = ((h/w)**2)+(t/(t+d))
            if criteria >= 1:
                dimensionless_df.at[idx,'LOF2_KH2']=1
            else:
                dimensionless_df.at[idx,'LOF2_KH2']=0
        except Exception as e1:
            dimensionless_df.at[idx,'LOF2_KH2']=np.nan 
            
    for value in dim_key_value:
        depth_column_name = f'depth_KH1_corr_{value}'
        for idx in dimensionless_df.index: #using depth after dimensional
            d = dimensionless_df.loc[idx, depth_column_name]
            
            w = dimensionless_df.loc[idx, 'width']
            h = dimensionless_df.loc[idx, 'Hatch_spacing_m']
            t = dimensionless_df.loc[idx, 'Powder_thick_m']
            
            try: 
                criteria = ((h/w)**2) + (t/(t+d))
                if criteria >= 1:
                    dimensionless_df.at[idx, f'LOF2_KH1_{value}'] = 1
                else:
                    dimensionless_df.at[idx, f'LOF2_KH1_{value}'] = 0
            except Exception as e1:
                dimensionless_df.at[idx, f'LOF2_KH1_{value}'] = np.nan  
            
    for idx in dimensionless_df.index: #lof after keyholing Ke defined 
        d = dimensionless_df.loc[idx,'depth_KH3']
        
        w = dimensionless_df.loc[idx,'width']
        h = dimensionless_df.loc[idx,'Hatch_spacing_m']
        t = dimensionless_df.loc[idx,'Powder_thick_m']
        
        try: 
            criteria = ((h/w)**2)+(t/(t+d))
            if criteria >= 1:
                dimensionless_df.at[idx,'LOF2_KH3']=1
            else:
                dimensionless_df.at[idx,'LOF2_KH3']=0
        except Exception as e1:
            dimensionless_df.at[idx,'LOF2_KH3']=np.nan             
    return dimensionless_df


def balling(dimensionless_df,T_amb=[288]):
    
    # Initialize Columns with NANs (empty)
    dimensionless_df['Solidification/Spread Time'] = np.nan
    dimensionless_df['Spreading Time (\u03BCs)'] = np.nan
    dimensionless_df['Solidification Time (\u03BCs)'] = np.nan
    dimensionless_df['Base/Initial Radius'] = np.nan
    dimensionless_df['Balling'] = np.nan
    dimensionless_df['Base Radius (\u03BCm)'] = np.nan
    dimensionless_df['Tau1'] = np.nan
    dimensionless_df['Tau2'] = np.nan
    
    for amb_temp in T_amb:
        for idx in dimensionless_df.index:
            # redefine variables
            Ta = amb_temp
            To = dimensionless_df.loc[idx,"T_liquidus"]
            Tf = dimensionless_df.loc[idx,"T_solidus"]
            L = dimensionless_df.loc[idx,'Latent_Heat']
            p = dimensionless_df.loc[idx,'Density_liq_kg/m3']  # convert to kg/m^3 for units to cancel
            c = dimensionless_df.loc[idx,'Cp_liq']
            A = dimensionless_df.loc[idx,'thermal_diff_liq'] #should be LT
            k = dimensionless_df.loc[idx,'thermal_cond_liq']
            s = dimensionless_df.loc[idx,'surf_tens_liq']
            a = 100e-06  # meters, initial droplet radius
            ka = dimensionless_df.loc[idx,'thermal_cond_RT']

            # Tau1 Removing superheat from droplet
            tau_1 = ((a ** 2 * k) / (3 * A * ka)) * np.log((To - Ta) / (Tf - Ta))

            # Tau2 Removing latent heat of fusion and solidify
            tau_2 = (((a ** 2 * k) / (3 * A * ka)) * (1 + ka / (2 * k)) * (L / (c * (Tf - Ta))))

            # Solidification time
            t_solid = 2 * (tau_1 + tau_2)

            # Spreading time
            t_spread = (p * a**3 / s)**0.5

            # Droplet base radius after solidification
            radius_b = a * 2.4 * (1 - np.exp(-0.9 * t_solid * (p * a ** 3 / s) ** -0.5))

            dimensionless_df.at[idx, 'Tau1'] = tau_1
            dimensionless_df.at[idx, 'Tau2'] = tau_2
            dimensionless_df.at[idx, 'Solidification Time (\u03BCs)'] = t_solid * 1e06
            dimensionless_df.at[idx, 'Spreading Time (\u03BCs)'] = t_spread * 1e06
            dimensionless_df.at[idx, 'Solidification/Spread Time'] = t_solid / t_spread
            dimensionless_df.at[idx, 'Base Radius (\u03BCm)'] = radius_b * 1e06
            dimensionless_df.at[idx, 'Base/Initial Radius'] = radius_b / a

            if radius_b/a < 1.2599:
                #Balling is true
                dimensionless_df.at[idx, 'Balling'] = 1.0
            else:
                #Balling is false
                dimensionless_df.at[idx, 'Balling'] = 0.0
                
            # You can save every 1,2,3... alloys by changing how you check the divisibility of i
#            if idx % 1 == 0:
#                results_df.to_hdf(savename + '.h5', key='df', mode='w')
#                results_df.to_excel(savename + '.xlsx')
#                print('Saved at i=', i)
    return dimensionless_df



#hot cracking criteria Function Function 8
def hot_cracking (results_df):
    hot_cracking = []
    fs = []
    Temp = []
    for idx in results_df.index:
        for label in scheil_curve:
            print(label)
            scheil_curve = results_df.loc[idx,'scheil_curve']
            section = scheil_curve[label]
            fs.extend(section.x)
            Temp.extend(np.array(section.y) - 273.15)
            #plt.plot(section.x, np.array(section.y) - 273.15, label=label)
            #plt.show()
        fs = np.array(fs)
        valid_indices = np.where((fs < .98)&(fs> .8)) # Hyperparameters
        Temp = np.array(Temp)
        dTemp = Temp[-1] - Temp[0]
        dsrtfs = np.sqrt(fs[-1]) - np.sqrt(fs[0])

        Kou_Criteria = np.abs(dTemp/dsrtfs)
        hot_cracking.append(Kou_criteria)

        return results_df 
    
##########################################################################################################################################

#USER INPUTS

if __name__ == '__main__':
    
    ##########################################################################
    ##########################################################################
    print("Which E-T model should be used: analytical,NN or scaled? ")
    #define your et model
    e_t_model_type = 'NN'
    
    #Upload Thermocalc calculations   
    PATH = os.getcwd()
	
    thermo_calc_df = pd.read_csv(PATH + '/THERMOCALC_Files/' 'THERMOCALC_TOTAL.csv')
    #upload composition file
    file_name = 'Database_Al'
    composition_df = pd.read_csv(file_name + '.csv')
    
    #define processing parameters 
    #Processing Params
    #power_w = list(np.arange(80,110,5))
    #vel_ms = list(np.arange(0.04,2.5,0.01))
    power_w = list(np.arange(20,401,1))
    vel_ms = list(np.arange(0.02,3.001,0.001))    
    powder_thickness = [40]
    hatch_spacing = [80]
    d_laser = [80]
    powder_grain_size = [20] #um
    laser_wavelength_nm = [1070] #nm
    T_amb = [288]
    
    dim_key_value = [1.8]
    dim_ball_value = [2.0]
    
    
    

    
    #for 100 points need 4 composition points and 5*5 grid for power and velocity with everything else is constant 
    ##########################################################################
    print("Files Uploaded")
    
    #create cobmination sheet for processing parameters called param
    #parameters sheet
    params=pd.DataFrame(list(product(power_w, vel_ms,powder_thickness,hatch_spacing,d_laser,powder_grain_size,laser_wavelength_nm,T_amb)), columns=['Power', 'Velocity_m/s','Powder_thickness_um','hatch_spacing_um','d_laser_um','powder_grain_size_um','laser_wavelength_nm','amb_temp_K'])
    #params = pd.read_csv('parameter_set.csv')
    
    
    #obtain a full only weight ratio composition and atomic ratio composition
    
    
    
    composition_df=composition_df.fillna(0)
    composition_df = composition_df.reset_index(drop=True)
    
    #create composition object 
    df_sep_elements = composition_df.copy()
    df_sep_elements = df_sep_elements.drop(['Comp_point','Unit [at% or wt%]','Unnamed: 0'],axis=1)
    cols = df_sep_elements.columns
    
    df_dropped_cols_element=df_sep_elements.drop(df_sep_elements.loc[:, (df_sep_elements==0).mean() == 1.0],axis=1)
    print(df_dropped_cols_element.head()) #42 columns 
    comp_el = np.array(df_dropped_cols_element.to_numpy())
    comp_el.shape
    comp_el = comp_el*0.01 #create composition rows 
    comp_el
    
    non_zero = df_sep_elements.apply(lambda x: x > 0)
    df_sep_elements['Element_present']= non_zero.apply(lambda x: list(cols[x.values]), axis=1)
    df_sep_elements['Element_present'].astype(str)
    df_sep_elements=df_sep_elements.fillna(0)
    df_sep_elements.reset_index()
    print("There are Nans in dataframe:",df_sep_elements.isnull().values.any())
    
    elements_col =cols
    elements_col
    len(comp_el)#split by this number to get array for each row 
    seperate_array = np.vsplit(comp_el,len(comp_el)) 
    #remove zeros
    remove_zeros_arr = [] 
    for i in range(0,len(seperate_array)):
        comp_ratio_arr= seperate_array[i][seperate_array[i] != 0.]
        remove_zeros_arr.append(comp_ratio_arr)
    df_sep_elements['composition_ratio'] = remove_zeros_arr
    
    df_sep_elements['composition_ratio'] = remove_zeros_arr
    element_list=df_sep_elements['Element_present'].to_numpy()
    comp_ratio = df_sep_elements['composition_ratio'].to_numpy()
    df_sep_elements['Units_Comp'] = composition_df['Unit [at% or wt%]']
    
    wt_per = []
    at_per = []
    for i in range(len(composition_df.index)):
        wt = 'wt'
        at = 'at'
        element_list_joined = []
        for ele in element_list:
            element_list_joined.append(''.join(ele))
        comp = element_list_joined[i]
        comp_obj = Composition(comp)
        element_obj_list = comp_obj.elements
        elemental_mass_arr = np.array([element_obj.atomic_mass for element_obj in element_obj_list]) #mass of each element
        if at in composition_df.loc[i,'Unit [at% or wt%]']:
            comp = df_sep_elements.loc[i,'composition_ratio']
            comp = np.around(comp,4)
            at = (df_sep_elements.loc[i,'composition_ratio'])*100
            num = at*elemental_mass_arr
            total_wt = sum(num)
            wt_frac_arr = np.array(num/total_wt)
            wt_frac_arr = np.around(wt_frac_arr,4)
            at_per.append(comp*100)
            wt_per.append(wt_frac_arr*100)
        elif wt in composition_df.loc[i,'Unit [at% or wt%]'] :
            comp = df_sep_elements.loc[i,'composition_ratio']
            comp = np.around(comp,4)
            wt = (df_sep_elements.loc[i,'composition_ratio'])*100
            moles = wt/elemental_mass_arr
            total_moles = sum(moles)
            atomic_fraction_arr = np.array(moles/total_moles)
            atomic_fraction_arr = np.around(atomic_fraction_arr,4)
            at_per.append(atomic_fraction_arr*100)
            wt_per.append(comp*100)
        else:
            print('Please recheck you units, they are not in atomic or weight')
    df_sep_elements['At%'] = at_per
    df_sep_elements['Wt%'] = wt_per
    
    df_sep_elements = df_sep_elements.drop(['composition_ratio','Units_Comp'],axis=1)        
    
    
    elements_col = []
    for col in composition_df.columns:
        elements_col.append(col)
    elements_col = elements_col[3:]
    
    atomic_arr = df_sep_elements['At%']
    
    #CBFV
    # =============================================================================
    # CBFV Features
    # =============================================================================
    
    PATH = os.getcwd()
    #NEW_PATH = os.chdir(PATH+'/cbfv')
    sys.path.append(PATH+'/cbfv')
    import composition
    
    
    prop_df = df_sep_elements.drop(columns=df_sep_elements.columns[-3:])
    # remove spaces character
    prop_df.columns = prop_df.columns.str.replace(' ', '')
    
    formula_list=[]
    
    #Add column for chemical 'formula'
    for x in range(0,len(element_list)):
        formula_str = []
        comp_formula_list = []
        elements = element_list[x]
        for j in range(len(elements)):
            ele = element_list[x][j]
            ratio = str(atomic_arr[x][j])
            formula_str_ele = ele+ratio
            formula_str.append(formula_str_ele)
            listToStr = ' '.join([str(elem) for elem in formula_str])
        formula_list.append(listToStr)
    
    comp_formula = []
    for i in formula_list:
        comp=i.replace(" ", "")
        comp_formula.append(comp)
    
    prop_df['target'] =np.nan
    prop_df['formula']=comp_formula
    input_df = prop_df[['formula','target']]
    elem_prop_name = 'oliynyk' #magpie = 66, oliynyk = 132, jarvis = 1314
    
    feats,y, formulae_train, skipped_train  = composition.generate_features(input_df, elem_prop=elem_prop_name, drop_duplicates=False, extend_features=False, sum_feat=False)
    feats
    
    os.chdir(PATH)
    print("Change back to original directory")
    results_df = feats.copy()
    results_df['Elements_active'] = element_list
    results_df['atomic_per'] = atomic_arr
    
    
    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        thread = executor.submit(ROM_THERMO,results_df)
        results_df = thread.result()
    
    #concat results_df to thermocalc 
    materials_features = thermo_calc_df.join(results_df) 
    
    print('ROM_THERMO Complete')
    
    
    print("Running Melt Pool/Dimensionless Function")
    
    results_df = materials_features 
    data = {'par':params,'mat':results_df}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        thread = executor.submit(melt_pool_dimensionless,data)
        params,dimensionless_df  = thread.result()
     
    print("Melt Pool Dimensionless Function Complete -- Save all files")
    
    params.to_csv('parameters_' + file_name +'_' + e_t_model_type + '.csv')
    #dimensionless_df.to_csv('dimensionless_df_' + file_name + '_' + e_t_model_type + '.csv')
    


    #ET Models if statement 
    
    if e_t_model_type == 'scaled':
        print("Running scaled ET model")
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            thread = executor.submit(scaled_ET,dimensionless_df)
            dimensionless_df = thread.result()
    elif e_t_model_type == 'NN':
        os.chdir(PATH + '/ET_Models/ET_NN_Nov11')
        print("Running NN ET Model")
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            thread = executor.submit(ET_NN,dimensionless_df)
            ET_NN = thread.result()
            #ET_NN.to_csv('ET_' + file_name +'_' + e_t_model_type + '.csv')
        #add length width and depth to the dimensionless_df spreadsheet 
            #concat length, width, depth, Tmax and Tmin columns to dimensionless_df 
        dimensionless_df["length"] = ET_NN["length"]
        dimensionless_df["depth"] = ET_NN["depth"]
        dimensionless_df["width"] = ET_NN['width']
        dimensionless_df["Tmax"] = ET_NN["Tmax"]
        dimensionless_df["Tmin"] = ET_NN["Tmin"]
    elif e_t_model_type == 'analytical':
        os.chdir(PATH + '/ET_Models/Analytical_ET')
        print("Running analytical ET Model")
        with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
            thread = executor.submit(analytical_ET,dimensionless_df)
            ET_Ana = thread.result()
        #ET_Ana.to_csv('ET_' + file_name +'_' + e_t_model_type + '.csv')
        dimensionless_df["length"] = ET_Ana["length"]
        dimensionless_df["depth"] = ET_Ana["depth"]
        dimensionless_df["width"] = ET_Ana['width']
        dimensionless_df["Tmax"] = ET_Ana["Tmax"]
        dimensionless_df["Tmin"] = ET_Ana["Tmin"]
    else:
        print('Insert scaled, analytical or NN')

    dimensionless_df.to_csv('ET_' + file_name +'_' + e_t_model_type + '.csv')        
    os.chdir(PATH)
    
### Cooling rate

    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        thread = executor.submit(cooling_rate,dimensionless_df)
        dimensionless_df = thread.result()          

### Keyholing 
    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        thread = executor.submit(keyholing_normalized,dimensionless_df)
        dimensionless_df = thread.result()
        
    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        thread = executor.submit(keyholing_criteria,dimensionless_df)
        dimensionless_df = thread.result()
        

## Update depth column - normalized enthalpy 
    dimensionless_df['depth_KH2_corrected'] = np.nan
    for i in dimensionless_df.index:
        dimensionless_df_row = dimensionless_df.iloc[i]
        if dimensionless_df.loc[i,'Keyholing_KH2'] == 1.0:
            with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
                thread = executor.submit(GS_depth,dimensionless_df_row)
                gs_d = thread.result()
            dimensionless_df.at[i,'depth_KH2_corrected'] = gs_d
        else:
            dimensionless_df.at[i,'depth_KH2_corrected'] = dimensionless_df.loc[i,'depth']
## Update depth column - dimensional keyholing values 
       
    for value in dim_key_value:
        kh_column_name = f'Keyholing_{value}'
        dimensionless_df[f'depth_KH1_corr_{value}'] = np.nan
    
    
# Iterate through each row in the DataFrame
    for i in dimensionless_df.index:
        dimensionless_df_row = dimensionless_df.iloc[i]
        # Iterate through each dim_key_value
        for value in dim_key_value:
            kh_column_name = f'Keyholing_{value}'
            # Check if Keyholing condition is met for the current dim_key_value
            if dimensionless_df_row[kh_column_name] == 1.0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    thread = executor.submit(GS_depth, dimensionless_df_row)
                    gs_d = thread.result()
                depth_column_name = f'depth_KH1_corr_{value}'
                dimensionless_df.at[i, depth_column_name] = gs_d
            else:
                dimensionless_df.at[i, f'depth_KH1_corr_{value}'] = dimensionless_df_row['depth']


    dimensionless_df['depth_KH3'] = np.nan
    for i in dimensionless_df.index:
        dimensionless_df_row = dimensionless_df.iloc[i]
        if dimensionless_df.loc[i,'Keyholing_KH3'] == 1.0:
            with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
                thread = executor.submit(GS_depth,dimensionless_df_row)
                gs_d = thread.result()
            dimensionless_df.at[i,'depth_KH3'] = gs_d
        else:
            dimensionless_df.at[i,'depth_KH3'] = dimensionless_df.loc[i,'depth']
            
##LOF 
    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        thread = executor.submit(lof_criteria,dimensionless_df)
        dimensionless_df = thread.result()  
        
# Balling 
    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        thread = executor.submit(balling,dimensionless_df)
        dimensionless_df = thread.result() 
            
 #Defining All Criteria Combinations and Defining Final Failure Mode
 
#Define the final criteria set 12 (2,5,7)
  #type of failure
    index = dimensionless_df.index
    for i in range(len(index)):
        if dimensionless_df.at[i,'Keyholing_KH3'] == 1:
            dimensionless_df.at[i,'failure_mode_LOF2KH3Ball2'] = 'Keyholing'
        elif ((np.pi*dimensionless_df.at[i,'width'])/dimensionless_df.at[i,'length']) < (np.sqrt(2/3)):
          dimensionless_df.at[i,'failure_mode_LOF2KH3Ball2'] = 'Balling'
        elif dimensionless_df.at[i,'LOF2_KH3'] == 1:
            dimensionless_df.at[i,'failure_mode_LOF2KH3Ball2'] = 'LOF'
        else:
            dimensionless_df.at[i,'failure_mode_LOF2KH3Ball2'] = 'Success'
 
 
 
#Define the final criteria set 11 (2,5,6)
    #type of failure
    for dim_ball in dim_ball_value:
      index = dimensionless_df.index
      for i in range(len(index)):
          if dimensionless_df.at[i,'Keyholing_KH3'] == 1:
              dimensionless_df.at[i,f'failure_mode_LOF2KH3Ball1_{dim_ball}'] = 'Keyholing'
          elif dimensionless_df.at[i,'length']/dimensionless_df.at[i,'width'] >= dim_ball :
            dimensionless_df.at[i,f'failure_mode_LOF2KH3Ball1_{dim_ball}'] = 'Balling'
          elif dimensionless_df.at[i,'LOF2_KH3'] == 1:
              dimensionless_df.at[i,f'failure_mode_LOF2KH3Ball1_{dim_ball}'] = 'LOF'
          else:
              dimensionless_df.at[i,f'failure_mode_LOF2KH3Ball1_{dim_ball}'] = 'Success'
       
#Define the final criteria set 10 (2,4,7)
    #type of failure
    index = dimensionless_df.index
    for i in range(len(index)):
        if dimensionless_df.at[i,'Keyholing_KH2'] == 1:
            dimensionless_df.at[i,'failure_mode_LOF2KH2Ball2'] = 'Keyholing'
        elif ((np.pi*dimensionless_df.at[i,'width'])/dimensionless_df.at[i,'length']) < (np.sqrt(2/3)):
            dimensionless_df.at[i,'failure_mode_LOF2KH2Ball2'] = 'Balling'
        elif dimensionless_df.at[i,'LOF2_KH2'] == 1:
            dimensionless_df.at[i,'failure_mode_LOF2KH2Ball2'] = 'LOF'
        else:
            dimensionless_df.at[i,'failure_mode_LOF2KH2Ball2'] = 'Success'

            
# Define the final criteria set 9 (2,4,6)
    for dim_ball in dim_ball_value:
        index = dimensionless_df.index
        for i in range(len(index)):
            if dimensionless_df.at[i, 'Keyholing_KH2'] == 1:
                dimensionless_df.at[i, f'failure_mode_LOF2KH2Ball1_{dim_ball}'] = 'Keyholing'
            elif dimensionless_df.at[i, 'length'] / dimensionless_df.at[i, 'width'] >= dim_ball:
                dimensionless_df.at[i, f'failure_mode_LOF2KH2Ball1_{dim_ball}'] = 'Balling'
            elif dimensionless_df.at[i, 'LOF2_KH2'] == 1:
                dimensionless_df.at[i, f'failure_mode_LOF2KH2Ball1_{dim_ball}'] = 'LOF'
            else:
                dimensionless_df.at[i, f'failure_mode_LOF2KH2Ball1_{dim_ball}'] = 'Success'

# Define the final criteria set 8 (2,3,7)
    # Type of failure
    for dim_key in dim_key_value:
        index = dimensionless_df.index
        for i in range(len(index)):
            if dimensionless_df.at[i, f'Keyholing_{dim_key}'] == 1:
                dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball2_{dim_key}'] = 'Keyholing'
            elif ((np.pi*dimensionless_df.at[i,'width'])/dimensionless_df.at[i,'length']) < (np.sqrt(2/3)):
                dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball2_{dim_key}'] = 'Balling'
            elif dimensionless_df.at[i, f'LOF2_KH1_{dim_key}'] == 1:
                dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball2_{dim_key}'] = 'LOF'
            else:
                dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball2_{dim_key}'] = 'Success'

# Define the final criteria set 7 (2,3,6)
    # Type of failure
    for dim_key in dim_key_value:
        for dim_ball in dim_ball_value:
            index = dimensionless_df.index
            for i in range(len(index)):
                if dimensionless_df.at[i, f'Keyholing_{dim_key}'] == 1:
                    dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball1_{dim_key}_{dim_ball}'] = 'Keyholing'
                elif dimensionless_df.at[i,'length'] / dimensionless_df.at[i,'width'] >= dim_ball:
                    dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball1_{dim_key}_{dim_ball}'] = 'Balling'
                elif dimensionless_df.at[i, f'LOF2_KH1_{dim_key}'] == 1:
                    dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball1_{dim_key}_{dim_ball}'] = 'LOF'
                else:
                    dimensionless_df.at[i, f'failure_mode_LOF2KH1Ball1_{dim_key}_{dim_ball}'] = 'Success'


#Define the final criteria set 6 (1,5,7)
    #type of failure
    index = dimensionless_df.index
    for i in range(len(index)):
        if dimensionless_df.at[i,'Keyholing_KH3'] == 1:
            dimensionless_df.at[i,'failure_mode_LOF1KH3Ball2'] = 'Keyholing'
        elif ((np.pi*dimensionless_df.at[i,'width'])/dimensionless_df.at[i,'length']) < (np.sqrt(2/3)):
            dimensionless_df.at[i,'failure_mode_LOF1KH3Ball2'] = 'Balling'
        elif dimensionless_df.at[i,'depth_KH2_corrected'] <= dimensionless_df.at[i,'Powder_thick_m'] :
            dimensionless_df.at[i,'failure_mode_LOF1KH3Ball2'] = 'LOF'
        else:
            dimensionless_df.at[i,'failure_mode_LOF1KH3Ball2'] = 'Success'
 
 
 
# Define the final criteria set 5 (1,5,6)
    for dim_ball in dim_ball_value:
        for i in range(len(dimensionless_df)):
            if dimensionless_df.at[i, 'Keyholing_KH3'] == 1:
                dimensionless_df.at[i, f'failure_mode_LOF1KH3Ball1_{dim_ball}'] = 'Keyholing'
            elif dimensionless_df.at[i, 'length'] / dimensionless_df.at[i, 'width'] >= dim_ball:
                dimensionless_df.at[i, f'failure_mode_LOF1KH3Ball1_{dim_ball}'] = 'Balling'
            elif dimensionless_df.at[i, 'depth_KH3'] <= dimensionless_df.at[i, 'Powder_thick_m']:
                dimensionless_df.at[i, f'failure_mode_LOF1KH3Ball1_{dim_ball}'] = 'LOF'
            else:
                dimensionless_df.at[i, f'failure_mode_LOF1KH3Ball1_{dim_ball}'] = 'Success'
       
#Define the final criteria set 4 (1,4,7)
    #type of failure
    index = dimensionless_df.index
    for i in range(len(index)):
        if dimensionless_df.at[i,'Keyholing_KH2'] == 1:
            dimensionless_df.at[i,'failure_mode_LOF1KH2Ball2'] = 'Keyholing'
        elif ((np.pi*dimensionless_df.at[i,'width'])/dimensionless_df.at[i,'length']) < (np.sqrt(2/3)):
            dimensionless_df.at[i,'failure_mode_LOF1KH2Ball2'] = 'Balling'
        elif dimensionless_df.at[i,'depth_KH2_corrected'] <= dimensionless_df.at[i,'Powder_thick_m'] :
            dimensionless_df.at[i,'failure_mode_LOF1KH2Ball2'] = 'LOF'
        else:
            dimensionless_df.at[i,'failure_mode_LOF1KH2Ball2'] = 'Success'


            
# Define the final criteria set 3 (1,4,6)
    for dim_ball in dim_ball_value:
        for i in range(len(dimensionless_df)):
            if dimensionless_df.at[i, 'Keyholing_KH2'] == 1:
                dimensionless_df.at[i, f'failure_mode_LOF1KH2Ball1_{dim_ball}'] = 'Keyholing'
            elif dimensionless_df.at[i, 'length'] / dimensionless_df.at[i, 'width'] >= dim_ball:
                dimensionless_df.at[i, f'failure_mode_LOF1KH2Ball1_{dim_ball}'] = 'Balling'
            elif dimensionless_df.at[i, 'depth_KH2_corrected'] <= dimensionless_df.at[i, 'Powder_thick_m']:
                dimensionless_df.at[i, f'failure_mode_LOF1KH2Ball1_{dim_ball}'] = 'LOF'
            else:
                dimensionless_df.at[i, f'failure_mode_LOF1KH2Ball1_{dim_ball}'] = 'Success'
                
            
            
# Define the final criteria set 2 (1,3,7)
    # Type of failure
    for dim_key in dim_key_value:
        index = dimensionless_df.index
        for i in range(len(index)):
            if dimensionless_df.at[i, f'Keyholing_{dim_key}'] == 1:
                dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball2_{dim_key}'] = 'Keyholing'
            elif ((np.pi*dimensionless_df.at[i,'width'])/dimensionless_df.at[i,'length']) < (np.sqrt(2/3)):
                dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball2_{dim_key}'] = 'Balling'
            elif dimensionless_df.at[i, f'depth_KH1_corr_{dim_key}'] <= dimensionless_df.at[i, 'Powder_thick_m']:
                dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball2_{dim_key}'] = 'LOF'
            else:
                dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball2_{dim_key}'] = 'Success'





# Define the final criteria set 1 (1,3,6)
    # Type of failure
    for dim_key in dim_key_value:
        for dim_ball in dim_ball_value:
            index = dimensionless_df.index
            for i in range(len(index)):
                if dimensionless_df.at[i, f'Keyholing_{dim_key}'] == 1:
                    dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball1_{dim_key}_{dim_ball}'] = 'Keyholing'
                elif dimensionless_df.at[i,'length'] / dimensionless_df.at[i,'width'] >= dim_ball:
                    dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball1_{dim_key}_{dim_ball}'] = 'Balling'
                elif dimensionless_df.at[i, f'depth_KH1_corr_{dim_key}'] <= dimensionless_df.at[i, 'Powder_thick_m']:
                    dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball1_{dim_key}_{dim_ball}'] = 'LOF'
                else:
                    dimensionless_df.at[i, f'failure_mode_LOF1KH1Ball1_{dim_key}_{dim_ball}'] = 'Success'
                    

    
    elements = dimensionless_df['Elements']
    values = dimensionless_df['Atomic_frac']
    
    # Find unique elements across all lists
    unique_elements = set().union(*elements)
    
    # Fill missing values with zeros for each list in elements
    filled_values = [] 
    for i in range(len(elements)):
        element_list = elements[i]
        value_list = values[i]
        filled_v_list = [value_list[element_list.index(element)] if element in element_list else 0 for element in unique_elements]
        filled_values.append(filled_v_list)
    
    # Create DataFrame
    df_recon = pd.DataFrame(filled_values, columns=unique_elements)
    
    # Concatenate the two DataFrames with columns from df_recon in front
    df_combined = pd.concat([df_recon, dimensionless_df], axis=1)
    
    # Drop 'Elements' and 'Atomic_frac' columns
    df_combined.drop(['Elements', 'Atomic_frac'], axis=1, inplace=True)

  


    #save
    df_combined.to_csv('Package_output_' + file_name +'_' + e_t_model_type + '.csv') 
    results_df.to_csv('Material_PROP_' + file_name +'_' + e_t_model_type + '.csv')  
    
    print('Completed All Calculations') 