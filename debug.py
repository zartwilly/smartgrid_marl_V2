# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:29:19 2021

@author: jwehounou
"""
import os
import numpy as np
import pandas as pd
import fonctions_auxiliaires as fct_aux

from pathlib import Path


def couple_CiPiState_excelfile():
    t_periods = 50
    setA_m_players = 10
    setB1_m_players = 3
    setB2_m_players = 5
    setC_m_players = 8
    scenario_name = "scenario1"
    
    name_arr = fct_aux.AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                            setA_m_players, setB1_m_players, 
                            setB2_m_players, setC_m_players, 
                            t_periods, scenario_name)
    ### ____  read arr ______
    # # read arr without state
    # path_to_variable = os.path.join("tests", "AUTOMATE_INSTANCES_GAMES")
    # arr_pl_M_T_vars = np.load(os.path.join(path_to_variable, name_arr),
    #                           allow_pickle=True)
    # read arr with state
    path_to_variable \
        = os.path.join(
            "tests",
            "A1B1Automate2gamma_V0_V1_V2_V3_V4_T50_ksteps250_setACsetAB1B2C",
            "simu_DDMM_HHMM_"+scenario_name+"_T50gammaV4",
            "pi_hp_plus_10_pi_hp_minus_20",
            "DETERMINIST")
            # "LRI1",
            # "0.1")
    name_arr = "arr_pl_M_T_K_vars.npy"
    arr_pl_M_T_vars = np.load(os.path.join(path_to_variable, name_arr),
                              allow_pickle=True)
    
    selected_cols = ["Ci","Pi", "state_i"]
    m_players = arr_pl_M_T_vars.shape[0]
    N = 10
    arr = np.zeros(shape=(m_players, N)).astype(object)
    
    for t in range(0, 10):
        for pl_i in range(0, m_players):
            Ci = arr_pl_M_T_vars[pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
            Pi = arr_pl_M_T_vars[pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
            state_i = arr_pl_M_T_vars[pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["state_i"]]
            setx = arr_pl_M_T_vars[pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["set"]]
            
            arr[pl_i, t] = (Ci, Pi, state_i, setx)
            
    columns = ["t_"+str(t) for t in range(0, 10)]
    index = ["player"+str(pl_i) for pl_i in range(0, m_players)]
    df = pd.DataFrame(arr, columns=columns, index=index)
    
    path_2_xls_df = os.path.join("tests", "debug")
    Path(path_2_xls_df).mkdir(parents=True, exist_ok=True)
    df.to_excel(os.path.join(path_2_xls_df, "dbg_couple_CiPiState.xls"),
                index=True)

def compute_upper_bound_quantity_energy(arr_pl_M_T_K_vars_modif, t):
    """
    compute bought upper bound quantity energy q_t_minus
        and sold upper bound quantity energy q_t_plus
    """
    q_t_minus, q_t_plus = 0, 0
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi, Ci, Si, Si_max = None, None, None, None
        if len(arr_pl_M_T_K_vars_modif.shape) == 4:
            Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, 0, fct_aux.AUTOMATE_INDEX_ATTRS["Si_max"]]
        else:
            Pi = arr_pl_M_T_K_vars_modif[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
            Ci = arr_pl_M_T_K_vars_modif[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
            Si = arr_pl_M_T_K_vars_modif[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Si"]]
            Si_max = arr_pl_M_T_K_vars_modif[num_pl_i, t, fct_aux.AUTOMATE_INDEX_ATTRS["Si_max"]]
        diff_Ci_Pi = fct_aux.fct_positive(sum_list1=Ci, sum_list2=Pi)
        diff_Pi_Ci_Si_max = fct_aux.fct_positive(sum_list1=Pi, sum_list2=Ci+Si_max-Si)
        diff_Pi_Ci = fct_aux.fct_positive(sum_list1=Pi, sum_list2=Ci)
        diff_Ci_Pi_Si = fct_aux.fct_positive(sum_list1=Ci, sum_list2=Pi+Si)
        diff_Ci_Pi_Simax_Si = diff_Ci_Pi - diff_Pi_Ci_Si_max
        diff_Pi_Ci_Si = diff_Pi_Ci - diff_Ci_Pi_Si
        
        q_t_minus += diff_Ci_Pi_Simax_Si
        q_t_plus += diff_Pi_Ci_Si
        
        print("Pi={}, Ci={}, Si_max={}, Si={}".format(Pi, Ci, Si_max, Si))
        print("**player {}: diff_Ci_Pi_Simax_Si={} -> q_t_minus={}, diff_Pi_Ci_Si={} -> q_t_plus={} ".format(
            num_pl_i, diff_Ci_Pi_Simax_Si, q_t_minus, diff_Pi_Ci_Si, q_t_plus))
        
    # print("q_t_minus={}, q_t_plus={}".format(q_t_minus, q_t_plus))
    q_t_minus = q_t_minus if q_t_minus >= 0 else 0
    q_t_plus = q_t_plus if q_t_plus >= 0 else 0
    return q_t_minus, q_t_plus

def compute_q_t(t, scenario_name):
    t_periods = 50
    setA_m_players = 8
    setB1_m_players = 5
    setB2_m_players = 5
    setC_m_players = 8
    
    name_arr = fct_aux.AUTOMATE_FILENAME_ARR_PLAYERS_ROOT_SETAB1B2C.format(
                            setA_m_players, setB1_m_players, 
                            setB2_m_players, setC_m_players, 
                            t_periods, scenario_name)
    ### ____  read arr ______
    # read arr without state
    path_to_variable = os.path.join("tests", "AUTOMATE_INSTANCES_GAMES")
    arr_pl_M_T_vars = np.load(os.path.join(path_to_variable, name_arr),
                              allow_pickle=True)
    # q_t_minus, q_t_plus = fct_aux.compute_upper_bound_quantity_energy(
    #                                 arr_pl_M_T_vars, t)
    q_t_minus, q_t_plus = compute_upper_bound_quantity_energy(
                                    arr_pl_M_T_vars, t)
    print("q_t_minus={}, q_t_plus={}".format(q_t_minus, q_t_plus))


if __name__ == "__main__":
    #couple_CiPiState_excelfile()
    
    compute_q_t(t=4, scenario_name="scenario1")