import numpy as np
import matplotlib.pylab as plt
import uproot
import awkward as ak
import seaborn
import RA_funcs as rf
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import gamma
import Scope_funcs as sf
print("imports work")






"Merging"




# a function that get 2 arrays, and groups them by categories of the first 1, returns the grouped data and the classes(categories)
def ak_groupby_for_scope(classes, data):
    

    # zip the arrays
    zipped_arrays = ak.zip({ "classes":classes, "data":data} ,depth_limit=1)

    # sort by classes
    data_classes_sorted = zipped_arrays[ak.argsort(zipped_arrays.classes)]

    # divide to subarrays by classes
    lengths = ak.run_lengths(data_classes_sorted.classes)

    # return lengths, data_classes_sorted
    data_by_class_divided = ak.unflatten(data_classes_sorted, lengths)

    # get the classes list
    reduced_classes = data_by_class_divided.classes[..., 0]


    # return data_by_class_divided 
    return data_by_class_divided, reduced_classes










# Merge DUT and Telescope Data - takes run number and returnes zipped array with telescope files
def DUT_TELE_merge(run_number, return_TLU = False):

    # read DUT
    dut = uproot.open(f"TB_FIRE/TB_reco/TB_FIRE_{run_number}_raw_reco.root")
    hits = dut['Hits']
    plane = hits['plane_ID'].array()
    channel = hits['ch_ID'].array()
    amp = hits['amplitude'].array()
    TLU = hits['TLU_number'].array()

    if return_TLU:
        hit_data = ak.zip({"TLU":TLU, "plane":plane, "ch":channel, "amp":amp})
    else:
        hit_data = ak.zip({"plane":plane, "ch":channel, "amp":amp})

    # sort DUT data to have a single TLU per event

    TLU_data, TLU = sf.ak_groupby_for_scope(TLU,hit_data)
    single_TLU_data = TLU_data[ak.num(TLU_data) == 1]
    single_TLU_data = single_TLU_data[ak.num(single_TLU_data) > 0]
    
    # read tele
    tele = uproot.open(f"TB_FIRE/TB_reco/run_{run_number}_telescope.root")
    tracks = tele['TrackingInfo']['Tracks']
    trigID = tracks['triggerid'].array()
    chi2_ndof = tracks['chi2'].array() / tracks['ndof'].array()
    x = tracks['x_dut'].array()
    y = tracks['y_dut'].array()
    tele_data = ak.zip({"x":x, "y":y, "chired":chi2_ndof})

    # sort TELE data by trigger ID
    trigID_data, trigID = sf.ak_groupby_for_scope(ak.flatten(trigID), tele_data)
    
    # events with 1 triger ID
    single_trigID = trigID_data[ak.num(trigID_data) == 1]

    # events with one telescope track only
    single_track_with_zeros = single_trigID[ak.num(single_trigID.data, axis = 2) == 1]
    single_track_data = single_track_with_zeros[ak.num(single_track_with_zeros) > 0]

    # create mask to get only data with TLU and trigger ID by substraction

    # get indices for events that with matching TLU and TriggerID
    single_TLU_index = single_TLU_data.classes
    single_track_index = single_track_data.classes
    mask_dut = np.isin(single_TLU_index, single_track_index)
    mask_tele = np.isin(single_track_index, single_TLU_index)

    # get the data from dut and tele for corresponding events (apply mask)
    final_dut_data = single_TLU_data[mask_dut]
    final_tele_data = single_track_data[mask_tele]

    # combine DUT and TELE data
    final_dut_hit_data = final_dut_data.data
    final_tele_fit_data = final_tele_data.data
    # return final_dut_hit_data, final_tele_fit_data
    merged = ak.zip({"hits":final_dut_hit_data, "tele":final_tele_fit_data},depth_limit=1)
    

    return merged





























# gal's merging
def gal_scope_merge(run_number):
    import gc
    import pandas as pd
    f = uproot.open(f"TB_FIRE/TB_reco/TB_FIRE_{run_number}_raw_reco.root")
    tele = uproot.open(f"TB_FIRE/TB_reco/run_{run_number}_telescope.root")

    arrs_events = f["Hits"].arrays(["TLU_number", "amplitude", "plane_ID", "ch_ID"], library="ak")
    tele_events = tele["TrackingInfo/Tracks"].arrays(["triggerid", "x_dut", "y_dut"], library="ak")

    arrs_events.TLU_number

    df_dut = pd.DataFrame({
        "TLU": ak.to_list(np.sort(arrs_events.TLU_number)),
        "amplitude": ak.to_list(arrs_events.amplitude),
        "plane_ID": ak.to_list(arrs_events.plane_ID),
        "ch_ID": ak.to_list(arrs_events.ch_ID)
    })

    nonempty_mask = (ak.num(tele_events["x_dut"]) == 1) & (ak.num(tele_events["y_dut"]) == 1)
    
    tele_trig_filtered = tele_events["triggerid"][nonempty_mask]
    tele_x_filtered    = tele_events["x_dut"][nonempty_mask]
    tele_y_filtered    = tele_events["y_dut"][nonempty_mask]

    flat_triggers = ak.to_list(ak.flatten(tele_trig_filtered, axis=None))
    flat_x       = ak.to_list(ak.flatten(tele_x_filtered, axis=None))
    flat_y       = ak.to_list(ak.flatten(tele_y_filtered, axis=None))

    df_tele = pd.DataFrame({
        "triggerid": np.sort(flat_triggers),
        "x_dut": flat_x,
        "y_dut": flat_y
    })
   
    print("DataFrames constructed.")

    df_tele = df_tele[df_tele["triggerid"].isin(df_dut["TLU"])]
    df_dut = df_dut[df_dut["TLU"].isin(df_tele["triggerid"])]
    
    # Group DUT hits by TLU
    grouped_dut = df_dut.groupby("TLU", sort=False).agg({
        "amplitude": lambda s: [item for sublist in s for item in (sublist if isinstance(sublist, list) else [sublist])],
        "plane_ID": lambda s: [item for sublist in s for item in (sublist if isinstance(sublist, list) else [sublist])],
        "ch_ID": lambda s: [item for sublist in s for item in (sublist if isinstance(sublist, list) else [sublist])]
    }).reset_index()

    # Group telescope hits by triggerid
    grouped_tele = df_tele.groupby("triggerid", sort=False).agg({
        "x_dut": lambda s: [item for sublist in s for item in (sublist if isinstance(sublist, list) else [sublist])],
        "y_dut": lambda s: [item for sublist in s for item in (sublist if isinstance(sublist, list) else [sublist])]
    }).reset_index().rename(columns={"triggerid": "TLU"})


    print("DataFrames grouped by TLU/triggerid")
    # Merge DUT and telescope on TLU
    merged = pd.merge(grouped_dut, grouped_tele, on="TLU", how="left")

    print("DataFrames merged")

    del flat_triggers, flat_x, flat_y, df_dut, df_tele, grouped_dut, grouped_tele
    gc.collect()

    print("collected garbage")

    events_awk = ak.zip({
        "TLU": ak.Array(merged["TLU"].tolist()),
        "Amplitudes": ak.Array(merged["amplitude"].tolist()),
        "Planes": ak.Array(merged["plane_ID"].tolist()),
        "Channels": ak.Array(merged["ch_ID"].tolist()),
        "x_dut": ak.flatten(ak.Array(merged["x_dut"].tolist()),axis=1),
        "y_dut": ak.flatten(ak.Array(merged["y_dut"].tolist()),axis=1)
    })

    return events_awk


















# filter chi2
def filter_chi2_scope_data(hit_data_scope, upper_chi2_bound):
    mask = hit_data_scope.tele.chired < upper_chi2_bound
    filtered_data = hit_data_scope[ak.flatten(mask)]
    filtered_data_clean = filtered_data[ak.num(filtered_data.tele) > 0]
    return filtered_data_clean











"Gap"


def E_vs_X_scope_gaussian_fit(hit_data, chi2, y_min=-10, y_max=10, x_min=-20, x_max=-20):

    # filter data by chi2
    hit_data_chi2 = sf.filter_chi2_scope_data(hit_data, chi2)
    y_max, y_min

    # take the data from the seletced y range
    data = hit_data_chi2[ak.flatten((hit_data_chi2.tele.y < y_max) & (hit_data_chi2.tele.y > y_min))]
    data = data[ak.num(data.tele) > 0]


    # compute X and E
    X = -ak.to_numpy(ak.mean(data.tele.x, axis=1))
    X = np.round(X,1)
    E = ak.sum(data.hits.amp, axis=1)
    print(E)

    # grouping
    amp, mean, pos = rf.ak_groupby(X, E, round="false")

    # statistics
    amp_avg = ak.mean(amp.data, axis=1)
    amp_std = ak.std(amp.data, axis=1) / np.sqrt(ak.num(amp.data, axis=1) - 1)

    # choose the range of X
    mask = (pos > x_min) & (pos < x_max)

    # convert Awkward → Numpy
    pos_m = ak.to_numpy(pos[mask])
    amp_m = ak.to_numpy(amp_avg[mask])
    err_m = ak.to_numpy(amp_std[mask])

    # Fitting

    #  Gaussian model 
    def gaussian_linear(x, c, m, A, mu, sigma):
        return c + m*x - A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # initial guesses
    c0 = 6000
    m0 = 0
    A0 = np.min(amp_m)
    # mu0 = pos_m[np.argmin(amp_m)]
    mu0 = 0
    sigma0 = 3
    # sigma0 = (np.max(pos_m) - np.min(pos_m)) / 6

    # fit
    popt, pcov = curve_fit(gaussian_linear, pos_m, amp_m, p0=[c0, m0, A0, mu0, sigma0])

    c_fit, m_fit, A_fit, mu_fit, sigma_fit = popt

    print("Gaussian fit parameters:")
    print(f"c     = {c_fit:.3f}")
    print(f"m     = {m_fit:.3f}")
    print(f"theta     = {np.arctan(m_fit):.3f} Radians")
    print(f"A     = {A_fit:.3f}")
    print(f"mu    = {mu_fit:.3f}")
    print(f"sigma = {sigma_fit:.3f}")

    # make smooth curve for plotting
    x_fit = np.linspace(np.min(pos_m), np.max(pos_m), 500)
    y_fit = gaussian_linear(x_fit, *popt)

    # --- plotting ---
    plt.errorbar(pos_m, amp_m, yerr=err_m, fmt='.', capsize=4, label="data")
    plt.plot(x_fit, y_fit, 'r-', label="Gaussian fit", zorder=10)
    print((gaussian_linear(-15, *popt) , gaussian_linear(150, *popt)))

    plt.grid()
    plt.xlabel("pos")
    plt.ylabel("amplitude (avg ± std)")
    plt.legend()
    plt.show()














"channel reconstruction"

# plots the XY distribution of events that started in the wanted pads
def pads_xy(data, central_pad):

    # determine the chosen pads
    base = list(range(central_pad - 2, central_pad + 3))
    pads = base + list(map(lambda x: x + 20, base)) + list(map(lambda x: x - 20, base))
    print(pads)

    # data from the first plane only
    first_plane_mask = data.hits.plane == 0
    first_plane_data = data.hits[first_plane_mask]
    
    # events with single hit in first plane
    single_hit_mask = ak.num(first_plane_data) == 1
    single_hit_first_plane_data = data[single_hit_mask] 

    # data from the first plane with single hit
    first_plane_single_hit = first_plane_data[single_hit_mask]

    # get events that starts at the chosen pads
    chosen_pads_mask = np.isin(first_plane_single_hit.ch, pads)
    chosen_pads_data = single_hit_first_plane_data[chosen_pads_mask]

    # chosen pads data
    chosen_pads = chosen_pads_data.hits.ch

    # get the xy data for each
    x = -chosen_pads_data.tele.x
    y = chosen_pads_data.tele.y

   
   
    # plot xy by color
    unique_pads = np.unique(chosen_pads)
    pad_to_idx = {p: i for i, p in enumerate(unique_pads)}
    color_idx = np.array([pad_to_idx[p] for p in chosen_pads])

    plt.scatter(x, y, c=color_idx, cmap='tab20', s=0.05)
    plt.colorbar(ticks=np.arange(len(unique_pads)),label='pad').set_ticklabels(unique_pads)
    plt.title("XY Pads Distribution")
    plt.show()

    return chosen_pads_data
    