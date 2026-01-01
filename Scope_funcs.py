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





# # find differrence between ben's and gal's merging
# tlu_ben = ak.firsts(hit_data_scope_1081_tlu.hits.TLU, axis = 1) 
# tlu_gal = ak.firsts(gal_scope_1081.TLU)
# gal_in_ben_mask = np.isin(tlu_gal,tlu_ben)
# tlu_gal_in_ben = tlu_gal[~gal_in_ben_mask]

# gal_ben_1081_scope = gal_scope_1081[~gal_in_ben_mask]

# gal_ben_1081_scope = gal_scope_1081[~gal_in_ben_mask]
# X_gal1 = ak.mean(gal_ben_1081_scope.x_dut, axis=1)
# Y_gal1 = ak.mean(gal_ben_1081_scope.y_dut, axis=1)

# X_gal = -ak.to_numpy(X_gal1)
# Y_gal = ak.to_numpy(Y_gal1)

# amp_gal1 = ak.sum(gal_ben_1081_scope.Amplitudes, axis=1)
# amp_gal = ak.to_numpy(amp_gal1)

# # Define bins
# bins = 100

# # Histogram of SUM of amplitudes
# sum_amp, xedges, yedges = np.histogram2d(X_gal, Y_gal, bins=bins, weights=amp_gal)

# # Histogram of COUNTS
# counts, _, _ = np.histogram2d(X_gal, Y_gal, bins=[xedges, yedges])

# # Avoid division by zero
# avg_amp = np.divide(sum_amp, counts, out=np.zeros_like(sum_amp), where=counts > 0)

# # Plot
# plt.figure(figsize=(6,5))
# plt.pcolormesh(xedges, yedges, avg_amp.T, cmap="tab20c")  
# plt.colorbar(label="Average Amplitude")
# plt.xlim(min(X_gal), max(X_gal))
# # plt.xlim(-30, 30)
# plt.ylim(min(Y_gal), max(Y_gal))
# # plt.ylim(-30, 20)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("2D Histogram of Average Amplitude - run 1081, Gal")
# plt.show()












# filter chi2
def filter_chi2_scope_data(hit_data_scope, upper_chi2_bound):
    mask = hit_data_scope.tele.chired < upper_chi2_bound
    filtered_data = hit_data_scope[ak.flatten(mask)]
    filtered_data_clean = filtered_data[ak.num(filtered_data.tele) > 0]
    return filtered_data_clean



























# colormap of the average showeer energy for its scope position
def avg_energy_scope_colormap(data, x_borders="false", y_borders="false", cmap="tab20c", bins=300):

    X_scope1 = ak.flatten(data.tele.x)
    Y_scope1 = ak.flatten(data.tele.y)

    X_scope = -ak.to_numpy(X_scope1)
    Y_scope = ak.to_numpy(Y_scope1)

    amp1 = ak.sum(data.hits.amp, axis = 1)
    amp = ak.to_numpy(amp1)

    # Histogram of SUM of amplitudes
    sum_amp, xedges, yedges = np.histogram2d(X_scope, Y_scope, bins=bins, weights=amp)

    # Histogram of COUNTS
    counts, _, _ = np.histogram2d(X_scope, Y_scope, bins=[xedges, yedges])

    # Avoid division by zero
    avg_amp = np.divide(sum_amp, counts, out=np.zeros_like(sum_amp), where=counts > 0)

    # Plot
    plt.figure(figsize=(6,5))
    plt.pcolormesh(xedges, yedges, avg_amp.T, cmap="tab20c")  
    plt.colorbar(label="Average Amplitude")
    plt.xlim(min(X_scope), max(X_scope))
    plt.ylim(min(Y_scope), max(Y_scope))
    
    if x_borders != "false":
        plt.xlim(-x_borders, x_borders)
    if y_borders != "false":
        plt.ylim(-y_borders, y_borders)
    
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title("Average Shower Energy per Position")
    plt.show()


















"Gap"












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
    


















# plots the energy profile vs the x axis and fits to gaussian
def E_vs_X_scope_gaussian_fit(hit_data, chi2, y_min=-10, y_max=10, x_min=-20, x_max=20, bin_size = 0.4, return_param=False):

    # filter data by chi2
    hit_data_chi2 = sf.filter_chi2_scope_data(hit_data, chi2)
    y_max, y_min

    # take the data from the seletced y range
    data = hit_data_chi2[ak.flatten((hit_data_chi2.tele.y < y_max) & (hit_data_chi2.tele.y > y_min))]
    data = data[ak.num(data.tele) > 0]


    # compute X and E
    X1 = -ak.to_numpy(ak.mean(data.tele.x, axis=1))
    X = bin_size * np.round(X1 / bin_size)
    E = ak.sum(data.hits.amp, axis=1)

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

    # fitted values
    c_fit, m_fit, A_fit, mu_fit, sigma_fit = popt
    
    # uncertainties
    perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties
    c_err, m_err, A_err, mu_err, sigma_err = perr

    # return the fitting results if needed
    if return_param:
        return popt, perr

    # slope and uncertainty of the fitted function
    theta_fit = np.arctan(m_fit)
    theta_err = m_err / (1.0 + m_fit**2)


    # chi2
    residuals = amp_m - gaussian_linear(pos_m, *popt)

    # avoid division by zero if any std are 0
    mask_err = err_m > 0
    chi2_val = np.sum((residuals[mask_err] / err_m[mask_err])**2)

    # number of points used in chi2
    N = np.sum(mask_err)
    
    # number of fit params (c, m, A, mu, sigma) = 5
    p = len(popt)
    ndof = N - p
    chi2_ndof = chi2_val / ndof if ndof > 0 else np.nan


    print("Gaussian fit parameters:")
    print(f"c     = {c_fit:.3f}")
    print(f"m     = {m_fit:.3f}")
    print(f"theta     = {np.arctan(m_fit):.3f} Radians")
    print(f"A     = {A_fit:.3f}")
    print(f"mu    = {mu_fit:.3f}")
    print(f"sigma = {sigma_fit:.3f}")


    # plot
    fig, ax = plt.subplots()
    ax.errorbar(pos_m, amp_m, yerr=err_m, fmt='.', capsize=4, label="data")

    # make a smooth fitted function
    x_fit = np.linspace(np.min(pos_m), np.max(pos_m), 500)
    y_fit = gaussian_linear(x_fit, *popt)
    ax.plot(x_fit, y_fit, 'r-', label="Gaussian fit", zorder=10)

    # build textbox string
    textstr = "\n".join([
        r"$\chi^2/\mathrm{ndof} = %.2f$" % chi2_ndof,
        r"$c = %.3f \pm %.3f$" % (c_fit, c_err),
        r"$m = %.3f \pm %.3f$" % (m_fit, m_err),
        r"$\theta = %.3f \pm %.3f$ rad" % (theta_fit, theta_err),
        r"$A = %.3f \pm %.3f$" % (A_fit, A_err),
        r"$\mu = %.3f \pm %.3f$" % (mu_fit, mu_err),
        r"$\sigma = %.3f \pm %.3f$" % (sigma_fit, sigma_err),
    ])

    # add textbox (axes coordinates: 0..1)
    ax.text(
        0.65, 0.36, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="pink", alpha=0.8, edgecolor="0.5")
    )

    ax.grid(True)
    ax.set_xlabel("X Scope")
    ax.set_ylabel("Energy [ADC] (avg ± std)")
    # ax.set_title(r"Average Energy vs Scope X Position \n Fit to $y = m \cdot x + c - A\cdot e^{\frac{-(x - \mu)^2}{(2 \sigma^2)}}$")
    ax.set_title("Average Energy vs Scope X Position\n"
             r"Fit to: $y = m \cdot x + c - A\cdot exp(\frac{-(x - \mu)^2}{2 \sigma^2})$")

    ax.legend(loc="upper left")
    # leg = ax.legend(loc="upper right", title=textstr, frameon=True)
    plt.show()
























    # plots the gap parameters for different sections of y on the shower
def plot_gap_vs_y(scope_data, y_range=10, y_bins=1, chi2=1):

    # lists for the plotting
    c_list = []
    c_err_list = []
    A_list = []
    A_err_list = []
    mu_list = []
    mu_err_list = []
    sigma_list = []
    sigma_err_list = []
    y_list = []

    for i in np.arange(-y_range, y_range, y_bins):

        # get the gaussian parameters for each y
        popt, perr = E_vs_X_scope_gaussian_fit(scope_data, chi2, i, i+y_bins, -13, 14, bin_size=0.04, return_param=True)
        c, m, A, mu, sigma = popt
        c_err, m_err, A_err, mu_err, sigma_err = perr
        
        y = i + y_bins/2
        y_list.append(y)

        # save the data
        c_list.append(c)
        c_err_list.append(c_err)
        A_list.append(A)
        A_err_list.append(A_err)
        mu_list.append(mu)
        mu_err_list.append(mu_err)
        sigma_list.append(sigma)
        sigma_err_list.append(sigma_err)


    # plot the parameters
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)

    linewdith = 1
    markersize = 3

    # plot C
    axs[0,0].errorbar(y_list, c_list, yerr=c_err_list, fmt='.', capsize=4, label="data")
    axs[0, 0].set_title("Average Energy Outside the Gap (c)")
    axs[0,0].set_ylabel("Energy Count [ADC]")

    # plot A
    axs[0,1].errorbar(y_list, A_list, yerr=A_err_list, fmt='.', color="red", capsize=4, label="data")
    axs[0, 1].set_title("Energy Inside the Gap (Gaussian Depth - A)")
    axs[0,1].set_ylabel("Energy Count Inside the Gap [ADC]")

    # plot gaussian mean (mu)
    axs[1,0].errorbar(y_list, mu_list, yerr=mu_err_list, fmt='.', color="purple", capsize=4, label="data")
    axs[1, 0].set_title("Gap Position on the scope (mu)")
    axs[1,0].set_ylabel("Gap Position [mm]")

    # plot gaussian width (sigma)
    axs[1,1].errorbar(y_list, sigma_list, yerr=sigma_err_list, fmt='.', color="green", capsize=4, label="data")
    axs[1, 1].set_title("Gap width (sigma)")
    axs[1,1].set_ylabel("sigma [mm]")

    for ax in axs.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("y")
        ax.tick_params(axis="x", labelleft=True)

    plt.show()
