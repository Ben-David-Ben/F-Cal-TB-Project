# ROOT ANALYSIS FUNCTIONS

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




"Awkward supportive functions"

# a function that get 2 arrays, and groups them by categories of the first 1, returns the grouped data, mean of data for each group, and the classes(categories)
def ak_groupby(classes, data, round = "true"):
    
    # round the values
    if round == "true":
        classes = ak.round(classes)

    # zip the arrays
    zipped_arrays = ak.zip({ "classes":classes, "data":data})

    # sort by classes
    data_classes_sorted = zipped_arrays[ak.argsort(zipped_arrays.classes)]

    # divide to subarrays by classes
    lengths = ak.run_lengths(data_classes_sorted.classes)
    data_by_class_divided = ak.unflatten(data_classes_sorted, lengths)

    # get the classes list
    reduced_classes = data_by_class_divided.classes[..., 0]

    # get the mean of data from the same class
    data_avg_per_location = ak.mean(data_by_class_divided.data, axis = 1)


    # return data_by_class_divided 
    return data_by_class_divided, data_avg_per_location, reduced_classes









# sums the cumulative value of each subarray in an ak jagged array
def ak_cumsum_per_sublist(a):

    # lengths of each subarray
    counts = ak.num(a)

    # flatten all values into one array
    flat = ak.flatten(a)

    # global cumulative sum
    cum = np.cumsum(ak.to_numpy(flat))

    # totals per subarray
    totals = ak.sum(a, axis=1)
    # cumulative totals of *previous* subarrays
    offsets = np.concatenate([[0], np.cumsum(ak.to_numpy(totals[:-1]))])

    # repeat each offset by the length of its corresponding subarray
    offsets_repeated = np.repeat(offsets, ak.to_numpy(counts))

    # subtract offsets to reset each subarray’s start to 0
    cum_shifted = cum - offsets_repeated

    # rebuild jagged structure
    return ak.unflatten(cum_shifted, counts)

















"Run Analysis - location, energy histogram etc"




# extracts arrays from ROOT file and zip them for every hit
def get_ROOT_data_zip(run_number, tlu = "false", time = "false", toa = "false" ):

    file_name = f"TB_FIRE\\{run_number}"
    # open the file
    infile = uproot.open(file_name)
    # print("Folders:", infile.keys())


    # open the first "folder" hits
    hits = infile['Hits']
    # print("Hits:")
    # hits.show()

    # create the arrays from all data
    amp = hits['amplitude'].array()
    plane = hits['plane_ID'].array()
    channel = hits['ch_ID'].array()
    if tlu == "true":
        tlu = hits['TLU_number'].array()
    if toa == "true":
        toa = hits['toa'].array()
    if time == "true":
        time = hits['timestamp'].array()

    # create a zipped array of data for every hit(reading in the sensor)
    hit_data = ak.zip({ "plane":plane, "ch":channel, "amp":amp})
    print(f"{run_number} finished")

    return hit_data














# extracts arrays from ROOT file and zip them for every hit, FOR RECONSTRUCTED files
def get_ROOT_data_zip_RECO(run_number, tlu = "false", time = "false", toa = "false" ):

    file_name = f"TB_FIRE\TB_reco\TB_FIRE_{run_number}_raw_reco.root"
    # open the file
    infile = uproot.open(file_name)
    # print("Folders:", infile.keys())


    # open the first "folder" hits
    hits = infile['Hits']
    # print("Hits:")
    # hits.show()

    # create the arrays from all data
    amp = hits['amplitude'].array()
    plane = hits['plane_ID'].array()
    plane = 7 - plane
    channel = hits['ch_ID'].array()
    if tlu == "true":
        tlu = hits['TLU_number'].array()
    if toa == "true":
        toa = hits['toa'].array()
    if time == "true":
        time = hits['timestamp'].array()

    # create a zipped array of data for every hit(reading in the sensor)
    hit_data = ak.zip({ "plane":plane, "ch":channel, "amp":amp})
    print(f"{run_number} RECONSTRUCTED finished")

    return hit_data






# extracts arrays from ROOT file and zip them for every hit, FOR RECONSTRUCTED files
def get_ROOT_data_zip_RECO_11(run_number, tlu = "false", time = "false", toa = "false" ):

    file_name = f"TB_FIRE\TB_reco\TB_FIRE_{run_number}_raw_reco.root"
    # open the file
    infile = uproot.open(file_name)
    # print("Folders:", infile.keys())


    # open the first "folder" hits
    hits = infile['Hits']
    # print("Hits:")
    # hits.show()

    # create the arrays from all data
    amp = hits['amplitude'].array()
    plane = hits['plane_ID'].array()
    channel = hits['ch_ID'].array()
    if tlu == "true":
        tlu = hits['TLU_number'].array()
    if toa == "true":
        toa = hits['toa'].array()
    if time == "true":
        time = hits['timestamp'].array()

    # create a zipped array of data for every hit(reading in the sensor)
    hit_data = ak.zip({ "plane":plane, "ch":channel, "amp":amp})
    print(f"{run_number} RECONSTRUCTED finished")

    return hit_data





































"Bad Channels Diagnostics"




# get the dead channels (state the path to the diagnostic file and number of planes in your run)
def channels_diagnostics(path_to_file, number_of_planes, dead_channel_list = True):
    
    # open the diagnostic file
    infile = uproot.open(path_to_file)

    # get the isgood, plane_ID and channel_ID data for 8 planes
    if number_of_planes < 10:
        bad_channels_result = infile['Diagnostics;1']['BadChannelResults']
        plots_rows = 2
    
    # get the isgood, plane_ID and channel_ID data for 8 planes
    if number_of_planes > 10:
        bad_channels_result = infile['BadChannelResults']
        plots_rows = 3
        
    isgood = bad_channels_result['isGood'].array()
    plane_ID = bad_channels_result['planeID'].array()
    channel_ID = bad_channels_result['channelID'].array()
    
    print("we got the data")

    # attach channels to planes
    Y, X = divmod(channel_ID, 20)
    XY = ak.zip({ "X":X, "Y":Y})
    one_d_channels = ak.zip({ "plane_ID":plane_ID, "channel_ID":channel_ID})
    all_channels = ak.zip({ "plane_ID":plane_ID, "channel_ID":XY})

    # get the good and bad channels
    bad_channels = all_channels[~isgood]
    bad_channels_one_d = one_d_channels[~isgood]

    if dead_channel_list:
        return bad_channels_one_d

    # create the matrices with the bad channels
    matrices = []
    for i in range(number_of_planes):
        bad_channel_matrix = np.zeros((13,20))

        # bad channels in the i'th plane
        bad_channels_plane_i = bad_channels[bad_channels.plane_ID == i].channel_ID

        # update the entries at the dead channels as 1
        for ch in bad_channels_plane_i:
            x = ch.X
            y = ch.Y
            bad_channel_matrix[-1-y][x] = 1

        matrices.append(bad_channel_matrix)

    # plot
    # fig, axes = plt.subplots(2, 4, figsize=(16, 10))  # larger figure
    fig, axes = plt.subplots(plots_rows, 4, figsize=(18, 12))  # larger figure
    axes = axes.ravel()

    # Generate pad numbering (13×20)
    pad_numbers = np.arange(260).reshape(13, 20)

    for idx, (ax, mat) in enumerate(zip(axes, matrices)):
        
        # Plot heatmap
        seaborn.heatmap(
            mat,
            ax=ax,
            cmap="coolwarm",
            square=False,
            annot = pad_numbers[::-1],   # write pad numbers
            fmt="d",
            cbar = False,
            annot_kws={"size": 6},
            linewidths=0.3,
            linecolor="black"
        )

        # title for each plane
        ax.set_title(f"Plane {idx}", fontsize=14)

        # add the gap seperator
        ax.axvline( x=12, color="white", linestyle="--", linewidth=1.2)

        # X ticks (0–19)
        ax.set_xticks(np.arange(20) + 0.5)
        ax.set_xticklabels(np.arange(20), rotation = 55)

        # Y ticks (0–12)
        ax.set_yticks(np.arange(13) + 0.5)
        ax.set_yticklabels(np.arange(12, -1, -1), rotation = 0)  # bottom = 0

        if len(axes) > len(matrices):
            axes[len(matrices)].set_visible(False)
    
    fig.suptitle("Bad Channels (Red). Plane 0 is Closest to the Beam", fontsize = 20)
    plt.tight_layout()
    plt.show()

    return bad_channels_one_d

# # replace this with your path to the file and number of planes
# path = "TB_FIRE\TB_reco\TB_FIRE_1340_raw_reco.root"
# number_of_planes = 11

# channels_diagnostics(path, number_of_planes)












"data analysis"




# gets data and returns for a specific plane: pads_2d - the pads with signals in 2d coordinates ; counts- amount of hits on each pad
def plane_hit_counts(hit_data, plane):

    # get only the hits on the wanted plane
    hits_plane = hit_data[hit_data.plane == plane]

    # get only the channels data and clean the array from empty cells
    clean_plane_ch = hits_plane.ch[ak.num(hits_plane.ch) > 0]

    # count the amount of hits in each pad on the plane
    pads_1d, counts = np.unique(ak.flatten(clean_plane_ch), axis=0, return_counts=True)

    # convert the 1d index of the pads into 2d coordinates
    pads_2d = divmod(pads_1d, 20)    

    return pads_2d, counts









# a colormap with the AMOUNT OF HITS in every channel(pad) of a chosen plane
def hits_amount_colormap_single_plane(hit_data, plane_number, cmap="berlin", save = False, inverse_plane_order=False):
    
    #  change index so that the first plane is 0 and last is 7
    if inverse_plane_order:
        plane_number = 8 - plane_number
    else:
        plane_number = plane_number-1

    # get only the hits on the wanted plane
    hits_plane_n = hit_data[hit_data.plane == plane_number]

    # get only the channels data and clean the array from empty cells
    clean_plane_n_ch = hits_plane_n.ch[ak.num(hits_plane_n.ch) > 0]

    # count the amount of hits in each pad on the plane
    pads_1d, counts = np.unique(ak.flatten(clean_plane_n_ch), axis=0, return_counts=True)

    # convert the 1d index of the pads into 2d coordinates
    pads_2d = divmod(pads_1d, 20)

    # distribute the counts for each pad on a 12x20 matrix
    counts_matrix = np.zeros((13, 20))
    for i in range(len(pads_1d)):
        q = pads_2d[0][i] # quotinent of the i'th pad (row from bottom)
        r = pads_2d[1][i] # remainder of the i'th pad (column)
        counts_matrix[-1-q][r] = counts[i]

    if inverse_plane_order:
        plane_number = 8 - plane_number
    else:
        plane_number = plane_number+1
    
    # creat the colormap
    ax = seaborn.heatmap(counts_matrix, cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Hit Counts'})
    plt.title(f'Number of Hits in each channel, plane {plane_number} [XO]')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(range(len(ax.get_yticks())-1, -1, -1))
    if save:
        plt.savefig(r"Plots\TB2025 Gap\run 1080 reco" + "\\" + f"plane_{plane_number}_hits.png", dpi=300, bbox_inches="tight")
    plt.show()





















# a colormap with the total AMPLITUDE of hits in every channel(pad) of the chosen plane
def amp_colormap_single_plane(hit_data, plane_number, cmap="berlin"):

    #  change index so that the first plane is 0 and last is 7
    plane_number = 8 - plane_number

    # get only the hits on the wanted plane
    hits_plane_n = hit_data[hit_data.plane == plane_number]

    # clean data from empty cells
    clean_plane_n = hits_plane_n[ak.num(hits_plane_n.ch) > 0]

    # create channel(pads) matrix
    counts_matrix = np.zeros((13, 20))
    
    # count the amplitude for every pad and add them to the channel matrix
    for l in range(256):
        plane_pad = clean_plane_n[clean_plane_n.ch == l]     # array with the data of channel l
        clean_plane_pad = plane_pad[ak.num(plane_pad) > 0]   # clean from empty entries
        plane_pad_amp = clean_plane_pad.amp                  # get only the amplitudes of the given pad in the plane
        plane_pad_total_amp = ak.sum(plane_pad_amp)          # sum all the amplitudes
        pad_2d = divmod(l, 20)                               # coordinates of the pad on the matrix
        q = pad_2d[0]                                        # quotinent of the l'th pad (row from bottom)
        r = pad_2d[1]                                        # remainder of the l'th pad (column)
        counts_matrix[-1-q][r] = plane_pad_total_amp

    # creat the colormap
    seaborn.heatmap(counts_matrix, cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Hit Counts'})
    plt.title(f'Total Amplitude in Each Channel, Plane {8 - plane_number} XO')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)

















# show the average amplitude of the entire run in each pad
def average_amp_colormap_single_plane(hit_data, plane_number, cmap="managua"):

    #  change index so that the first plane is 0 and last is 7
    plane_number = 8 - plane_number

    # get only the hits on the wanted plane
    hits_plane_n = hit_data[hit_data.plane == plane_number]

    # clean data from empty cells
    clean_plane_n = hits_plane_n[ak.num(hits_plane_n.ch) > 0]

    # create channel(pads) matrix
    counts_matrix = np.zeros((13, 20))
    
    # count the amplitude for every pad and add them to the channel matrix
    for l in range(256):
        plane_pad = clean_plane_n[clean_plane_n.ch == l]     # array with the data of channel l
        clean_plane_pad = plane_pad[ak.num(plane_pad) > 0]   # clean from empty entries
        plane_pad_amp = clean_plane_pad.amp                  # get only the amplitudes of the given pad in the plane
        plane_pad_total_amp = ak.sum(plane_pad_amp)          # sum all the amplitudes
        pad_2d = divmod(l, 20)                               # coordinates of the pad on the matrix
        q = pad_2d[0]                                        # quotinent of the l'th pad (row from bottom)
        r = pad_2d[1]                                        # remainder of the l'th pad (column)
        # make sure we dont try to update the matrix with null value
        if len(plane_pad_amp) == 0:
            avg_amp_l = 0
        else:
            avg_amp_l = plane_pad_total_amp / len(plane_pad_amp)
        counts_matrix[-1-q][r] = avg_amp_l
        
    # creat the colormap
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(counts_matrix, cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Hit Amplitude'} , annot=True, fmt=".0f")
    plt.title(f'Average Amplitude in Each Channel, Plane {8 - plane_number} XO')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)































# Shows the shower evolution of a SINGLE EVENT in the sensor with the amplitude in every sensor
def single_event_evolution_amp(hit_data, TLU_number, cmap="berlin", save = "false"):

    # get the path of the specific event
    TLU_event_data = hit_data[TLU_number]

    
    # count the amount of hits of each pad in every plane
    for plane in range(7,-1,-1):
        
        # get the channels hits on the wanted plane
        hits_plane = TLU_event_data[TLU_event_data.plane == plane]
        hits_plane_ch = TLU_event_data[TLU_event_data.plane == plane].ch

        # create channel(pads) matrix
        counts_matrix = np.zeros((13, 20))
        
        # update the pads matrix only if there are any hits on the plane
        if len(hits_plane_ch) != 0:
            # count the amount of hits in each pad on the plane
            pads_1d, counts = np.unique(hits_plane_ch, return_counts=True)

            # convert the 1d index of the pads into 2d coordinates
            pads_2d = divmod(pads_1d, 20)    
    
            # distribute the counts for each pad on a 12x20 matrix
            counts_matrix = np.zeros((13, 20))
            for i in range(len(pads_1d)):
                q = pads_2d[0][i]           # quotinent of the i'th pad (row from bottom)
                r = pads_2d[1][i]           # remainder of the i'th pad (column)
                pad_amp = hits_plane[hits_plane.ch == pads_1d[i]].amp
                counts_matrix[-1-q][r] = pad_amp[0]
                
        # creat the colormap
        plt.figure(figsize=(10, 8))
        seaborn.heatmap(counts_matrix, cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Hit Amplitude'}, annot=True, fmt=".0f")
        plt.title(f'Amplitude in Each Channel, plane {8-plane} XO')
        plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)
        # plt.gca().invert_yaxis()

        # save the plot 
        if save != "false":
            save_path = f"Plots\\TB2025 Gap\\run {save}\\run {save} event {TLU_number}\\evolution of event {TLU_number} plane {8-plane}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()























# histogram of counts for each amp in a specific plane
def amp_histo_single_plane(hit_data, plane):

    # change index so that the first plane is 0 and the last is 7
    plane = 8 - plane

    # get the data of the wanted plane
    hit_plane = hit_data[hit_data.plane == plane]

    # create an array of only the amplitudes in the wanted plane
    hit_plane_amp = hit_plane.amp

    # create and plot an histo to count how many time did we get each amp
    counts, bins, patches = plt.hist(ak.flatten(hit_plane_amp), bins=501, range=(0,500))
    max_bin_index = np.argmax(counts)
    peaks, _ = find_peaks(counts)
    peak_x = (bins[peaks] + bins[peaks + 1]) / 2

    # get the most common amp
    max_bin_center = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
    max_inputs = np.round(max_bin_center)

    # plot settings
    # plt.axvline(max_inputs, color='red', linestyle='--', label= max_inputs)
    colors = ['green', 'blue', 'orange', 'purple']
    for i, px in enumerate(peak_x[:len(colors)]):
        plt.axvline(px, color=colors[i], linestyle='--', linewidth=1, label= np.round(px))
    plt.legend()
    plt.grid(which='major', linestyle='-', linewidth=0.7)
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.title(f'Amplitude of Hits Counter, plane {8 - plane} XO', fontsize=16)
    plt.xlabel('Amplitude', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.show()












# histogram of the total amp from an event in a specific plane - sums the amp from all the activated pads in the event
def amp_histo_single_plane_total_event(hit_data, plane):

    # change index so that the first plane is 0 and the last is 7
    plane = 8 - plane

    # get the data of the wanted plane
    hit_plane = hit_data[hit_data.plane == plane]

    # create an array of only the amplitudes in the wanted plane
    hit_plane_amp = hit_plane.amp
    hit_plane_amp_clean = hit_plane_amp[ak.num(hit_plane_amp) > 0]
    sum_plane_amp = ak.sum(hit_plane_amp_clean, axis = 1)


    # create and plot an histo to count how many time did we get each amp
    counts, bins, patches = plt.hist(sum_plane_amp, bins=501, range=(0,500))
    max_bin_index = np.argmax(counts)
    peaks, _ = find_peaks(counts, prominence = 200)
    peak_x = (bins[peaks] + bins[peaks + 1]) / 2

    # get the most common amp
    max_bin_center = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
    max_inputs = np.round(max_bin_center)

    # plot settings
    # plt.axvline(max_inputs, color='red', linestyle='--', label= max_inputs)
    colors = ['green', 'blue', 'orange', 'purple', 'pink']
    for i, px in enumerate(peak_x[:len(colors)]):
        plt.axvline(px, color=colors[i], linestyle='--', linewidth=1, label= np.round(px))
    plt.legend()
    plt.grid(which='major', linestyle='-', linewidth=0.7)
    plt.grid(which='minor', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.title(f'Amplitude of Hits Counter, plane {8 - plane} XO', fontsize=16)
    plt.xlabel('Amplitude', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.show()












"Shower Properties - avg energy per plane, Shower initiation plane"







# Energy vs plane in a graph
def average_amp_vs_plane(hit_data):

    # total amount of events in the run(TLU's)
    events = len(hit_data)

    # list of planes
    planes = np.arange(1,9,1)

    # create a list of total energy(amp) for each plane
    plane_avg_amp_list = []
    run_avg_amp_list = []
    plane_hits_amount_list = []

    # get the amp for each plane
    for plane in range(7,-1,-1):
    
        # get only the hits on the wanted plane
        hits_plane_n = hit_data[hit_data.plane == plane]
        hits_plane_n_amp = hits_plane_n.amp
        clean_plane_n_amp = hits_plane_n_amp[ak.num(hits_plane_n_amp) > 0]

        # counts the amplitude for every plane
        plane_total_amp = ak.sum(clean_plane_n_amp)             # sum all the amplitudes

        # get the average over all events in the run
        run_avg_amp = plane_total_amp / events
        run_avg_amp_list.append(run_avg_amp)

        # get the average over all the hits in the specific plane
        plane_avg_amp = plane_total_amp / len(clean_plane_n_amp)
        plane_avg_amp_list.append(plane_avg_amp)

        # total amount of hits in each plane
        plane_hits_amount_list.append(len(ak.flatten(clean_plane_n_amp)))
        print(f"amount of hits in plane {7 - plane}:", len(clean_plane_n_amp))

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # plot the averages for each plane
    ax1.plot(planes, run_avg_amp_list, marker='o', label = "AMP over events in the run")
    ax1.plot(planes, plane_avg_amp_list, marker='v', label = "AMP over hits in plane")
    ax1.set_xlabel('Plane [XO]')
    ax1.set_ylabel('AVG AMP')
    ax1.set_title('AMP/total events for each plane')
    ax1.grid(True)
    ax1.legend()
    
    # show the amounnt of hits in each plane on a bar chart
    ax2.bar(planes, plane_hits_amount_list, color='blue')
    ax2.set_title('Amount of Hits in Each Plane')
    ax2.set_xlabel('Planes [XO]')
    ax2.set_ylabel('amount of hits')
    ax2.grid(True)

    plt.show()


        




















# plot the percentage of events with readings only after a specific plane.
def plot_empty_first_planes(hit_data):

    total_amount_of_events = len(hit_data)
    first_occupied_plane_list = list(range(8,0,-1))
    number_of_events_list = []
    percentage_of_events_list = []

    
    for occupied_plane in range(7,-1,-1):

        # start filltering the data plane by plane until we get to the first occupied plane
        filtered_events = hit_data

        # repeat choosing the events with the following planes being empty until we reach the wanted amount of empty first planes
        empty_plane = 7
        while empty_plane > occupied_plane:
            filtered_events = filtered_events[ak.all(filtered_events.plane != empty_plane, axis=1)]
            empty_plane -= 1

        # from the filtered data, take only the events with the next plane being non empty
        hits_plane = filtered_events[filtered_events.plane == occupied_plane]
        hits_plane_clean = hits_plane[ak.num(hits_plane) > 0]
        
        # count the amount of events with the wanted first occupied plane
        number_of_events_plane = len(hits_plane_clean)
        number_of_events_list.append(number_of_events_plane)

        # find the percentage it makes of the total amount of events
        percent_of_events_plane = (number_of_events_plane / total_amount_of_events) * 100
        percentage_of_events_list.append(percent_of_events_plane)

    print("total percentage of events:", sum(percentage_of_events_list))


    # reverse the percentage list to get the first plant to be indexed as 0
    percentage_list_reverse = percentage_of_events_list[::-1]

    # plot the data
    bar_container = plt.bar(first_occupied_plane_list, percentage_list_reverse, color = "red")
    plt.bar_label(bar_container,  fmt='{:,.2f}')
    plt.xlabel('Shower Starting Plane [XO]')
    plt.ylabel('Percentage of Events (%)')
    plt.title('The Percent of Events VS Number of First Empty Planes')
    plt.grid(True)
    # plt.legend()
    
    





    





















"Spatial dependence Analysis"










# determine the initial position of the shower (x,y)
def initial_X_position_DUT(hit_data, return_y = "false"):
    
    # get only showers starting at the first plane
    plane_7 = hit_data[hit_data.plane == 7]
    mask = ak.num(plane_7) == 1

    # get the channels data of the first plane
    plane_7_clean = plane_7[mask]
    plane_7_channel = plane_7_clean.ch
    
    # convert to x, y positions
    y, x = divmod(plane_7_channel, 20) #y is the quontinent and is the row, x is the remainder and column
    
    # take the average of the X positions
    x_list = x.to_list()
    x_ak = ak.Array(x_list)
    x_avg = ak.mean(x_ak, axis = 1)

    # take the average of the Y positions
    if return_y == "true":
        y_list = y.to_list()
        y_ak = ak.Array(y_list)
        y_avg = ak.mean(y_ak, axis = 1)
        return x_avg, y_avg

    else:
        return x_avg
    





















def event_shower_energy_vs_X_position(hit_data, single_pad_only = "true", specific_Y = "false", Log_Scale = False):
    
    # get only showers starting at the first plane to identify the initial location
    plane_7 = hit_data[hit_data.plane == 7]
    if single_pad_only == "false":
        mask = ak.num(plane_7) > 0

    if single_pad_only == "true":
        mask = ak.num(plane_7) == 1

    first_plane_starting_events = hit_data[mask]

    # determine the initial location of the shower
    # get the data on the first plane
    plane_7_clean = plane_7[mask]
    plane_7_channel = plane_7_clean.ch
    
    # get the x and y positions of each channel
    y, x = divmod(plane_7_channel, 20) #y is the quontinent and is the row, x is the remainder and column
    
    # make x and y one dimensional
    x_list = x.to_list()
    x_ak = ak.Array(x_list)
    x_avg = ak.mean(x_ak, axis = 1)

        
    # compute the shower energy for each event
    hit_amp_array = first_plane_starting_events.amp
    event_shower_amp_array = ak.sum(hit_amp_array, axis = 1)

    # calculate for a specific row if needed
    if specific_Y  != "false":
        y = ak.flatten(y)
        mask_Y = y == specific_Y
        x_avg = x_avg[mask_Y]
        event_shower_amp_array = event_shower_amp_array[mask_Y]

    # get the average shower energy for each X position
    div, avg_amps, classes = rf.ak_groupby(x_avg, event_shower_amp_array)
    
    # get the energies in all column (2d array)
    energies_per_column = div.data
    
    # get the statistics for each column
    means = ak.mean(energies_per_column, axis=1)
    sigmas = ak.std(energies_per_column, axis=1)

    # mean unsertinty
    sem = sigmas / np.sqrt(ak.num(energies_per_column-1, axis=1))

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # plot the energy avg per position vs the initial X position of the shower
    ax1.errorbar(
    classes, avg_amps, yerr=sem, fmt='-o',ecolor='black', elinewidth=1, capsize=4, capthick=1, markerfacecolor='blue', markersize=6, label='Data with error')

    ax1.set_xticks(np.arange(0, 20))
    ax1.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax1.set_xlabel('X Position at Shower Initiation [Pad Column]')
    ax1.set_ylabel('AVG Shower Energy')
    
    # set title for chosen Y
    if specific_Y == "false":
        ax1.set_title('Average Shower Energy vs Initial Location')
    
    else:
        ax1.set_title(f'Average Shower Energy vs Initial Location, y = {specific_Y}')

    # show the amounnt of hits in each plane on a bar chart
    bins = np.arange(0, 21, 1) 
    ax2.hist(x_avg, bins=bins, edgecolor='black', rwidth=0.8)
    ax2.set_xticks(np.arange(0, 20) + 0.5)  # shift by 0.5
    ax2.set_xticklabels(np.arange(0, 20)) 
    ax2.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    ax2.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax2.set_xlabel('X Position [Pad Column]')
    ax2.set_ylabel('amount of hits')
    ax2.set_title('Amount of Events initiating in Each Column of the Sensor')
    if Log_Scale:
        ax2.set_yscale("log")

    
    plt.show()






















# get the k columns with the maximum amount of hits
def columns_with_max_hits(hit_data, number_of_columns):

    # get the initial x positions
    x_avg = initial_X_position_DUT(hit_data)
    # x_avg = ak.round(x_avg)

    x_avg_np = ak.to_numpy(x_avg)

    # Compute histogram for X positions
    counts, bin_edges = np.histogram(x_avg, bins=np.arange(0, 21))

    # get the positions with the highest entries
    bins = np.arange(0, 20)

    # Get indices that would sort counts descending
    top_indices = np.argsort(-counts)[:number_of_columns]

    # Get the corresponding bin numbers
    top_bins = bins[top_indices]
    top_counts = counts[top_indices]

    return top_bins
    
























































# average ENERGY in a shower vs columns and planes
def avg_ENERGY_vs_plane_per_X_position(hit_data, number_of_highest_ocupied_columns, print_energies = "false"):
    
    planes = np.arange(0,8)

    # attach the positions to the data
    positions = initial_X_position_DUT(hit_data)
    plane_7 = hit_data[hit_data.plane == 7]
    mask = ak.num(plane_7) == 1
    events_starting_at_7 = hit_data[mask]
    hit_data_positions = ak.zip({ "hits":events_starting_at_7, "positions":positions},depth_limit=1)
    
    # get the most ocupied columns
    top_columns = columns_with_max_hits(hit_data, number_of_highest_ocupied_columns)
    top_columns = np.sort(top_columns)
    print(top_columns)

    # array to store all the data of planes per column
    total_avg_energy_planes = []

    # create to plot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for column in top_columns:
        print(column)
        # creat an array to store the average ENERGY in each plane
        energy_plane_array = []

        # get the data of events initiating in the wanted column
        hit_data_column = hit_data_positions[(hit_data_positions.positions >= column) & (hit_data_positions.positions < column + 1)]


        # find the amount of hits in a plane for the column and add to the array
        for plane in planes:
            plane = 7-plane #adjust index so the first plane is 0
            hit_data_column_plane = hit_data_column.hits[hit_data_column.hits.plane == plane] # the events initiatin at the wanted column and a specific plane 
            hit_data_column_plane = hit_data_column_plane[ak.num(hit_data_column_plane) > 0] # clean from empty entries
            num_of_events_column_plane = len(hit_data_column_plane)
            energy_column_plane = ak.sum(hit_data_column_plane.amp)
            avg_shower_energy_of_event_column_plane = energy_column_plane / num_of_events_column_plane
            energy_plane_array.append(avg_shower_energy_of_event_column_plane)

        #  add the array of energies for a single column to the total data array.
        total_avg_energy_planes.append(energy_plane_array)
        
        # print the energies per plane if needed
        if print_energies == "true":
            print(energy_plane_array)

        # plot avg Shower Energy of hits per plane
        ax1.plot(planes + 1, energy_plane_array, label=f"X Position: {column} Column", marker=".")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.set_xlabel("Plane [XO]")
        ax1.set_ylabel("Average Shower Energy")
        ax1.set_title("Energy Distribution in the Sensor for Differrnt Columns")
        ax1.legend()

    
    # plot avg Shower Energy of hits per position
    total_avg_energy_columns = np.transpose(np.array(total_avg_energy_planes))
    for plane in planes:
        avg_energy_plane_column = total_avg_energy_columns[plane]
        ax2.plot(top_columns, avg_energy_plane_column, label = f"Plane: {plane + 1}", marker="D")
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Average Shower Energy')
        ax2.set_title('Plane Energy in Each Column')
        ax2.grid(True)
        ax2.legend()

    fig.suptitle("Average Shower Energy for Different Planes, for Events Initiating in Different Positions", fontsize=14)
    plt.show()


































# returns only the events with one hit on the first plane, with a section of radius from the position in each hit
def Radii_from_Initial_position(hit_data):
    
    # reduce data for events starting at 0 plane and acivated one pad only
    plane_7 = hit_data[hit_data.plane == 7]
    mask = ak.num(plane_7) == 1
    first_plane_starting_events = hit_data[mask]

    # get the initial position for each event
    x_init, y_init =  initial_X_position_DUT(first_plane_starting_events, return_y = "true")

    # find the (x,y) ch position for every hit 
    y_pos, x_pos = divmod(first_plane_starting_events.ch, 20)
    
    # get the distances from the initial position
    x_dist = x_pos - x_init
    y_dist = y_pos - y_init

    # get the radii
    radii_squared = x_dist**2 + y_dist**2
    radii = np.sqrt(radii_squared)

    # add the radii section to the data
    events_from_first_plane_with_Radii = ak.with_field(first_plane_starting_events, radii, "Distance")

    return events_from_first_plane_with_Radii




























# Histograms of shower ENERGIES of events from the most occupied columns
def Histo_shower_energy_for_X_position(hit_data, number_of_highest_ocupied_columns, single_pad_only = "true", specific_Y = "false", normalize = "false"):
    
    # get only showers starting at the first plane to identify the initial location
    plane_7 = hit_data[hit_data.plane == 7]
    if single_pad_only == "false":
        mask = ak.num(plane_7) > 0

    if single_pad_only == "true":
        mask = ak.num(plane_7) == 1

    first_plane_starting_events = hit_data[mask]

    # determine the initial location of the shower

    # get the data on the first plane
    plane_7_clean = plane_7[mask]
    plane_7_channel = plane_7_clean.ch
    
    # divide by x positions
    y, x = divmod(plane_7_channel, 20) #y is the quontinent and is the row, x is the remainder and column
    x_list = x.to_list()
    x_ak = ak.Array(x_list)
    x_avg = ak.mean(x_ak, axis = 1)
    
    # compute the shower energy for each event
    hit_amp_array = first_plane_starting_events.amp
    event_shower_amp_array = ak.sum(hit_amp_array, axis = 1)

    # Filter for specific row only
    if specific_Y  != "false":
        y = ak.flatten(y)
        mask_Y = y == specific_Y
        x_avg = x_avg[mask_Y]
        event_shower_amp_array = event_shower_amp_array[mask_Y]
        

    # get the average shower energy for each X position
    div, avg_amps, classes = rf.ak_groupby(x_avg, event_shower_amp_array)

    # get the most ocupied columns
    top_columns = rf.columns_with_max_hits(hit_data, number_of_highest_ocupied_columns)
    top_columns = np.sort(top_columns)
    

        
    
    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram of energy per position
    for column in top_columns:
        Column_data = div[div.classes == column]
        energies_in_column = ak.flatten(Column_data[ak.num(Column_data) > 0].data) #get only the amplitudes of the needed column
        print(column, len(energies_in_column))
        print("########")
        
        # find the peak of each histo
        counts, bins = np.histogram(energies_in_column, bins=200)
        peak_idx = np.argmax(counts)
        peak_center = (bins[peak_idx] + bins[peak_idx + 1]) / 2


        # plot the histo
        if normalize == "false":
            ax1.hist(energies_in_column, bins=200, histtype='step', label= f'Column: {column}, Peak = {peak_center:.2f}')
        
        if normalize == "true":
            ax1.hist(energies_in_column, bins=200, histtype='step', density=True, label= f'Column: {column}, Peak = {peak_center:.2f}')
    
    # ax1.set_xticks(np.arange(0, 20))
    ax1.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_xlabel('Energy [ADC]')
    ax1.set_ylabel('Counts')

    if specific_Y == "false":
        ax1.set_title('Energy Histograms for events starting at different initial columns')
    
    else:
        ax1.set_title(f'Energy Histograms for events starting at different initial columns, y = {specific_Y}')

    # show the amounnt of hits in each plane on a bar chart
    bins = np.arange(0, 21, 1) 
    ax2.hist(ak.round(x_avg), bins=bins, edgecolor='black', rwidth=0.8)
    ax2.set_xticks(np.arange(0, 20) + 0.5)  # shift by 0.5
    ax2.set_xticklabels(np.arange(0, 20)) 
    ax2.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    ax2.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax2.set_xlabel('X Position [Pad Column]')
    ax2.set_ylabel('amount of hits')
    ax2.set_title(f'Amount of Events initiating in Each Column of the Sensor')
    
    plt.show()






































# Returns a list of the distance (in terms of pad) where a specific percentage of the total energy is contained in each event
def frac_contained_energy_radius(data_with_distances, energy_percentage):

    # get the distances and amplitudes of the hits and organise them from smallest distance to furthest
    zipped_arrays = ak.zip({ "R":data_with_distances.Distance, "amp":data_with_distances.amp})
    data_classes_sorted = zipped_arrays[ak.argsort(zipped_arrays.R)] # sort by distances
    R = data_classes_sorted.R
    amp = data_classes_sorted.amp

    # total energy in each event
    total_amp = ak.sum(amp, axis = 1)
    
    # summ cumulative energy in each hit of the event
    cumulative_amp = rf.ak_cumsum_per_sublist(amp)

    # fraction of the energy in every hit
    f = cumulative_amp / total_amp  

    # get the index where the cumulative amplitude is bigger the the wanted energy percentage
    mask = f >= energy_percentage
    R_beyond_percent = R[mask]
    R_at_percent = ak.firsts(R_beyond_percent)


    return R_at_percent
    
































def frac_energy_radii_histo(hit_data_with_distances, fraction_of_energy, calculate_R=False, bin_size = 0.5):

    if calculate_R:
        hit_data_with_distances = rf.Radii_from_Initial_position(hit_data_with_distances)

    R_at_frac = rf.frac_contained_energy_radius(hit_data_with_distances, fraction_of_energy)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, ak.max(R_at_frac) + bin_size, bin_size)
    counts, edges, patches = ax1.hist(R_at_frac, bins=bins, color='skyblue', edgecolor='black')

    ax1.set_xlabel("Radius [Pad Units]")
    ax1.set_ylabel("Counts")
    ax1.set_title(f"Radii containing {fraction_of_energy} of the energy in event")
    ax1.set_xticks(np.arange(0, np.max(R_at_frac) + 1, 1))
    ax1.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, alpha=0.7)

    ax1.yaxis.set_minor_locator(plt.MultipleLocator((ax1.get_yticks()[1] - ax1.get_yticks()[0]) / 2))
    ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.grid(True, which='minor', axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

    # --- Secondary y-axis (percentages) ---
    ax2 = ax1.twinx()
    total = np.sum(counts)                # total number of events
    ax2.set_ylim(ax1.get_ylim())          # match limits to left axis
    ax2.set_ylabel("Percentage out of all events")

    # Find the maximum visible count on the left axis
    ymax = ax1.get_ylim()[1]

    # Create tick positions corresponding to 5% increments of *total counts*
    max_percent = 100 * ymax / total      # how far up the left axis goes in %
    percent_ticks = np.arange(0, max_percent + 5, 5)   # every 5% up to that value
    count_ticks = percent_ticks / 100 * total           # convert % back to counts

    # Apply these ticks and labels to the right axis
    ax2.set_yticks(count_ticks)
    ax2.set_yticklabels([f"{p:.0f}%" for p in percent_ticks])


    plt.tight_layout()
    plt.show()



















# returns histogram of the energie of events starting at the stated X position
def shower_energy_histo_single_location(hit_data, Position, specific_Y = "all_rows", bin_size = 50):

    # get only showers starting at the first plane to identify the initial location
    plane_7 = hit_data[hit_data.plane == 7]
    mask = ak.num(plane_7) == 1
    first_plane_starting_events = hit_data[mask]

    # determine the initial location of the shower
    # get the data on the first plane
    plane_7_clean = plane_7[mask]
    plane_7_channel = plane_7_clean.ch

    # divide by x positions
    y, x = divmod(plane_7_channel, 20) #y is the quontinent and is the row, x is the remainder and column
    x_list = x.to_list()
    x_ak = ak.Array(x_list)
    x_avg = ak.mean(x_ak, axis = 1)
    
    # compute the shower energy for each event
    hit_amp_array = first_plane_starting_events.amp
    event_shower_amp_array = ak.sum(hit_amp_array, axis = 1)

    # Filter for specific row only
    if specific_Y  != "all_rows":
        y = ak.flatten(y)
        mask_Y = y == specific_Y
        x_avg = x_avg[mask_Y]
        event_shower_amp_array = event_shower_amp_array[mask_Y]
        title = f'Energy Histograms for events starting at different initial columns, y = {specific_Y}'
    
    
    
    # get the shower energy for the X position for all events
    amps_divided_by_class, avg_amps, classes = rf.ak_groupby(x_avg, event_shower_amp_array)
    
    amps_class_position = amps_divided_by_class[amps_divided_by_class.classes == Position]
    amps_position = amps_class_position.data
    amps_position_clean = ak.flatten(amps_position[ak.num(amps_position) > 0])
    
    

    # Histogram
    max_range = 12000
    range = (0, max_range)
    bins = np.arange(0, max_range + 1, bin_size)
    
    # most common energy (peak of the histo)
    counts, bins = np.histogram(amps_position_clean, bins = bins, range=range)
    peak_idx = np.argmax(counts)
    peak_center = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    

    # Histo Mean
    avg_amp = ak.mean(amps_position_clean)
    
    # Histo Standard deveiation (how spread is the data) marked as sigma
    std = ak.std(amps_position_clean)

    # standard deviation error of mean (how uncertain the mean is) marked as sigma_mean
    SEM = std/np.sqrt(len(amps_position_clean))
    


    # Multi-line string for the box
    textstr = '\n'.join((
        f'Mean = {avg_amp:.2f}',
        f'Peak = {peak_center:.2f}',
        rf'$\sigma$ = {std:.2f}',
        rf'$\sigma_{{\mu}}$ = {SEM:.2f}',
        # f'RMS = {RMS:.2f}',
        # f'STD = {STD:,}'
    ))

    # Place the box inside the axes
    plt.text(
        0.65, 0.75, textstr, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )


    # plot the histogram with matplotlib
    plt.hist(amps_position_clean, bins = bins, range=range, label=f"avg = {avg_amp} \n peak = {peak_center}")
    plt.grid()
    plt.xlabel(f"Energy in Column {Position}")
    plt.ylabel("Counts")
    plt.title(f'Energy Histograms for events starting at different initial columns, y =  {specific_Y}')
    plt.show()






























'Statistics'




# returns statistic values of histogram for shower energies of events starting at specific column
def histo_statistics_single_column(hit_data, Position, specific_Y = "all_rows"):

    # get only showers starting at the first plane to identify the initial location
    plane_7 = hit_data[hit_data.plane == 7]
    mask = ak.num(plane_7) == 1
    first_plane_starting_events = hit_data[mask]

    # determine the initial location of the shower
    # get the data on the first plane
    plane_7_clean = plane_7[mask]
    plane_7_channel = plane_7_clean.ch

    # divide by x positions
    y, x = divmod(plane_7_channel, 20) #y is the quontinent and is the row, x is the remainder and column
    x_list = x.to_list()
    x_ak = ak.Array(x_list)
    x_avg = ak.mean(x_ak, axis = 1)
    
    # compute the shower energy for each event
    hit_amp_array = first_plane_starting_events.amp
    event_shower_amp_array = ak.sum(hit_amp_array, axis = 1)

    # Filter for specific row only
    if specific_Y  != "all_rows":
        y = ak.flatten(y)
        mask_Y = y == specific_Y
        x_avg = x_avg[mask_Y]
        event_shower_amp_array = event_shower_amp_array[mask_Y]
        
    # get the shower energy for the X position for all events
    amps_divided_by_class, avg_amps, classes = rf.ak_groupby(x_avg, event_shower_amp_array)
    
    amps_class_position = amps_divided_by_class[amps_divided_by_class.classes == Position]
    amps_position = amps_class_position.data
    energies_in_column = ak.flatten(amps_position[ak.num(amps_position) > 0])
    
    # Histogram
    max_range = 12000
    range = (0, max_range)
    bin_size = 50
    bins = np.arange(0, max_range + 1, bin_size)
    
    # most common energy (peak of the histo)
    counts, bins = np.histogram(energies_in_column, bins = bins, range=range)
    peak_idx = np.argmax(counts)
    peak_energy = (bins[peak_idx] + bins[peak_idx + 1]) / 2
    
    # Histo Mean
    mean_energy = ak.mean(energies_in_column)
    
    # Histo Standard deveiation (how spread is the data) marked as sigma
    std = ak.std(energies_in_column)

    # standard deviation error of mean (how uncertain the mean is) marked as sigma_mean
    sem = std/np.sqrt(len(energies_in_column))
    
    return mean_energy, std, sem, peak_energy, energies_in_column










































# Gaussian fit

# returns the optimized fit values for gaussian fit for histogram
def fit_histo_to_gaussian(histo_data, bin_size):
    
    # convert data to numpy in order to fit
    histo_data = ak.to_numpy(histo_data)

    # bins paramaters
    max_range = 12000
    range = (0, max_range)
    bins = np.arange(0, max_range + 1, bin_size)

    # histogram data
    counts, bin_edges = np.histogram(histo_data, bins=bins, range = range)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Define Gaussian function
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Fit the Gaussian to histogram data 
    popt, pcov = curve_fit(gaussian, bin_centers, counts, p0=[1, np.mean(histo_data), np.std(histo_data)],bounds = ([0, 0, 0], [max(counts), np.inf, np.inf]))
    A_fit, mu_fit, sigma_fit = popt
    gaussian_Y = gaussian(bin_centers, *popt)
    return A_fit, mu_fit, sigma_fit, bin_centers, gaussian_Y





























# plot histogram with fit and statistics values
def Single_column_energy_histo_Gaussian_fit(hit_data, Position, specific_Y = "all_rows", bin_size = 50):

    # data statistics
    mean_energy, std, sem, peak_energy, energies_in_column = histo_statistics_single_column(hit_data, Position, specific_Y)
    counts = len(energies_in_column)

    # optimized gaussian values for the histo
    A_fit, mu_fit, sigma_fit, bin_centers, gaussian_Y = fit_histo_to_gaussian(energies_in_column, bin_size)

    # Multi-line string for the box
    textstr = '\n'.join((
        # histo
        rf'$\bf{{Histogram:}}$',
        f'Entries = {counts:.2f}',
        f'Mean = {mean_energy:.2f}',
        rf'$\sigma$ = {std:.2f}',
        rf'$\sigma_{{\mu}}$ = {sem:.2f}',
        f'Peak = {peak_energy:.2f}',
        '\n',
        # gaussian
        rf'$\bf{{Gaussian:}}$',
        f'Mean = {mu_fit:.2f}',
        rf'$\sigma$ = {sigma_fit:.2f}',
        rf'$\sigma_{{\mu}}$ = {sigma_fit/np.sqrt(counts):.2f}',    
    ))

    # Place the box inside the axes
    plt.text(
        0.98, 0.98, textstr,
        transform=plt.gca().transAxes,      # axes coordinates: (0,0) bottom-left, (1,1) top-right
        fontsize=10,
        verticalalignment='top',     # anchor the top of the box
        horizontalalignment='right', # anchor the right side of the box
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )


    # Histogram
    max_range = 12000
    range = (0, max_range)
    bins = np.arange(0, max_range + 1, bin_size)
    title = f'Energy Histograms for events starting at different initial columns, y = {specific_Y}'

    # plot the histogram with matplotlib
    plt.hist(energies_in_column, bins = bins, range=range)
    plt.plot(bin_centers, gaussian_Y, 'r-', linewidth=2, label='Gaussian fit')
    plt.grid()
    plt.xlabel(f"Energy in Column {Position}")
    plt.ylabel("Counts")
    plt.title(f'Energies of Events Starting at Column {Position}, y =  {specific_Y}, \n Gaussian fit')
    plt.show()

























# Gamma fit



# returns the optimized fit values for Gamma
def fit_histo_to_Gamma(histo_data, bin_size):
    
    # convert data to numpy in order to fit
    histo_data = ak.to_numpy(histo_data)

    # statistics
    mean = np.mean(histo_data)
    sigma = np.std(histo_data)

    # bins paramaters
    max_range = 12000
    data_range = (0, max_range)
    bins = np.arange(0, max_range + 1, bin_size)

    # histogram data
    # counts, bin_edges = np.histogram(histo_data, bins=bins, range = data_range)
    counts, bin_edges = np.histogram(histo_data, bins=bins, range = data_range, density = True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Define Gaussian function
    def Gamma_pdf(x, A, alpha, theta):
        return A * 1/(gamma(alpha)*theta**alpha) * x**(alpha-1) * np.exp(-x/theta)
    
    # def Gamma_pdf(x, alpha, theta):
    #     return 1/(gamma(alpha)*theta**alpha) * x**(alpha-1) * np.exp(-x/theta)
    
    # Fit the Gaussian to histogram data 
    popt, pcov = curve_fit(Gamma_pdf, bin_centers, counts, p0=[1, (mean/sigma)**2, (sigma**2)/mean],bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]))
    A_fit, alpha_fit, theta_fit = popt
    Gamma_Y = Gamma_pdf(bin_centers, *popt)
    return A_fit, alpha_fit, theta_fit, bin_centers, Gamma_Y



























# plot histogram with fit and statistics values
def single_column_energy_Gamma_fit(hit_data, Position, specific_Y = "all_rows", bin_size = 50):

    # data statistics
    mean_energy, std, sem, peak_energy, energies_in_column = rf.histo_statistics_single_column(hit_data, Position, specific_Y)
    counts = len(energies_in_column)

    # optimized gaussian values for the histo
    A_fit, alpha_fit, theta_fit, bin_centers, Gamma_Y = rf.fit_histo_to_Gamma(energies_in_column, bin_size)
    
    # Fit statistics
    mu_fit = alpha_fit*theta_fit
    sigma_fit = np.sqrt(alpha_fit) * theta_fit

    # Multi-line string for the box
    textstr = '\n'.join((
        # histo
        rf'$\bf{{Histogram:}}$',
        f'Entries = {counts:.2f}',
        f'Mean = {mean_energy:.2f}',
        rf'$\sigma$ = {std:.2f}',
        rf'$\sigma_{{\mu}}$ = {sem:.2f}',
        f'Peak = {peak_energy:.2f}',
        '\n',
        # Gamma
        rf'$\bf{{Gamma \quad pdf:}}$',
        f'Mean = {mu_fit:.2f}',
        rf'$\sigma$ = {sigma_fit:.2f}',
        rf'$\sigma_{{\mu}}$ = {sigma_fit/np.sqrt(counts):.2f}',
        ))


    # place the text in the plot
    plt.text(
        0.98, 0.98, textstr,
        transform=plt.gca().transAxes,      # axes coordinates: (0,0) bottom-left, (1,1) top-right
        fontsize=10,
        verticalalignment='top',     # anchor the top of the box
        horizontalalignment='right', # anchor the right side of the box
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Histogram
    max_range = 12000
    range = (0, max_range)
    bins = np.arange(0, max_range + 1, bin_size)
    title = f'Energy Histograms for events starting at different initial columns, y = {specific_Y}'

    # plot the histogram with matplotlib
    plt.hist(energies_in_column, bins = bins, range=range, density = True)
    plt.plot(bin_centers, Gamma_Y, 'r-', linewidth=2, label='Gaussian fit')
    plt.grid()
    plt.xlabel(f"Energy in Column {Position}")
    plt.ylabel("Counts")
    # plt.title(f'Energy Histograms for events starting at different initial columns, y =  {specific_Y}')
    plt.title(f'Energies of Events Starting at Column {Position}, y =  {specific_Y}, \n Gamma fit')
    plt.show()






























"the following 3 functions work for the pad reconstruction, try to add the data in batches and create the zipped arrays only in the end"




# reconstructed data for dead pads - returns an array with the reconstruccted value of the pad in each event
def pads_reconstruct(hit_data, plane, channel, plot = False, yscale = "linear"):

    # get the data from previous plane
    previous_plane = hit_data[hit_data.plane == plane - 1]
    prev_plane_same_ch = previous_plane[previous_plane.ch == channel]
    prev_plane_same_ch_zeros = ak.where(ak.num(prev_plane_same_ch.amp) != 1, [[0]], prev_plane_same_ch.amp)

    # get the data from plane 5
    next_plane = hit_data[hit_data.plane == plane + 1]
    next_plane_same_ch = next_plane[next_plane.ch == channel]
    next_plane_same_ch_zeros = ak.where(ak.num(next_plane_same_ch.amp) != 1, [[0]], next_plane_same_ch.amp)
    
    # average the data 
    reco_ch_data = (next_plane_same_ch_zeros + prev_plane_same_ch_zeros) / 2

    if plot:
        # plot
        data_reco = np.array(reco_ch_data)
        print(data_reco.min())
        # Define bin width
        bin_width = 25
        # Create bins from min to max with step of 5
        bins = np.arange(data_reco.min(), data_reco.max() + bin_width, bin_width)
        # Plot histogram
        plt.figure()
        plt.hist(data_reco, bins=bins, alpha = 1, label = "Reconstructed")
        plt.grid(True)
        plt.legend()
        plt.xlim(0, data_reco.max())
        # Labels
        plt.xlabel("Energy")
        plt.ylabel("Counts")
        plt.title(f"plane {plane} ch {channel} -Origin and reconstructed data, Z axis avg")
        # plt.yscale("log")
        plt.yscale(yscale)
        plt.show()

    return reco_ch_data





















# add the reconstructed data of the dead pad to the hits data
def add_reconstruct_data_single_dead_pad(data, plane, pad, scope_data = True):

    if scope_data:
        hit_data = data.hits

    # get the array of the reconstructed data
    reco_pad = pads_reconstruct(hit_data, plane, pad)

    # turn empty data in the reconstructed results to zeros
    reco_amp = ak.fill_none(ak.firsts(reco_pad), 0.0)

    # zip the reconstructed data to plane and ch of dead pad
    reco_pad_plane_ch = ak.zip(
        {
            "plane": ak.full_like(reco_amp, plane, dtype="int32"),
            "ch":    ak.full_like(reco_amp, pad, dtype="int32"),
            "amp":   reco_amp,
        })

    # turn the reconstructed data into subarrays per event
    reco1 = ak.unflatten(reco_pad_plane_ch, 1)

    # turn all zeros to empty spaces
    keep = reco_amp != 0
    reco_to_add = ak.fill_none(ak.mask(reco1, keep), [])

    # combine with original data
    hits_with_reco_pad = ak.concatenate([hit_data, reco_to_add], axis=1)

    data_with_reco_pad = ak.with_field(data, hits_with_reco_pad, "hits")

    return data_with_reco_pad













"works!"
# reconstruct the dead channels around the shower in a run - radius of reco dtermines the distance from the center of the shower to be reconstructed
def reconstruct_data_all_dead_pads(data, radius, path_to_diagnostics, number_of_planes = 8):

# find the center of the shower

    # single hit in the first plane
    first_plane1 = data.hits[data.hits.plane == 0]
    first_plane = first_plane1[ak.num(first_plane1) == 1]

    # ch activated in the first plane
    first_plane_ch = first_plane.ch

    # define center of the shower as the most activated pad in the first plane 
    counted_channels, counts = np.unique(ak.flatten(first_plane_ch), axis=0, return_counts=True)
    imax = np.argmax(counts)
    central_pad = counted_channels[imax]
    print("shower center:", central_pad)   
    

# get the dead channels list

    # list of all dead channels
    all_dead_channels = rf.channels_diagnostics(path_to_diagnostics, number_of_planes)
    print("amount of dead channels:", len(all_dead_channels))
    
    # get the pads in the wanted radius
    base = list(range(central_pad - radius, central_pad + radius +1))
    pads = [x + 20*i for i in range(-radius, radius+1) for x in base]
    
    # get the dead channels only from the wanted radius
    radius_mask = np.isin(all_dead_channels.channel_ID, pads)
    dead_channels_in_radius1 = all_dead_channels[radius_mask] 

    # delete dead channels starting at the first plane as we cant average for them
    dead_channels_in_radius = dead_channels_in_radius1[(dead_channels_in_radius1.plane_ID > 0) & (dead_channels_in_radius1.plane_ID < 7)]


# add the reconstructed data
    counter = len(dead_channels_in_radius)
    for channel in dead_channels_in_radius:
        print(channel)
        data = rf.add_reconstruct_data_single_dead_pad(data, channel.plane_ID, channel.channel_ID)
        counter -= 1
        print(counter, "channels left")
    
    
    return data
