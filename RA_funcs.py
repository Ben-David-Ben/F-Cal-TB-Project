# ROOT ANALYSIS FUNCTIONS

import awkward as ak
import numpy as np
import matplotlib.pylab as plt
import uproot
import seaborn
import RA_funcs as rf
from scipy.signal import find_peaks

print("imports work")



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
def hits_amount_colormap_single_plane(hit_data, plane_number, cmap="berlin"):
    
    #  change index so that the first plane is 0 and last is 7
    plane_number = 7 - plane_number

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

    # creat the colormap
    seaborn.heatmap(counts_matrix, cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Hit Counts'})
    plt.title(f'Number of Hits in each channel, plane {7 - plane_number}')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)




















# a colormap with the total AMPLITUDE of hits in every channel(pad) of the chosen plane
def amp_colormap_single_plane(hit_data, plane_number, cmap="berlin"):

    #  change index so that the first plane is 0 and last is 7
    plane_number = 7 - plane_number

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
    plt.title(f'Total Amplitude in Each Channel, Plane {7 - plane_number}')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)

















# show the average amplitude of the entire run in each pad
def average_amp_colormap_single_plane(hit_data, plane_number, cmap="managua"):

    #  change index so that the first plane is 0 and last is 7
    plane_number = 7 - plane_number

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
    plt.title(f'Average Amplitude in Each Channel, Plane {7 - plane_number}')
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
        plt.title(f'Amplitude in Each Channel, plane {7-plane}')
        plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)
        # plt.gca().invert_yaxis()

        # save the plot 
        if save != "false":
            save_path = f"Plots\\TB2025 Gap\\run {save}\\ evolution of event{TLU_number} plane {7-plane}"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()























# histogram of counts for each amp in a specific plane
def amp_histo_single_plane(hit_data, plane):

    # change index so that the first plane is 0 and the last is 7
    plane = 7 - plane

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
    plt.title(f'Amplitude of Hits Counter, plane {7 - plane}', fontsize=16)
    plt.xlabel('Amplitude', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.show()












# histogram of counts for the total amp of an event in a specific plane
def amp_histo_single_plane_total_event(hit_data, plane):

    # change index so that the first plane is 0 and the last is 7
    plane = 7 - plane

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
    plt.title(f'Amplitude of Hits Counter, plane {7 - plane}', fontsize=16)
    plt.xlabel('Amplitude', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.show()












"Shower Properties - avg energy per plane, Shower initiation plane"







# Energy vs plane in a graph
def average_amp_vs_plane(hit_data):

    # total amount of events in the run(TLU's)
    events = len(hit_data)

    # list of planes
    planes = np.arange(0,8,1)

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
    ax1.set_xlabel('Plane')
    ax1.set_ylabel('AVG AMP')
    ax1.set_title('AMP/total events for each plane')
    ax1.grid(True)
    ax1.legend()
    
    # show the amounnt of hits in each plane on a bar chart
    ax2.bar(planes, plane_hits_amount_list, color='blue')
    ax2.set_title('Amount of Hits in Each Plane')
    ax2.set_xlabel('Planes')
    ax2.set_ylabel('amount of hits')
    ax2.grid(True)

    plt.show()


        




















# plot the percentage of events with readings only after a specific plane.
def plot_empty_first_planes(hit_data):

    total_amount_of_events = len(hit_data)
    first_occupied_plane_list = list(range(7,-1,-1))
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
    plt.xlabel('First Occupied Plane')
    plt.ylabel('Percentage of Events (%)')
    plt.title('The Percent of Events VS Number of First Empty Planes')
    plt.grid(True)
    # plt.legend()
    
    





    





















"Spatial dependence Analysis"



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
    
















# Shower energy for different initial X positions of the shower
def event_shower_energy_vs_X_position(hit_data, single_pad_only = "false"):
    
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

    # get the average shower energy for each X position
    div, avg_amps, classes = ak_groupby(x_avg, event_shower_amp_array)


    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # plot the energy avg per position vs the initial X position of the shower
    ax1.plot(classes, avg_amps, marker='o')
    ax1.set_xticks(np.arange(0, 20))
    ax1.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax1.set_xlabel('X Position at Shower Initiation [Pad Column]')
    ax1.set_ylabel('AVG Shower Energy')
    ax1.set_title('Average Shower Energy vs Initial Location')

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
    
































# average amount of hits in a shower vs columns and planes
def avg_hit_amount_vs_plane_per_X_position(hit_data, number_of_highest_ocupied_columns):
    
    planes = np.arange(0,8)

    # attach the positions to the data
    positions = initial_X_position_DUT(hit_data)
    plane_7 = hit_data[hit_data.plane == 7]
    mask = ak.num(plane_7) > 0
    events_starting_at_7 = hit_data[mask]
    hit_data_positions = ak.zip({ "hits":events_starting_at_7, "positions":positions},depth_limit=1)
    
    # get the most ocupied columns
    top_columns = columns_with_max_hits(hit_data, number_of_highest_ocupied_columns)
    top_columns = np.sort(top_columns)
    print(top_columns)

    # array to store all the data of planes per column
    total_avg_amount_planes = []

    # create to plot figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for column in top_columns:
        print(column)
        # creat an array to store the amount of hits in each plane
        hits_amount_plane_array = []

        # get the data of events initiating in the wanted column
        hit_data_column = hit_data_positions[(hit_data_positions.positions >= column) & (hit_data_positions.positions < column + 1)]


        # find the amount of hits in a plane for the column and add to the array
        for plane in planes:
            plane = 7-plane #adjust index so the first plane is 0
            hit_data_column_plane = hit_data_column.hits[hit_data_column.hits.plane == plane] # the events initiatin at the wanted column and a specific plane 
            hit_data_column_plane = hit_data_column_plane[ak.num(hit_data_column_plane) > 0] # clean from empty entries
            num_of_events_column_plane = len(hit_data_column_plane)
            num_of_hits_column_plane = len(ak.flatten(hit_data_column_plane))
            avg_hits_per_event_column_plane = num_of_hits_column_plane / num_of_events_column_plane
            hits_amount_plane_array.append(avg_hits_per_event_column_plane)
        total_avg_amount_planes.append(hits_amount_plane_array)
        print(hits_amount_plane_array)

        # plot avg amount of hits per plane
        ax1.plot(planes, hits_amount_plane_array, label=f"X Position: {column} Column", marker=".")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.set_xlabel("Plane")
        ax1.set_ylabel("Average Amount of Hits")
        ax1.set_title("Amount of Hits in a in Each Plane")
        ax1.legend()



    
    # plot avg amount of hits per position
    total_avg_amount_columns = np.transpose(np.array(total_avg_amount_planes))
    for plane in planes:
        avg_ampunt_plane_column = total_avg_amount_columns[plane]
        ax2.plot(top_columns, avg_ampunt_plane_column, label = f"Plane: {plane}", marker="D")
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Averag Amount of Hits')
        ax2.set_title('Amount of Hits in Each Column')
        ax2.grid(True)
        ax2.legend()

    fig.suptitle("Average Amount of Hits in a Shower for Different Plane, for Events Initiating in Different Positions", fontsize=14)
    plt.show()































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

        # plot avg amount of hits per plane
        ax1.plot(planes, energy_plane_array, label=f"X Position: {column} Column", marker=".")
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.set_xlabel("Plane")
        ax1.set_ylabel("Average Shower Energy")
        ax1.set_title("Energy Distribution in the Sensor for Differrnt Columns")
        ax1.legend()

    
    # plot avg amount of hits per position
    total_avg_energy_columns = np.transpose(np.array(total_avg_energy_planes))
    for plane in planes:
        avg_energy_plane_column = total_avg_energy_columns[plane]
        ax2.plot(top_columns, avg_energy_plane_column, label = f"Plane: {plane}", marker="D")
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



























# Histogram of ENERGies in all showers starting at a certain position
  
def Histo_shower_energy_for_X_position(hit_data, number_of_highest_ocupied_columns, single_pad_only = "false"):
    
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
        ax1.hist(energies_in_column, bins=200, histtype='step', label= f'Column: {column}, Peak = {peak_center:.2f}')
    
    # ax1.set_xticks(np.arange(0, 20))
    ax1.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_xlabel('X Position at Shower Initiation [Pad Column]')
    ax1.set_ylabel('Counts')
    ax1.set_title('Average Shower Energy vs Initial Location')

    # show the amounnt of hits in each plane on a bar chart
    bins = np.arange(0, 21, 1) 
    ax2.hist(ak.round(x_avg), bins=bins, edgecolor='black', rwidth=0.8)
    ax2.set_xticks(np.arange(0, 20) + 0.5)  # shift by 0.5
    ax2.set_xticklabels(np.arange(0, 20)) 
    ax2.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
    ax2.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
    ax2.set_xlabel('X Position [Pad Column]')
    ax2.set_ylabel('amount of hits')
    ax2.set_title('Amount of Events initiating in Each Column of the Sensor')
    
    plt.show()




