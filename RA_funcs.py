# ROOT ANALYSIS FUNCTIONS

import awkward as ak
import numpy as np
import matplotlib.pylab as plt
import uproot
import seaborn
from scipy.signal import find_peaks

print("imports work")






# extracts arrays from ROOT file and zip them for every hit
def get_ROOT_data_zip(file_name, tlu = "false", time = "false", toa = "false" ):

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
    print(f"{file_name} finished")

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
def single_event_evolution_amp(hit_data, TLU_number, cmap="berlin"):

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
    plt.legend()
    



    # plt.bar(first_occupied_plane_list, percentage_of_events_list, color='pink')


    

