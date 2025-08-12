# ROOT ANALYSIS FUNCTIONS

import awkward as ak
import numpy as np
import matplotlib.pylab as plt
import uproot
import seaborn

print("import work")






# extractS arrays from ROOT file and zip them for every hit
def get_ROOT_data_zip(file_name, tlu = "false", time = "false", toa = "false" ):

    # open the file
    infile = uproot.open(file_name)
    print("Folders:", infile.keys())
    print()

    # open the first "folder" hits
    hits = infile['Hits']
    print("Hits:")
    hits.show()

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

    return hit_data









# a colormap with the AMOUNT OF HITS in every channel(pad) of the chosen plane
def hits_amount_colormap_single_plane(hit_data, plane_number, cmap="berlin"):
      
    
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
    plt.title(f'Number of Hits in each channel, plane {plane_number}')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)




     





# a colormap with the total AMPLITUDE of hits in every channel(pad) of the chosen plane
def hits_amp_colormap_single_plane(hit_data, plane_number, cmap="berlin"):

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
    plt.title(f'Total Amplitude in Each Channel, Plane {plane_number}')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)











# show the average amplitude of each pad
def average_amp_colormap_single_plane(hit_data, plane_number, cmap="berlin"):

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
        # Code that might raise an exception
        if len(plane_pad_amp) == 0:
            avg_amp_l = 0
        else:
            avg_amp_l = plane_pad_total_amp / len(plane_pad_amp)
        counts_matrix[-1-q][r] = avg_amp_l
        
    # creat the colormap
    seaborn.heatmap(counts_matrix, cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Hit Counts'})
    plt.title(f'Number of Hits in each channel, plane {plane_number}')
    plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)















# Shows the shower evolution of a SINGLE EVENT in the sensor
def single_event_evolution(hit_data, TLU_number, cmap="berlin"):

    # get the path of the specific event
    TLU_event_data = hit_data[TLU_number]

    
    # count the amount of hits of each pad in every plane
    for plane in range(7,-1,-1):
        
        # get the channels hits on the wanted plane
        hits_plane = TLU_event_data[TLU_event_data.plane == plane].ch

        # create channel(pads) matrix
        counts_matrix = np.zeros((13, 20))
        
        # modify the pads matrix only if there are any hits on the plane
        if len(hits_plane) != 0:
            # count the amount of hits in each pad on the plane
            pads_1d, counts = np.unique(hits_plane, return_counts=True)

            # convert the 1d index of the pads into 2d coordinates
            pads_2d = divmod(pads_1d, 20)    

    
            # distribute the counts for each pad on a 12x20 matrix
            counts_matrix = np.zeros((13, 20))
            for i in range(len(pads_1d)):
                q = pads_2d[0][i]           # quotinent of the i'th pad (row from bottom)
                r = pads_2d[1][i]           # remainder of the i'th pad (column)
                counts_matrix[-1-q][r] = counts[i]
                
        # creat the colormap
        seaborn.heatmap(counts_matrix, cmap=cmap, linewidths=0.5, cbar_kws={'label': 'Hit Counts'})
        plt.title(f'Number of Hits in each channel, plane {7-plane}')
        plt.axvline(x=12, color='purple', linestyle='--', linewidth=1)
        plt.show()
