
import awkward as ak
import uproot
import numpy as np




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
    #replace this with the path to your file
    path = f"TB_FIRE/TB_reco/TB_FIRE_{run_number}_raw_reco.root"
    dut = uproot.open(path)
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

    TLU_data, TLU = ak_groupby_for_scope(TLU,hit_data)
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
    trigID_data, trigID = ak_groupby_for_scope(ak.flatten(trigID), tele_data)
    
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

