"# F-Cal-TB-Project" 

First step is introduction to UPROOT which is a library to read and process files in "ROOT" language by CERN.
We will be processing the data also using the "awkward" library which is helpful for many math actions such as dealing with non linear arrays (i.e arrays that cannot be written in the form of an NxM matrix)


ROOT FILE:
for each run we have a ROOT file with the ttree. In it we have all the events categorized and the parameters for them.

Each event is marked with a TLU number (this is like an index)

For each TLU(event) we have corresponding arrays - showing which pad(pixle) was activated, on which plane, and with what amplitud. 
Hence for every event we have 1 TLU number and an array of all the hits.
