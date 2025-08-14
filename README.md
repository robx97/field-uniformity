# field-uniformity
### Track selection for LArTPC data using DBSCAN + PCA Fits
If running at NERSC, load corresponding environment as :  

`$ module load conda  `  
`$ conda activate /global/common/software/dune/efield2_rcmanduj/ `  
Then run as:  
`$ python3 track_selection_fsd.py <input_file> <output_file>.npy False `  
Where the last bool determines whether the detector has 2x2 (True) or FSD geometry (False).
<br>
One can then read out the outputs with h5py:
<br>
`$ with h5py.File("output.h5", "r") as fin:`
<br>
`$ print(list(fin.keys()))`
<br>
` ['tracks', 'hits']`
<br>
and then load variables:
`$ charges = fin['hits/charge']` 
<br>
`$ labels  = fin['hits/label']`
<br>
`$ track_dirs = fin['tracks/dir']`
<br>
