# field-uniformity
### Track selection for LArTPC data using DBSCAN + PCA Fits
If running at NERSC, load corresponding environment as :  

`$ module load conda  `  
`$ conda activate /global/common/software/dune/efield2_rcmanduj/ `  
Then run as:  
`$ python3 track_selection_fsd.py <input_file> <output_file>.npy False `  
Where the last bool determines whether the detector has 2x2 (True) or FSD geometry (False).
<br>
One can then read out the outputs with NumPy:
<br>
`$ data = np.load(path, allow_pickle=True)`
<br>
`$ true_hits, reco_hits, PCA_dir, track_labels, a2a_flag , pca_variance, q, t_drift, io_group = data[:9]`
<br>
For non-position variables (1d), the values were tripled to ensure symmetric arrays. You can extract those single values as:
<br>
`$ q = q[:,0]`
