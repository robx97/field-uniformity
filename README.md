# field-uniformity
### Track selection for LArTPC data using DBSCAN + PCA Fits
If running at NERSC, load corresponding environment as :
> $ module load conda
> $ conda activate /global/common/software/dune/efield2_rcmanduj/
Then run as:
> $ python3 track_selection_fsd.py <input_file> <output_file>.npy False
Where the last bool determines whether the detector has 2x2 (True) or FSD geometry (False)
