# CAP_tools - Python Package for Identification of fMRI Co-activation Patterns (CAPs)
Uses consensus k-means + PAC score for optimal k identification to identify fMRI CAPs from ptseries data.


# Example Usage
```
python CAP_tools/CAP_tools.py -c subject_*ciftis.ptseries.nii -o outdir/file_prefix -d subjects*dtseries.nii
```
or using `.txt` files with each line being a path to file:

```
python ../CAP_tools.py -c ptseries_cifti_paths_file.txt \
                       -o output_files/test
                       -d dtseries_cifti_paths_file.txt
```


# ISC Filtering:
![Alt text](assets/isc_plot_example.png?raw=true "ISC Plot")

# Example CAP States - DSCALARS
![Alt text](assets/dCAP_plot_example.png?raw=true "ISC Plot")



<br><br>
# Sources:
- #TODO


# TODO:
[] refactor __init__.py + any needed package structure
