cifti_paths=input_files/cifti_paths_file.txt
dtseries_paths=input_files/dtseries.txt

# python CAP_analysis.py -c $cifti_paths -o output_files/test \
# 					   -d $dtseries_paths -i .15 -k 3

# python CAP_analysis.py -c $cifti_paths -o output_files/test -i 0.25 -d $dtseries_paths

python ../CAP_analysis.py -c $(head -n 3 $cifti_paths) -o output_files/test -d $(head -n 3 $dtseries_paths)
