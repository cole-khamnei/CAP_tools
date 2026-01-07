#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"


cifti_paths=${SCRIPT_DIR}/input_files/cifti_paths_file.txt
dtseries_paths=${SCRIPT_DIR}/input_files/dtseries.txt

# python CAP_tools.py -c $cifti_paths -o output_files/test \
# 					   -d $dtseries_paths -i .15 -k 3

# python CAP_tools.py -c $cifti_paths -o output_files/test -i 0.25 -d $dtseries_paths

# python ${SCRIPT_DIR}/../CAP_tools.py -c $cifti_paths -o ${SCRIPT_DIR}/output_files/test \
# 					   --n-reps 3 --k-max 10 --overwrite

python ${SCRIPT_DIR}/../CAP_tools.py -c $(head -n 3 $cifti_paths) \
									 -o ${SCRIPT_DIR}/output_files/test \
					   				 -d $(head -n 3 $dtseries_paths) --n-reps 3 \
					   				 --k-max 10 --overwrite
