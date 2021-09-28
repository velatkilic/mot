#!/bin/bash

# A script generating all merged videos. All paths are hard-coded, and 
# are relative.
cd /mnt/d/JHU/Research/Machine_Learning_Characterization/ml_explosives/bin

python3 ./merge_pictures.py ../tests/reproduced/merged_non_overlaid \
    ../tests/playback_test300AlZr plot_frame jpg \
    ../tests/reproduced/non_overlaid frame_ png 9 301
python3 ./gen_video.py ../tests/reproduced/merged_non_overlaid merged_ png \
    ../tests/reproduced/merged_non_overlaid/merged_non_overlaid.avi 9

python3 ./merge_pictures.py ../tests/reproduced/merged_overlaid_line \
    ../tests/playback_test300AlZr plot_frame jpg \
    ../tests/reproduced/overlaid_line frame_ png 9 301
python3 ./gen_video.py ../tests/reproduced/merged_overlaid_line merged_ png \
    ../tests/reproduced/merged_overlaid_line/merged_overlaid_line.avi 9