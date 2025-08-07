
# Move to the correct directory
cd /home/app

# Run the processing code
source ~/.credentials && /root/miniforge3/envs/ettoolbox/bin/python ETtoolbox_riogrande.py

# Run the post processing code
source ~/.credentials && /root/miniforge3/envs/ettoolbox/bin/python postprocess.py