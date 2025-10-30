"""
pulsedata example

This script provides an example of how to use the pulsepytools.pulsedata module.
The important bit is the
scan.get_scaled_frame(frame_nr, scaling_factor)
method which allows fast repetitive frame reading by storing processed local
copies.

Before running this, you have to set the environment variables
PULSE_DATA_DIR and LOCAL_DATA_DIR, e.g. by executing
    export PULSE_DATA_DIR=/netshares/ibme_biomedia/Projects_1/pulse2
    export LOCAL_DATA_DIR=<some-local-directory>
in your shell.

The tools will read data from the PULSE_DATA_DIR and automatically save
processed versions in your LOCAL_DATA_DIR to speed up subsequent reads.

Overview of pulsepytools.pulsedata.data:
    select_scans:  Get a list of scans as specified in one of the selection*.yml
        config files. This calls register_scans on the first run.
    Data:  Base class that handles the pulse share files and local files.
    Scan:  Subclass of Base that handles video frames. It can be used to read
        raw frames or pre-processed frames. When reading a pre-processed frame,
        a copy is stored locally as a .jpg file to speed up subsequent reads.
        An example is provided below. This is particularly useful for reading
        data for training a neural network.
    Selection:  Base class for handling subsets of frames of scans.
    LiveScan:  Subclass of Selection. Extracts segments of uninterrupted
        live-scanning from a scan.
    Gaze: Handles gaze data for a given selection.

Moreover, pulsepytools.pulsedata contains functions to retrieve technical
annotations (retrieve.py) and to visualize a scan selection (visualize.py)

TBD:
    * Add class that handles probe motion data
"""

import time
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pulsepytools.pulsedata.data import Scan

# ====== Create the a Scan class object ======
# The scan_id uniquely identifies each scan by it's data and time
scan_id = '2018-06-04T091744'
# If this is the first time that a scan instance is created for this scan, then
# this might take a moment since the xl_mode annotation is read from the pulse
# share and processed since it is needed for subsequent cropping of the frames.
scan = Scan(scan_id)

# ====== Read a raw video frame (in BGR) ======
frame_nr = 10000
example_frame = scan.get_frame(frame_nr)

# ====== Read a pre-processed greyscale frame and time the read ======
# Scaling factor fow down-sampling the original HD video frame. A factor of
# 4/7 is chosen since it downsamples the cropped frame to 576x448 pixels,
# which is two times 288x224, which in turn nicely factors into 9*2^5x7*2^5.
scaling_factor = 4. / 7
# Optionally remove the zoom-preview region from the frame
rm_miniframe = False
# Make sure that the local_file doesn't exist
local_frame_file = scan.get_scaled_frame_file(frame_nr, scaling_factor)
if local_frame_file.is_file():
    local_frame_file.unlink()
# Read the frame
start_time = time.time()
example_frame_preproc = scan.get_scaled_frame(frame_nr, scaling_factor)
elapsed_time = time.time() - start_time
print(f"Elapsed time at first read: {elapsed_time:.2f}s")

# ====== Plot both frames ======
f = plt.figure()
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
ax0, ax1 = (plt.subplot(gs[idx]) for idx in range(2))
ax0.imshow(example_frame[:,:,::-1])
ax1.imshow(example_frame_preproc, cmap='gray')
plt.show()

# ====== Reading the same pre-processed frame again ======
# Now look into your LOCAL_DATA_DIR and check that a local copy of the
# pre-processed frame has been created. From now on, when we read that frame, it
# will be read from LOCAL_DATA_DIR, which should be faster. Let's try it:
start_time = time.time()
example_frame_preproc = scan.get_scaled_frame(frame_nr, scaling_factor)
elapsed_time = time.time() - start_time
print(f"Elapsed time at second read: {elapsed_time:.2f}s")
