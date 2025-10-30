
from pulsepytools.pulsesampler import Sampler
from pulsepytools.pulsedata.data import select_scans

register = select_scans(data_suffix='samples', config_suffix='annotated')
# register = select_scans(force=True, data_suffix='samples', config_suffix='annotated')

# start = register.index('2018-10-01T094052')
start = 0

print('Extracting biometry planes')
for scan_id in register[start:]:
    print(scan_id)
    sampler = Sampler(None, scan_id)
    sampler.sample()

    sampler.write_frames(add_info=False, folder="biometry_planes")
    sampler.write_bg_frames(add_info=False, folder="biometry_planes_info")
