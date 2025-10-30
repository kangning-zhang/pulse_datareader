
from pulsepytools.pulsesampler import Sampler
from pulsepytools.pulsedata.data import select_scans

register = select_scans(data_suffix='samples', config_suffix='annotated')
# register = select_scans(force=True, data_suffix='samples', config_suffix='annotated')
# register = select_scans(force=True, data_suffix='samples',
#                         config_suffix='annotated_withTwins')

# start = register.index('2018-10-23T105544')
start = 0
text_frames_only = False
exclude_text_frames = False

print('Starting manual annotation of saved frames')
for scan_id in register[start:]:
    print(scan_id)
    sampler = Sampler(None, scan_id)
    sampler.annotate_misc_frames(text_frames_only=text_frames_only,
                                 exclude_text_frames=exclude_text_frames)

    sampler.write_frames(add_info=False, folder="misc")
    # sampler.write_frames(add_info=True, folder="misc_info")
    sampler.write_bg_frames(add_info=False, folder="misc")
    # sampler.write_bg_frames(add_info=True, folder="misc_info")
