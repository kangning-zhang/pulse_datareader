
from pulsepytools.pulsesampler.anatomical import Anatomical

# !! Edit scan_id and destination here !!
scan_id = '2018-07-09T150052'
destination = "/local/ball4916/bdata/pulse2/Labeled_clips_v1.4_Richard"

def main():
    if destination is None:
        print("Please enter the destination of the annotations in"
              "run_anatomical_annotations.py\nExiting.")
        return
    ana = Anatomical(destination, scan_id=scan_id, mode='start')
    ana.annotate()

if __name__ == '__main__':
    main()
