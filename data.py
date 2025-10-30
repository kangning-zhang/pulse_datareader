
from collections import OrderedDict
import json
from pathlib import Path
import os
import re

import yaml
import cv2

from .retrieve import retrieve_annotations, retrieve_frames
from ..utilities.image import get_sub_bbox
from ..utilities.median import geometric_median
# from utilities.io import read_config


def in_bbox(coords, bbox):
    for i_xy in [0, 1]:
        if coords[i_xy] < bbox[i_xy] or coords[i_xy] >= bbox[i_xy + 2]:
            break
    else:
        return True
    return False


def read_settings(file):
    """Read some fields of the settings.xml file"""
    patterns = {'scan_type': r'<ScanType>(\w+)</ScanType>',
                'patient_id': r'<PatientID>(\d+)</PatientID>',
                'sonographer_id': r'<SonographerID>(\d+)</SonographerID>',
                'GestationalAge': r'<GestationalAge>(.+)</GestationalAge>'}
    settings = {key: None for key in patterns.keys()}
    with open(file, 'r') as f:
        for line in f:
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    settings[key] = match.groups()[0]
    return settings


def register_scans(gaze_required=True, scan_type=('Anomaly',),
                   sonographer_ids=None, exclude_patients=None,
                   exclude_scans=None, required_annotations=None,
                   exclude_sonographers=None):
    """Create a register of the desired scans"""
    if required_annotations is None:
        required_annotations = ['context', 'xl_mode', 'freeze', 'probe']

    pulse_data_dir = Path(os.environ['PULSE_DATA_DIR'])
    pulse_data_dir.expanduser().resolve()
    processing_dir = pulse_data_dir / 'processing'
    raw_dir = pulse_data_dir / 'raw'

    register = []
    weeks = []
    for date_dir in processing_dir.glob('20*-*-*'):
        if not date_dir.is_dir():
            continue
        for scan_dir in date_dir.glob('20*-*-*T*'):
            if not scan_dir.is_dir():
                continue
            if not (scan_dir / 'gaze.dat').is_file() and gaze_required:
                continue
            if not (scan_dir / 'Frames').is_dir():
                continue
            annotations_dir = scan_dir / 'Annotations'
            if required_annotations and not annotations_dir.is_dir():
                continue
            if not all((annotations_dir / (file + '.csv')).is_file()
                       for file in required_annotations):
                continue
            scan_id = scan_dir.stem
            if exclude_scans is not None and scan_id in exclude_scans:
                continue
            # settings_file = scan_dir / (scan_id + '_settings.xml')
            settings_file = raw_dir / date_dir.stem /\
                (scan_id + '_settings.xml')
            if not settings_file.is_file():
                continue
            settings = read_settings(settings_file)
            if settings['scan_type'] is None:
                manual_scan_type_file = scan_dir / 'manual_scan_type.dat'
                if manual_scan_type_file.exists():
                    with open(manual_scan_type_file) as f:
                        settings['scan_type'] = f.readline().strip('\n')
            if settings['scan_type'] not in scan_type:
                continue
            if sonographer_ids is not None and\
                    settings['sonographer_id'] not in sonographer_ids:
                continue
            if exclude_sonographers is not None and\
                    settings['sonographer_id'] in exclude_sonographers:
                continue
            if exclude_patients is not None and\
                    settings['patient_id'] in exclude_patients:
                continue
            register.append(scan_id)

            if 'GestationalAge' in settings and settings['GestationalAge']\
                is not None:
                weeks.append(
                    int(settings['GestationalAge'].split('+')[0]))

    print(f"Gestational weeks: min {min(weeks)}, max, {min(weeks)}, "
          f"mean {sum(weeks) / len(weeks)}")
    return sorted(register)


def select_scans(config_path=None, force=False, data_suffix=None,
                 config_suffix='annotated', to_save=True):

    local_data_dir = Path(os.environ['LOCAL_DATA_DIR'])
    local_data_dir = local_data_dir.expanduser().resolve()

    if data_suffix is None:
        regsiter_file = local_data_dir / 'selection.json'
    else:
        regsiter_file = local_data_dir / f'selection_{data_suffix}.json'
    if regsiter_file.exists() and not force:
        with open(regsiter_file, 'r') as f:
            return json.load(f)

    if config_path is not None:
        config_dir = Path(config_path)
    else:
        config_dir = Path(__file__).parent
    config_dir = Path(config_dir).expanduser().resolve()
    if config_suffix is None:
        config_file = config_dir / "selection.yml"
    else:
        config_file = config_dir / f"selection_{config_suffix}.yml"
    with open(config_dir / config_file, 'r') as stream_:
        config = yaml.load(stream_)

    register = register_scans(**config)
    if to_save:
        with open(regsiter_file, 'w') as f:
            json.dump(register, f, indent=4)

    return sorted(register)


class Data:

    def __init__(self, scan_id, config_path=None, save_local=True):
        self.scan_id = scan_id
        self.config_path = config_path
        self.save_local = save_local
        self.config_dir = None
        self.config = None

        self.read_data_config()
        self._local_data_dir = None
        self._local_dir = None
        self._pulse_data_dir = None
        self._raw_dir = None
        self._dir = None
        self._frame_dir = None
        self._label_dir = None
        self._gaze_file = None

    def reset_data_directories(self):
        self._local_data_dir = None
        self._local_dir = None
        self._pulse_data_dir = None
        self._raw_dir = None
        self._dir = None
        self._frame_dir = None
        self._label_dir = None
        self._gaze_file = None

    def read_data_config(self):
        if self.config_path is not None:
            self.config_dir = Path(self.config_path)
        else:
            self.config_dir = Path(__file__).parent
        self.config_dir = Path(self.config_dir).expanduser().resolve()
        with open(self.config_dir / "data.yml", 'r') as stream_:
            self.config = yaml.load(stream_)

    @property
    def local_data_dir(self):
        if self._local_data_dir is None:
            self._local_data_dir = Path(os.environ['LOCAL_DATA_DIR'])
            self._local_data_dir = self.local_data_dir.expanduser().resolve()
        return self._local_data_dir

    @property
    def local_dir(self):
        if self._local_dir is None:
            self._local_dir = self.local_data_dir / self.scan_id
            self.mkdirs()
        return self._local_dir

    def mkdirs(self):
        self.local_dir.mkdir(exist_ok=True)

    @property
    def frame_dir(self):
        if self._frame_dir is None:
            self._frame_dir = self.dir / 'Frames'
        return self._frame_dir

    @property
    def label_dir(self):
        if self._label_dir is None:
            self._label_dir = self.dir / 'Annotations'
        return self._label_dir

    @property
    def gaze_file(self):
        if self._gaze_file is None:
            self._gaze_file = self.dir / self.config['gaze_file']
        return self._gaze_file

    @property
    def pulse_data_dir(self):
        if self._pulse_data_dir is None:
            self._pulse_data_dir = Path(os.environ['PULSE_DATA_DIR'])
            self._pulse_data_dir.expanduser().resolve()
        return self._pulse_data_dir

    @property
    def dir(self):
        if self._dir is None:
            self._dir = self.pulse_data_dir / 'processing' / self.scan_id[:10] /\
                self.scan_id
        return self._dir

    @property
    def raw_dir(self):
        if self._raw_dir is None:
            self._raw_dir = self.pulse_data_dir / 'raw' / self.scan_id[:10]
        return self._raw_dir

    @property
    def clocks_file(self):
        return self.raw_dir / (self.scan_id + '_clocks.dat')


class Scan(Data):

    def __init__(self, scan_id, config_path=None, save_frames=True,
                 check_jpg=False, **kwargs):
        super().__init__(scan_id, config_path=config_path, **kwargs)
        self.save_frames = save_frames and self.save_local
        self.check_jpg = check_jpg
        self.config = None
        self._res = None
        self._timestamps = None
        self.crop_res = None
        self.text_sub_bbox = None
        self._frames_xl = None

        self.read_config()

        if self.save_frames:
            self.scaled_frame_dir.mkdir(exist_ok=True)
        if self.save_local:
            _ = self.frames_xl

    def read_config(self):
        with open(self.config_dir / "scan.yml", 'r') as stream:
            self.config = yaml.load(stream)
        crop = self.config['bboxes']['image']
        self.crop_res = crop[2] - crop[0], crop[3] - crop[1]
        self.text_sub_bbox = get_sub_bbox(
            self.config['bboxes']['image'],
            self.config['bboxes']['text'])

    def frame_iterator(self):
        return range(1, len(self.timestamps) + 1)

    @property
    def res(self):
        if self._res is None:
            with open(self.frame_dir / 'height.dat', 'r') as f:
                height = int(f.read())
            with open(self.frame_dir / 'width.dat', 'r') as f:
                width = int(f.read())
            self._res = (width, height)
        return self._res

    @property
    def timestamps(self):
        if self._timestamps is None:
            timestamp_file = self.frame_dir / 'frames_timestamps.dat'
            if not timestamp_file.exists():
                raise FileNotFoundError('Timestamp file not found at\n'
                                        + str(timestamp_file))
            self._timestamps = []
            with open(timestamp_file, 'r') as f:
                for line in f:
                    self._timestamps.append(int(line))
        return self._timestamps

    def retrieve_frames_xl(self):
        frames_xl = retrieve_annotations(
            self.label_dir, 'xl_mode', column=1, include='True')[0]
        self.frames_xl = set(frames_xl)

    def get_scaled_res(self, sf):
        return tuple(round(res * sf) for res in self.crop_res)

    def get_frame_file(self, frame_nr):
        return self.frame_dir / 'frame{:06d}.png'.format(frame_nr)

    def get_frame(self, frame_nr):
        frame_file = self.get_frame_file(frame_nr)
        frame = cv2.imread(
            str(frame_file))
        if frame is None:
            print('Cannot read frame\n{}'.format(frame_file))
        return frame

    def get_bbox(self, name, frame_nr):
        bbox = self.config['bboxes'][name][:]
        if frame_nr in self.frames_xl:
            xl_mode_shift = self.config['xl_mode_shift'][name]
            bbox[0] += xl_mode_shift
            bbox[2] += xl_mode_shift
        return bbox

    def get_proc_frame(self, frame_nr):
        frame = self.get_frame(frame_nr)
        if frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = self.get_bbox('text', frame_nr)
        frame[text[1]: text[3], text[0]: text[2]] = 0
        crop = self.get_bbox('image', frame_nr)
        return frame[crop[1]: crop[3], crop[0]: crop[2]]

    def get_scaled_frame(self, frame_nr, sf, rm_miniframe=False):
        scaled_frame_file = self.get_scaled_frame_file(frame_nr, sf)
        frame = None
        if self.save_frames and scaled_frame_file.exists():
            if self.check_jpg and scaled_frame_file.suffix == '.jpg':
                with open(scaled_frame_file, 'rb') as f:
                    check_chars = f.read()[-2:]
                    if check_chars != b'\xff\xd9':
                        print(str(scaled_frame_file))
            frame = cv2.imread(str(scaled_frame_file), cv2.IMREAD_GRAYSCALE)

        if frame is None:
            # print(f'Reading frame\n{str(scaled_frame_file)}')
            frame = self.get_proc_frame(frame_nr)
            if frame is None:
                return frame
            frame = cv2.resize(frame, None, fx=sf, fy=sf,
                           interpolation=cv2.INTER_AREA)
            if self.save_frames:
                self.write_frame(scaled_frame_file, frame)
                # frame = cv2.imread(str(scaled_frame_file), cv2.IMREAD_GRAYSCALE)

        if rm_miniframe:
            crop = self.get_bbox('image', frame_nr)
            for region in ('miniframe', 'watermark'):
                bbox = self.get_bbox(region, frame_nr)
                bbox = get_sub_bbox(crop, bbox, tozero=True)
                bbox = [int(sf * coord) for coord in bbox]
                frame[bbox[1]: bbox[3], bbox[0]: bbox[2]] = 0

        return frame

    @staticmethod
    def write_frame(file, frame):
        if frame is None:
            print('Input is None. Not saving anything.')
            return
        cv2.imwrite(str(file), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 100])

    def get_scaled_frame_file(self, frame_nr, sf):
        return self.scaled_frame_dir / 'frame{:06d}_{:d}x{:d}_{:d}x{:d}.jpg'\
            .format(frame_nr, *self.crop_res,
                    *[int(r * sf) for r in self.crop_res])

    @property
    def scaled_frame_dir(self):
        return self.local_dir / 'frames'

    @property
    def frames_xl(self):
        if self._frames_xl is None:
            if self.frames_xl_file.exists():
                self.read_frames_xl()
            else:
                self.retrieve_frames_xl()
                if self.save_local:
                    self.write_frames_xl()
        return self._frames_xl

    @frames_xl.setter
    def frames_xl(self, frames_xl):
        self._frames_xl = frames_xl

    @property
    def frames_xl_file(self):
        return self.local_dir / 'frames_xl.json'

    def write_frames_xl(self):
        with open(self.frames_xl_file, 'w') as f:
            json.dump(list(self.frames_xl), f)

    def read_frames_xl(self):
        with open(self.frames_xl_file, 'r') as f:
            frames_xl = json.load(f)
            # Legacy
            if isinstance(frames_xl, dict):
                frames_xl = [fnr for fnr, val in frames_xl.items() if val]
            self.frames_xl = set(frames_xl)
            # self.frames_xl = {int(key): val for key, val in frames_xl.items()}


class Selection:

    def __init__(self, scan=None, scan_id=None, config_path=None,
                 save_frames=True):
        if scan is None:
            scan = Scan(scan_id, config_path=config_path,
                        save_frames=save_frames)
        self.scan = scan
        self._frames = None
        self._segments = None

    def read_frames(self):
        self.frames = list(self.scan.frame_iterator())

    def make_segments(self):
        self._segments = [[self.frames[0], self.frames[-1]]]

    @property
    def segments(self):
        if self._segments is None:
            self.make_segments()
        return self._segments

    @property
    def frames(self):
        if self._frames is None:
            self.read_frames()
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = frames



class LiveScan(Selection):

    def __init__(self, scan=None, scan_id=None, config_path=None,
                 save_frames=True):
        super().__init__(scan, scan_id, config_path, save_frames)
        self.config = None
        self.read_config()

    def read_config(self):
        with open(self.scan.config_dir / "livescan.yml", 'r') as stream:
            self.config = yaml.load(stream)

    def read_frames(self):
        self.frames = retrieve_frames(
            self.scan.label_dir, self.config['labels'])

    def rm_frame(self, frame_idx):
        pass

    def make_segments(self):
        # Split frames into segments
        min_seg_len = self.config['min_seg_len']
        self._segments = []
        prev_f_nr = self.frames[0]
        this_seg = [prev_f_nr] * 2
        for f_nr in self.frames[1:]:
            diff = f_nr - prev_f_nr
            if diff > 1 or f_nr == self.frames[-1]:
                if this_seg[1] - this_seg[0] + 1 >= min_seg_len:
                    self.segments.append(list(this_seg))
                this_seg = [f_nr] * 2
            else:
                this_seg[1] = f_nr
            prev_f_nr = f_nr

    def write_segments(self):
        with open(self.segments_file, 'w') as f:
            json.dump(self.segments, f)

    def read_segments(self):
        with open(self.segments_file, 'r') as f:
            self._segments = json.load(f)

    @property
    def segments_file(self):
        return self.scan.local_dir / 'scan_segments.json'

    @property
    def segments(self):
        if self._segments is None:
            if self.segments_file.exists():
                self.read_segments()
            else:
                self.make_segments()
                self.write_segments()
        return self._segments


class Gaze:

    def __init__(self, selection=None, scan=None, save_local=True,
                 load_local=True):
        assert(bool(selection) ^ bool(scan))    # XOR
        if selection is None:
            selection = Selection(scan)
        self.selection = selection
        self.scan = selection.scan
        self.save_local = save_local
        self.load_local = load_local

        self.config = None
        self._gaze_data = None
        self._segments = None
        self._frames = None
        self._gaze_proc = None
        self._clock_interp = None

        self.read_config()

    def process_gaze(self):
        self._gaze_proc = []
        frame_period = self.config['frame_period']
        for frame_nr in self.frames[::frame_period]:
            gaze_point = self.compute_gaze(frame_nr)
            if not gaze_point:
                continue  # TODO: Shouldn't happen
            self._gaze_proc.append((frame_nr, gaze_point))

    def compute_gaze(self, frame_nr):
        # TODO: Check

        gaze_points = self.gaze_data[frame_nr]
        if not gaze_points:
            return None
        if len(gaze_points) == 1:
            return tuple(gaze_points[0][1])

        all_points = [point[1] for point in gaze_points]

        if self.config['processing'] == 'median':
            point = geometric_median(all_points, method='minimize')
        else:
            raise ValueError('Unknown processing {}'
                             .format(self.config['processing']))
        point = [int(round(val)) for val in point]
        return tuple(point)

    def write_gaze_proc(self):
        with open(self.proc_file, 'w') as f:
            json.dump(self.gaze_proc, f)

    def read_gaze_proc(self):
        with open(self.proc_file, 'r') as f:
            return json.load(f)

    def read_config(self):
        with open(self.scan.config_dir / "gaze.yml", 'r') as stream:
            self.config = yaml.load(stream)

    def write_gaze_data(self):
        with open(self.file, 'w') as f:
            # json.dump(self.gaze_data, f)
            string = json.dumps(self.gaze_data)
            string = string.replace(", \"", ",\n  \"")
            string = "{\n  " + string[1:-1] + "\n}"
            f.write(string)

    def read_gaze_data(self):
        with open(self.file, 'r') as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
            return {int(k): v for k, v in data.items()}

    def write_gaze_csv(self, n_decimals=4):
        with open(self.scan.local_dir / 'gaze.csv', 'w') as f:
            for frame_nr in sorted(self.gaze_data.keys()):
                for time, coords in self.gaze_data[frame_nr]:
                    f.write('{0:06d} {1} {2[0]:.{3}f} {2[1]:.{3}f}\n'
                            .format(time, frame_nr, coords, n_decimals))

    @property
    def clock_interp(self):
        if self._clock_interp is None:
            with open(self.scan.clocks_file, 'r') as f:
                lines = f.readlines()
            anchors =\
                [[int(w) * 10 if w_idx == 0 else int(w) // 100
                  for w_idx, w in enumerate(lines[l_idx].split(' ')[:2])]
                 for l_idx in (999, 1000)]
            self._clock_interp = [anchors[0][0], anchors[0][1],
                            (anchors[1][0] - anchors[0][0]) /
                            (anchors[1][1] - anchors[0][1])]
        return self._clock_interp

    def systime2dshowtime(self, timestamp):
        timestamp = int(round(
            self.clock_interp[0] +
            (timestamp * 10 - self.clock_interp[1]) * self.clock_interp[2]))
        return timestamp

    def read_raw_gaze(self):
        """
        Read gaze in format (x,y) = (right, down)
        """
        self.gaze_data = OrderedDict(
            [(frame_nr, []) for frame_nr in self.selection.frames])
        with open(self.scan.gaze_file, 'r') as fin:

            valid_eye_idx = [14, 29]
            coord_idx = [[12, 27], [13, 28]]
            frame_nr = 1
            for data_line in fin:
                words = data_line.split()
                if words[0][0] == '#':
                    continue
                valid_eye = [int(words[v_idx]) for v_idx in valid_eye_idx]
                if not any(valid_eye):
                    continue

                timestamp = int(round(float(words[0])))
                timestamp = self.systime2dshowtime(timestamp)
                if timestamp < self.scan.timestamps[frame_nr - 1]:
                    continue

                while timestamp > self.scan.timestamps[frame_nr]:
                    frame_nr += 1
                    if frame_nr == len(self.scan.timestamps):
                        # print(f'Frames exhausted, timestamp {timestamp}')
                        return

                if frame_nr not in self.selection.frames:
                    frame_nr += 1
                    continue

                coords = []
                valid = True
                for i_xy in [0, 1]:
                    rel_eye_list = [float(words[coord_idx[i_xy][lr]])
                                    for lr, v in enumerate(valid_eye) if v]
                    rel_eye_avg = sum(rel_eye_list) / len(rel_eye_list)
                    coord = self.scan.res[i_xy] * rel_eye_avg
                    coord = coord - self.scan.get_bbox('image', frame_nr)[i_xy]
                    if coord < 0 or coord >= self.scan.crop_res[i_xy]:
                        valid = False
                        break
                    coords.append(coord)

                if not valid or in_bbox(coords, self.scan.text_sub_bbox):
                    continue

                self.gaze_data[frame_nr].append((timestamp, coords))

        # print(f'Gaze file exhausted, timestamp {timestamp}')

    def temporally_process_gaze(self):
        pass
        # TODO: Detect saccades and fixations

    def make_gaze_segments(self):
        # Split frames into segments
        min_seg_len = self.config['min_seg_len']
        max_void_frames = self.config['max_void_frames']
        self.segments = []
        self._frames = []
        for seg in self.selection.segments:
            prev_f_nr = seg[0]
            this_seg = [prev_f_nr] * 2
            for f_nr in range(seg[0] + 1, seg[1] + 1):
                if not self.gaze_data[f_nr]:
                    continue
                diff = f_nr - prev_f_nr
                if diff > max_void_frames or f_nr == seg[1]:
                    if this_seg[1] - this_seg[0] + 1 >= min_seg_len:
                        self.segments.append(list(this_seg))
                        self._frames += list(
                            range(this_seg[0], this_seg[1] + 1))
                    this_seg = [f_nr] * 2
                else:
                    this_seg[1] = f_nr
                prev_f_nr = f_nr
        pass

    def write_segments(self):
        with open(self.segments_file, 'w') as f:
            json.dump(self.segments, f)
        with open(self.frames_file, 'w') as f:
            json.dump(self.frames, f)

    def read_segments(self):
        with open(self.segments_file, 'r') as f:
            self.segments = json.load(f)
        with open(self.frames_file, 'r') as f:
            self._frames = json.load(f)

    @property
    def base_file_name(self):
        if isinstance(self.selection, LiveScan):
            return 'gaze'
        elif isinstance(self.selection, Selection):
            return 'gaze_all'
        else:
            return f'gaze_{self.selection.__class__.__name__}'

    @property
    def file(self):
        return self.scan.local_dir / (self.base_file_name + '.json')

    @property
    def segments_file(self):
        return self.scan.local_dir / (self.base_file_name + '_segments.json')

    @property
    def frames_file(self):
        return self.scan.local_dir / (self.base_file_name + '_frames.json')

    @property
    def proc_file(self):
        if self.config['processing'] is None:
            return None
        return self.scan.local_dir /\
            (self.base_file_name +
             '_every{config[frame_period]}_{config[processing]}.json'
             .format(config=self.config))

    @property
    def gaze_data(self):
        if self._gaze_data is None:
            if self.file.exists() and self.load_local:
                self._gaze_data = self.read_gaze_data()
            else:
                self.read_raw_gaze()
                if self.save_local:
                    self.write_gaze_data()
        return self._gaze_data

    @gaze_data.setter
    def gaze_data(self, gaze_data):
        self._gaze_data = gaze_data

    @property
    def segments(self):
        if self._segments is None:
            if self.segments_file.exists() and self.frames_file.exists()\
                    and self.load_local:
                self.read_segments()
            else:
                self.make_gaze_segments()
                self.write_segments()
        return self._segments

    @segments.setter
    def segments(self, segments):
        self._segments = segments

    @property
    def gaze_proc(self):
        if self._gaze_proc is None:
            if self.proc_file is None:
                return None
            if self.proc_file.exists() and self.load_local:
                self._gaze_proc = self.read_gaze_proc()
            else:
                self.process_gaze()
                if self.save_local:
                    self.write_gaze_proc()
        return self._gaze_proc

    @gaze_proc.setter
    def gaze_proc(self, gaze_proc):
        self._gaze_proc = gaze_proc

    @property
    def frames(self):
        if self._frames is None:
            if self._segments is None:
                # segment getter sets
                _ = self.segments
            else:
                raise ValueError("Unexpected internal error")
        return self._frames
