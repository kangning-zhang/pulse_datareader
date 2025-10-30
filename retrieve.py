
from collections import Iterable
import csv
from pathlib import Path


# nan = config.nan
# not_applicable = config.not_applicable
csv_config = {'delimiter': ' ', 'quotechar': '|', 'quoting': csv.QUOTE_MINIMAL}
nan = 'NaN'  # NaN value in csv
not_applicable = 'NA'  # Filler value in csv
# skipped = '_skipped'


def to_path(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError('{} does not exist'.format(str(directory)))
    return directory


def retrieve_annotations(label_dir, label, column=None, include=None,
                         exclude=None, include_na=False, include_nans=False,
                         sniff=False, data_type=None, frame_set=None):

    label_dir = to_path(label_dir)
    if column is not None:
        assert(column >= 1)
    for var in (include, exclude):
        assert(var is None or isinstance(var, Iterable))

    # max_frame = None
    if frame_set is not None:
        if not isinstance(frame_set, set):
            frame_set = set(frame_set)
        # max_frame = max(frame_set)

    annotations = []
    frame_nrs = []
    file = label_dir / (label + '.csv')
    if not file.is_file():
        return None, None
    with open(file, 'r', newline='') as csv_file:
        if sniff:
            dialect = csv.Sniffer().sniff(csv_file.read(1024))
            csv_file.seek(0)
            reader = csv.reader(csv_file, dialect)
        else:
            reader = csv.reader(csv_file, **csv_config)

        for line in reader:
            frame_nr = int(line.pop(0))
            if frame_set is not None:
                if frame_nr not in frame_set:
                    continue
                # if frame_nr > max_frame:
                #     break
            n_words = len(line)

            if n_words == 1 and line[0] == not_applicable:
                if include_na:
                    annotations.append(not_applicable)
                    frame_nrs.append(frame_nr)
                continue

            if column is None:
                if all(word == nan for word in line) and not include_nans:
                    continue
                if include is not None and line[0] not in include:
                    continue
                if exclude is not None and line[0] in exclude:
                    continue

                annotations.append(line)
                frame_nrs.append(frame_nr)
                continue

            # if n_words < column:
            #     raise ValueError(
            #         'Fewer columns than expected:\n{}'.format(line))

            if n_words >= column:
                word = line[column - 1]
            else:
                word = nan
            if not include_nans and word == nan:
                continue
            annotation = data_type(word) if data_type is not None\
                else word

            if include is not None and annotation not in include:
                continue
            if exclude is not None and annotation in exclude:
                continue

            annotations.append(annotation)
            frame_nrs.append(frame_nr)

    # Sort the annotations via the frame numbers
    annotations = [
        x for _, x in sorted(zip(frame_nrs, annotations),
                             key=lambda pair: pair[0])]
    frame_nrs = sorted(frame_nrs)
    return frame_nrs, annotations


def retrieve_frames(label_dir, labels_values, columns=None):
    """
    Retrieve frames for given label values

    Args:
        label_dir: Directory containing the annotations
        labels_values: dictionary of {label: values, ...}
        columns: Column to read for each label. Defaults to ones.

    Returns:
        List of matching frames
    """

    if columns is None:
        columns = [1] * len(labels_values)

    label_dir = to_path(label_dir)

    frames = []
    for label, column in zip(labels_values.keys(), columns):
        frames.append(retrieve_annotations(
            label_dir, label, column=column, include=labels_values[label],
            include_na=False, include_nans=False)[0])

    # return the sorted intersection
    return sorted(list(set.intersection(*map(set, frames))))
