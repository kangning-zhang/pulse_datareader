
import cv2

class Visualise:

    def __init__(self, gaze, fps=30):
        self.gaze = gaze
        self.fps = fps

    def write_all_videos(self, every=1):
        # Generate videos
        for seg in self.gaze.selection.segments[::every]:
            self.write_gaze_video(seg, title='video')

    def write_all_gaze_videos(self, every=1):
        # Generate videos
        for seg in self.gaze.segments[::every]:
            self.write_gaze_video(seg)

    def write_gaze_video(self, segment, title='gaze_video', new_size=None,
                         is_color=True, codec="XVID", speed=1):

        video_dir = self.gaze.scan.local_dir / 'videos'
        video_dir.mkdir(exist_ok=True)
        video_file = str(video_dir / '{}_{:06d}-{:06d}.avi'.format(
            title, *segment))
        fourcc = cv2.VideoWriter_fourcc(*codec)

        vid = None
        for frame_nr in list(range(*segment)) + segment[1:]:

            frame = self.gaze.scan.get_proc_frame(frame_nr)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if frame_nr in self.gaze.gaze_data:

                for gaze_point in self.gaze.gaze_data[frame_nr]:
                    coords = gaze_point[1]
                    # coords = coords[::-1]
                    coords = [round(c) for c in coords]
                    frame = cv2.circle(
                        frame, tuple(coords), 6, (100, 255, 100), -1)

            if new_size is not None:
                frame = cv2.resize(frame, new_size)

            if vid is None:
                size = frame.shape[1], frame.shape[0]
                vid = cv2.VideoWriter(
                    video_file, fourcc, float(self.fps * speed), size, is_color)

            vid.write(frame)
        vid.release()




