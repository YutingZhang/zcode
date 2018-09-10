import os
import numpy as np
import cv2


class VideoSegWriter():

    def __init__(self, folder_path, seg_len=20, fps=None, ext="mp4", fourcc="mp4v", verbose=False):
        self._folder_path = folder_path
        self._seg_len = seg_len
        self._fps = 30 if fps is None else fps
        self._ext = ext
        self._fourcc = fourcc
        self._verbose = verbose

        if self._verbose:
            print("Start: ", self._folder_path)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self._frame_count = 0
        self._video_writer = None
        self._current_frame = 0
        self._seg_frame_count = 0

    def _release_current_video_writer(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            if self._verbose:
                print("$")
            self._seg_frame_count = 0

    def close(self):
        if self._current_frame is None:
            return
        self._release_current_video_writer()
        self._current_frame = None
        with open(os.path.join(self._folder_path, "info.txt"), "w") as f:
            f.write("frame_count:\t%d\n" % self._frame_count)
            f.write("seg_len:\t%d\n" % self._seg_len)
            f.write("fps:\t%d\n" % self._fps)
            f.write("ext:\t%s\n" % self._ext)
        if self._verbose:
            print("End: ", self._folder_path)

    def release(self):
        self.close()

    def write(self, frame):
        if self._video_writer is None:
            out_fn = "%d.%s" % (self._frame_count, self._ext)
            out_path = os.path.join(self._folder_path, out_fn)
            if os.path.exists(out_path):
                os.remove(out_path)
            if self._verbose:
                print("%s : " % out_fn, end="", flush=True)
            is_color = True
            if len(frame.shape) == 2:
                frame = np.expand_dims(frame, axis=-1)
            if frame.shape[2] == 1:
                is_color = False
            elif frame.shape[2] == 3:
                pass
            else:
                raise ValueError("unsupported number of channels")

            self._video_writer = cv2.VideoWriter(
                out_path,
                # apiPreference=cv2.CAP_ANY,
                apiPreference=cv2.CAP_FFMPEG,
                fourcc=cv2.VideoWriter.fourcc(*self._fourcc),
                fps=self._fps,
                frameSize=(frame.shape[1], frame.shape[0]),
                isColor=is_color
            )
            self._video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
            assert self._video_writer.isOpened(), "Cannot create the video segment"
        if self._verbose:
            print(".", end="", flush=True)
        self._video_writer.write(frame)
        self._seg_frame_count += 1
        self._frame_count += 1
        if self._seg_frame_count >= self._seg_len:
            self._release_current_video_writer()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoSegReader:

    def __init__(self, folder_path):
        self._folder_path = folder_path
        self._video_cap = None
        self._current_seg_index = None

        self._frame_count = None
        self._seg_len = None
        self._fps = None
        self._ext = None
        with open(os.path.join(folder_path, "info.txt"), "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, val = line.split()
                if key == "frame_count:":
                    self._frame_count = int(val)
                elif key == "seg_len:":
                    self._seg_len = int(val)
                elif key == "fps:":
                    self._fps = float(val)
                elif key == "ext:":
                    self._ext = val
                else:
                    raise ValueError("unrecognized key %s in info.txt" % key)

        assert self._frame_count is not None, "frame_count is not configured"
        assert self._seg_len is not None, "seg_len is not configured"
        assert self._fps is not None, "fps is not configured"
        assert self._ext is not None, "ext is not configured"

    @property
    def folder_path(self):
        return self._folder_path

    @property
    def fps(self):
        return self._fps

    @property
    def seg_len(self):
        return self._seg_len

    @property
    def frame_count(self):
        return self._frame_count

    def release_segment(self):
        if self._video_cap is not None:
            self._video_cap.release()
        self._current_seg_index = None
        self._video_cap = None

    def release(self):
        self.release_segment()

    def read_at(self, frame_id):

        assert frame_id < self.frame_count, "frame_id out of range: " + self._folder_path

        seg_index = (frame_id // self._seg_len) * self._seg_len
        seg_frame_id = frame_id - seg_index

        _cont = True
        while _cont:
            # frame_id is zero-based
            if seg_index != self._current_seg_index:
                self.release_segment()
                seg_fn = os.path.join(self._folder_path, str(seg_index) + "." + self._ext)
                self._video_cap = cv2.VideoCapture(seg_fn)
                assert self._video_cap.isOpened(), "failed to open: %s" % seg_fn
                self._current_seg_index = seg_index

            if int(self._video_cap.get(cv2.CAP_PROP_POS_FRAMES)) != seg_frame_id:
                self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, seg_frame_id)

            ret, frame = self._video_cap.read()
            if ret:
                _cont = False
            else:
                print('  WARNING: VideoSegReader cannot read frame try to reload the video: %s' %
                        os.path.join(self._folder_path, "%d.%s" % (seg_index, self._ext)) )
                self.release_segment()

        frame = np.flip(frame, axis=2)   # BGR --> RGB

        return frame

    def __del__(self):
        self.release_segment()


class RandomAccessVideoSegReader:

    def __init__(self):
        self._vid_fn = None
        self._vid_seg_reader = None

    def set_video(self, folder_path):
        if self._vid_fn != folder_path:
            self._vid_fn = folder_path
            self.release()
            self._vid_seg_reader = VideoSegReader(folder_path)

    def read_at(self, frame_id):
        assert isinstance(self._vid_seg_reader, VideoSegReader), "need to set_video first"
        return self._vid_seg_reader.read_at(frame_id)

    @property
    def video_filename(self):
        return self._vid_fn

    def release(self):
        self._vid_fn = None
        if self._vid_seg_reader is not None:
            self._vid_seg_reader.release_segment()
            self._vid_seg_reader = None

    @property
    def folder_path(self):
        return self._vid_seg_reader.folder_path

    @property
    def fps(self):
        return self._vid_seg_reader.fps

    @property
    def seg_len(self):
        return self._vid_seg_reader.seg_len

    @property
    def frame_count(self):
        return self._vid_seg_reader.frame_count

    def release_segment(self):
        return self._vid_seg_reader.release_segment()


def image_folder_to_video_seg(
        image_folder_path, video_seg_path, *args, **kwargs
):

    assert isinstance(image_folder_path, str), "image_folder_path must be a str"
    assert isinstance(video_seg_path, str), "video_seg_path must be a str"

    # list images
    all_filenames = os.listdir(image_folder_path)
    imgs = []
    for fn in all_filenames:
        bare_fn, ext = os.path.splitext(fn)
        if ext.lower() in (".png", '.jpg', '.jpeg', '.gif', '.bmp'):
            imgs.append((int(bare_fn), fn))
    assert imgs, "nothing in the folder: %s" % image_folder_path
    imgs = sorted(imgs, key=lambda t: t[0])
    img_filenames = [fn for _, fn in imgs]

    # write to video seg
    vsw = VideoSegWriter(
        folder_path=video_seg_path, *args, **kwargs,
    )
    for fn in img_filenames:
        img_full_fn = os.path.join(image_folder_path, fn)
        im = cv2.imread(img_full_fn)
        vsw.write(im)
    vsw.close()


def main():

    from glob import glob
    import random
    import time

    vid_reader = RandomAccessVideoSegReader()
    all_video_fn = glob("/home/yutingzh/data/vid_seg/Human3.6M/*/Videos/*")
    vid_reader.set_video(all_video_fn[1])
    _ = vid_reader.read_at(1)

    m = 5
    s = 1
    n = len(all_video_fn)
    total_frames = 1000000
    t = time.time()
    for j in range(total_frames):
        print("%d / %d" % (j, total_frames))
        i = random.randint(0, n-1)
        vid_reader.set_video(all_video_fn[i])
        r = random.randint(0, vid_reader.frame_count-m*s)
        for _ in range(m):
            _ = vid_reader.read_at(r)
            r += s
    elapsed = time.time() - t
    print("%g sec for %d frames = %g sec/frame" % (elapsed, total_frames*m, elapsed / total_frames / m))


if __name__ == "__main__":
    main()

