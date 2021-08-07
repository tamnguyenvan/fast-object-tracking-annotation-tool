import os
import glob
import argparse
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.function_base import delete
from sortedcontainers import SortedDict


@dataclass
class KeyEvent:
    QUIT = ord('q')
    NEXT = ord('d')
    FAST_NEXT = ord('k')
    PREV = ord('a')
    FAST_PREV = ord('j')
    UP = ord('w')
    DOWN = ord('s')
    REWIND = ord('0')
    JUMP_END = ord('9')
    REMOVE = ord('r')
    UNDO = ord('u')
    SAVE = ord('f')
    TOGGLE_EDIT = ord('e')
    TRACK = ord('t')
    HIDE_ANCHORS = ord('h')


@dataclass
class ViewSettings:
    RENDER_TIME: int = 50
    WINDOW_TITLE: str = 'MOT'
    TITLE_COLOR = (0, 255, 0)
    TITLE_THICKNESS = 2


@dataclass
class ViewState:
    QUIT: str = 'stopped'
    INITIAL: str = 'initial'
    VIEW: str = 'view'
    TRACK: str = 'tracking'
    EDIT: str = 'edit'


class Saver:
    def __init__(self, save_path):
        self.save_path = save_path
        self.frame_dict = SortedDict()
        self.track_dict = SortedDict()
        self.track_anchors = SortedDict()
        self.undo_stack = []

    def setup(self):
        """
        """
        # Setup metadata file
        filename = Path(self.save_path).stem
        self.metadata_file = filename + '_metadata.pkl'
        self.frame_dict, self.track_dict = self.load()
        self.track_anchors = self.load_metadata()
    
    def load(self):
        """
        """
        frame_dict = {}
        track_dict = {}
        if os.path.isfile(self.save_path):
            with open(self.save_path) as f:
                for line in f:
                    line = line.strip().split(',')
                    row = list(map(float, line))
                    row = list(map(int, row[:6]))

                    frame_index, track_id = row[:2]
                    frame_index -= 1  # conver to 0-based index
                    xywh = row[2:]
                    if frame_index not in frame_dict:
                        frame_dict[frame_index] = SortedDict({track_id: xywh})
                    else:
                        frame_dict[frame_index][track_id] = xywh 
                    
                    if track_id not in track_dict:
                        track_dict[track_id] = SortedDict({frame_index: xywh})
                    else:
                        track_dict[track_id][frame_index] = xywh
        return frame_dict, track_dict
    
    def save(self):
        """
        """
        with open(self.save_path, 'wt') as f:
            for frame_index, frame_info in self.frame_dict.items():
                for track_id, xywh in frame_info.items():
                    row = [frame_index + 1, track_id, *xywh, -1, -1, -1, -1]
                    f.write(','.join(map(str, row)) + '\n')
        
        # Save anchors
        self.save_metadata()
        print(f'Saved results as {self.save_path}')
    
    def load_tracks(self, frame_index, track_id=None):
        """
        """
        bboxes = {}
        if isinstance(track_id, int):
            if frame_index in self.frame_dict and track_id in self.frame_dict[frame_index]:
                return {track_id: self.frame_dict[frame_index][track_id]}
            return bboxes
        
        if track_id is None and frame_index in self.frame_dict:
            bboxes = self.frame_dict[frame_index]
        return bboxes
    
    def get_bbox(self, frame_index, track_id):
        """
        """
        if frame_index in self.frame_dict and track_id in self.frame_dict[frame_index]:
            return self.frame_dict[frame_index][track_id]
    
    def add_bbox(self, frame_index, track_id, bbox):
        """
        """
        if frame_index in self.frame_dict:
            self.frame_dict[frame_index][track_id] = bbox
        else:
            self.frame_dict[frame_index] = SortedDict({track_id: bbox})
        
        if track_id in self.track_dict:
            self.track_dict[track_id][frame_index] = bbox
        else:
            self.track_dict[track_id] = SortedDict({frame_index: bbox})
    
    def delete_bbox(self, frame_index, track_id):
        """
        """
        if frame_index in self.frame_dict and track_id in self.frame_dict[frame_index]:
            del self.frame_dict[frame_index][track_id]
        
        if track_id in self.track_dict and frame_index in self.track_dict[track_id]:
            del self.track_dict[track_id][frame_index]
    
    def load_anchors(self, track_id=None):
        """
        """
        anchor_bboxes = SortedDict()
        if track_id in self.track_anchors:
            return self.track_anchors[track_id]
        return anchor_bboxes

    def _calc_inter_bbox(self, steps, start_bbox, end_bbox):
        """
        """
        inter_bboxes = []
        sx, sy, sw, sh = start_bbox
        ex, ey, ew, eh = end_bbox
        for i in range(steps):
            # Interpolate the bounding boxes in the middle
            ix = sx + int((ex - sx) * i / steps)
            iy = sy + int((ey - sy) * i / steps)
            iw = sw + int((ew - sw) * i / steps)
            ih = sh + int((eh - sh) * i / steps)
            inter_bboxes.append((ix, iy, iw, ih))
        return inter_bboxes
    
    def interpolate_bboxes(self, track_id):
        """
        """
        # Reset frame_dict
        new_frame_dict = SortedDict()
        for fi, frame_data in self.frame_dict.items():
            new_frame_dict[fi] = SortedDict()
            if track_id in frame_data:
                del frame_data[track_id]
            new_frame_dict[fi] = frame_data
        self.frame_dict = new_frame_dict
    
        anchor_bboxes = self.load_anchors(track_id)
        if len(anchor_bboxes) > 1:
            frame_indices = list(anchor_bboxes.keys())
            num_anchors = len(frame_indices)
            for i in range(num_anchors - 1):
                start = frame_indices[i]
                end = frame_indices[i+1]

                start_bbox = anchor_bboxes[start]
                end_bbox = anchor_bboxes[end]
                inter_bboxes = self._calc_inter_bbox(end - start + 1, start_bbox, end_bbox)
                for j, bbox in enumerate(inter_bboxes):
                    self.add_bbox(start + j, track_id, bbox)

    def add_anchor(self, frame_index, track_id, bbox):
        """
        """
        if track_id in self.track_anchors:
            self.track_anchors[track_id][frame_index] = bbox
        else:
            self.track_anchors[track_id] = SortedDict({frame_index: bbox})
    
    def delete_anchor(self, frame_index, track_id):
        """
        """
        # Delete the anchor
        if track_id in self.track_anchors:
            sorted_frame_indices = list(self.track_anchors[track_id].keys())
            start_bin = np.digitize(frame_index, sorted_frame_indices)
            start_bin = min(max(0, start_bin), len(sorted_frame_indices) - 1)
            delete_fi = sorted_frame_indices[start_bin]

            for fi in sorted_frame_indices:
                if fi == delete_fi:
                    # Append to undo_stack
                    self.undo_stack.append((track_id, fi, self.track_anchors[track_id][fi]))

                    del self.track_anchors[track_id][fi]
                    break
        
        # Interpolate again
        self.interpolate_bboxes(track_id)
    
    def undo_delete_anchor(self, track_id):
        """
        """
        if len(self.undo_stack):
            track_id, fi, bbox = self.undo_stack.pop()
            self.add_anchor(fi, track_id, bbox)

    def load_metadata(self):
        """
        """
        track_anchors = SortedDict()
        if os.path.isfile(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                return pkl.load(f)
        return track_anchors
    
    def save_metadata(self):
        """
        """
        with open(self.metadata_file, 'wb') as f:
            pkl.dump(self.track_anchors, f)


class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
    
    def start(self):
        """
        """
        self.model.setup()
        self.view.setup(self)
        self.view.start_main_loop()
    
    def handle_key_event(self, event):
        """
        """
        if event == KeyEvent.QUIT:
            print('Exit!')
            self.view.quit()
        elif event == KeyEvent.NEXT:
            self.view.next()
        elif event == KeyEvent.FAST_NEXT:
            self.view.next(step=10)
        elif event == KeyEvent.PREV:
            self.view.prev()
        elif event == KeyEvent.FAST_PREV:
            self.view.prev(step=10)
        elif event == KeyEvent.UP:
            self.view.increase_track_id()
        elif event == KeyEvent.DOWN:
            self.view.decrease_track_id()
        elif event == KeyEvent.REWIND:
            self.view.seek(0)
        elif event == KeyEvent.JUMP_END:
            self.view.seek(-1)
        elif event == KeyEvent.TOGGLE_EDIT:
            if self.view.state == ViewState.EDIT:
                self.view.state = ViewState.VIEW
            else:
                self.view.state = ViewState.EDIT
        elif event == KeyEvent.TRACK:
            self.model.interpolate_bboxes(self.view.track_id)
            self.view.state = ViewState.VIEW
        elif event == KeyEvent.HIDE_ANCHORS:
            self.view.is_hide_anchors = not self.view.is_hide_anchors
        elif event == KeyEvent.REMOVE:
            if self.view.state == ViewState.VIEW:
                self.model.delete_anchor(self.view.frame_index,
                                        self.view.track_id)
        elif event == KeyEvent.UNDO:
            if self.view.state == ViewState.VIEW:
                self.model.undo_delete_anchor(self.view.track_id)
        elif event == KeyEvent.SAVE:
            self.model.save()
    
    def add_bbox(self, frame_index, track_id, bbox):
        """
        """
        self.model.add_bbox(frame_index, track_id, bbox)
    
    def delete_bbox(self, frame_index, track_id):
        """
        """
        self.model.delete_bbox(frame_index, track_id)
    
    def add_anchor(self, frame_index, track_id, bbox):
        """
        """
        self.model.add_anchor(frame_index, track_id, bbox)
    
    def load_tracks(self, frame_index, track_id=None):
        """
        """
        return self.model.load_tracks(frame_index, track_id)
    
    def load_anchors(self, track_id=None):
        """
        """
        return self.model.load_anchors(track_id)
    
    def load_latest_track_id(self):
        """
        """
        latest_track_id = 1
        track_ids = list(self.model.track_dict.keys())
        if len(track_ids):
            return track_ids[-1]
        return latest_track_id


class View:
    def __init__(self, source):
        self.track_id = 1
        self.frame_index = 0
        self.source = source
        self.state = ViewState.INITIAL
        self.segments = {}
        self.frame = None
        self.is_hide_anchors = False

    def setup(self, controller):
        """Setup View initial state."""
        self.state = ViewState.VIEW
        self.controller = controller
        self.load_source()
        self.frame = self.load_frame()

        # Reassign track_id to the latest annotated id
        self.track_id = controller.load_latest_track_id()

        # Colors for tracks
        self.colors = [tuple(map(int, color)) for color in np.random.randint(120, 250, (1000, 3))]

    @property
    def num_frames(self):
        """Returns the number of frames."""
        total_frames = 0
        if isinstance(self.frame_paths, (tuple, list)):
            total_frames = len(self.frame_paths)
        return total_frames
    
    def _get_valid_index(self, frame_index):
        """Return valid index ranges in [0, num_frames-1]."""
        if 0 <= frame_index <= self.num_frames - 1:
            return frame_index
        elif frame_index == -1:
            return self.num_frames - 1
        elif frame_index < -1:
            return 0
        else:
            return self.num_frames - 1
    
    def load_frame(self):
        """Load the current frame."""
        frame_index = self._get_valid_index(self.frame_index)
        frame_path = self.frame_paths[frame_index]
        if os.path.isfile(frame_path):
            image = cv2.imread(self.frame_paths[frame_index])
            return image
    
    def load_source(self):
        """
        """
        assert os.path.isdir(self.source)
        self.frame_paths = sorted(glob.glob(os.path.join(self.source, '*.jpg')))
    
    def next(self, step=1):
        """
        """
        self.frame_index = self._get_valid_index(self.frame_index + step)
    
    def prev(self, step=1):
        """
        """
        self.frame_index = self._get_valid_index(max(0, self.frame_index - step))
    
    def seek(self, position=0):
        """
        """
        self.frame_index = self._get_valid_index(position)
    
    def increase_track_id(self):
        """"""
        self.track_id += 1
    
    def decrease_track_id(self):
        """"""
        self.track_id = max(1, self.track_id - 1)
    
    def quit(self):
        """
        """
        self.state = ViewState.QUIT
    
    def _draw_centroid(self, frame, centroid, color):
        """
        """
        cv2.circle(frame, centroid, 3, color, -1)
        return frame
    
    def _draw_line(self, frame, points, color):
        """
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        if points.ndim != 3:
            points = np.reshape(points, (-1, 1, 2))

        cv2.polylines(frame, [points], False, color, 2)
        return frame
    
    def _draw_bboxes(self, frame, bbox, color, track_id=None):
        """
        """
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track_id is not None:
            (tw, th), baseline = cv2.getTextSize(str(track_id), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            th += baseline + 2
            cv2.rectangle(frame, (x1, y1 - th), (x1 + tw, y1), color, -1)
            cv2.putText(frame, str(track_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, 1, 2)
        return frame
    
    def draw_anchor(self, frame):
        """
        """
        color = self.colors[self.track_id]
        anchor_bboxes = self.controller.load_anchors(self.track_id)
        centroids = []
        for _, bbox in anchor_bboxes.items():
            x, y, w, h = bbox
            centroid = int(x + w / 2), int(y + h / 2)
            centroids.append(centroid)
            frame = self._draw_bboxes(frame, bbox, color, self.track_id)
            frame = self._draw_centroid(frame, centroid, color)

        if len(centroids):
            points = np.reshape(centroids, (-1, 1, 2))
            frame = self._draw_line(frame, [points], color)
        return frame
    
    def _render_title(self, frame):
        """
        """
        text = 'Frame: {:06d} Track ID: {} Mode: {}'.format(
            self.frame_index + 1, self.track_id, self.state)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 2.,
                    ViewSettings.TITLE_COLOR, ViewSettings.TITLE_THICKNESS)
        return frame

    def render_frame(self, frame):
        """
        """
        self._render_title(frame)

        # Draw bounding boxes if available
        track_data = self.controller.load_tracks(self.frame_index)
        for track_id, bbox in track_data.items():
            frame = self._draw_bboxes(frame, bbox, self.colors[track_id], track_id)
        
        cv2.imshow(ViewSettings.WINDOW_TITLE, frame)
        key = cv2.waitKey(ViewSettings.RENDER_TIME)
        return key

    def start_main_loop(self):
        """
        """
        while True:
            # Grab a frame
            self.frame = self.load_frame()
            frame = self.frame

            # Draw anchor
            if not self.is_hide_anchors:
                frame = self.draw_anchor(frame)

            # Edit
            if self.state == ViewState.EDIT:
                self.state == ViewState.EDIT
                frame = self._render_title(frame)
                roi = cv2.selectROI(ViewSettings.WINDOW_TITLE, frame, showCrosshair=False)
                if sum(roi) > 0:
                    self.controller.add_anchor(self.frame_index, self.track_id, roi)
                self.state = ViewState.VIEW
            
            # Render frame
            key = self.render_frame(frame)

            # Handle key event
            self.controller.handle_key_event(key)
            if self.state == ViewState.QUIT:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Path to source video.')
    parser.add_argument('--result-file', type=str, default='', help='The output file.')
    args = parser.parse_args()

    if args.result_file:
        result_file = args.result_file
    else:
        result_file = Path(args.source).stem + '_result.txt'
    model = Saver(result_file)
    view = View(args.source)
    controller = Controller(model, view)
    controller.start()


if __name__ == '__main__':
    main()