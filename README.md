# Fast and Reliable Object Tracking Annotation Tool
I couldn't found any handy tool for object tracking annotation out there so I've written this tool.
This is only 1-day-written project so use it for youw own risk.

## Installation
It works with any version of OpenCV. Another dependency is [sortedcontainers](http://www.grantjenks.com/docs/sortedcontainers/introduction.html#installation). You can easily install all packages by `pip`.
```
pip install -r requirements.txt
```

## Usage
You need to extract frames from video file first.
```
python extract.py --source VIDEO_FILE [--out-dir OUT_DIR]
```
The extracted frames should be appeared in `extracted_frames` folder by default.

Then just run the main script and do annotate the video.
```
python main.py --source extracted_frames [--result-file RESULT_FILE]
```

## How it works
To reduce the annotation time, we will slice the whole video into segments in which the object moves nearly straight.
So we can assign two anchors each segment. One at the beginning and the other at the end. Every single bounding box of the object in that
segment would be interpolated linearly.
```
sx, sy, sw, sh -> the beginning bbox.
ex, ey, ew, eh -> the end bbox

-->
xi = sx + (ex - sx) * fi / seg_len
yi = sy + (ey - sy) * fi / seg_len
wi = sw + (ew - sw) * fi / seg_len
hi = sw + (eh - sh) * fi / seg_len
```
That's it.

## How to use it
The keyboard shortcuts are listed as beflow:
- `d`: Move to next frame.
- `a`: Move to previous frame.
- `w`: Increase track id.
- `s`: Decrease track id.
- `f`: Save results.
- `e`: Enter edit mode.
- `c`: Cancel edit mode.
- `t`: Interpolate bounding boxes.
- `r`: Remove next nearest anchor.
- `j`: Fast move forward.
- `k`: Fast move backward.
- `h`: Toggle all annotations.

The annotatoin process can be like this.
- 1. Enter edit mode.
- 2. Use key `w` and `s` to chose `track_id`. Use key `a` and `d` to move through frame space.
- 3. Select bounding box (anchor).
- 4. Move to next frame(s) then create anchors as many as you'd like.
- 5. When you finished the anchor selection step, you can interpolate all bounding boxes in the middle of segments by using key `t`.
- 6. Save result by key `f`.

## Contact
Feel free to leave issues. Enjoy!