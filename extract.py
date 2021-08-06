import os
import argparse
from pathlib import Path
import cv2
import tqdm


def extract(source, out_dir):
    """
    """
    filename = Path(source).stem
    frame_out_dir = os.path.join(out_dir, filename)
    os.makedirs(frame_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm.tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        save_path = os.path.join(frame_out_dir, '%06d.jpg' % i)
        cv2.imwrite(save_path, frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Path to source video.')
    parser.add_argument('--out-dir', type=str, default='extracted_frames',
                        help='Path to output file.')
    args = parser.parse_args()

    print(f'Extracting frames from {args.source}')
    extract(args.source, args.out_dir)
    print('Done!')


if __name__ == '__main__':
    main()