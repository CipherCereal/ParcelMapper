from pathlib import Path

import cv2

from drone.video import (
    get_video_info,
    read_sampled_frames,
)


VIDEO_PATH = Path(
    "data/input/drone_demo.mp4"
)

OUTPUT_FOLDER = Path(
    "data/output/sample_frames"
)

EVERY_N_FRAMES = 15

NUMBER_OF_TEST_FRAMES = 5


def main():

    print()
    print("ParcelMapper - Drone Video Test")
    print("--------------------------------")

    if not VIDEO_PATH.exists():

        print(
            f"ERROR: Video not found at "
            f"{VIDEO_PATH}"
        )

        print(
            "Put an MP4 video inside "
            "data/input/ and rename it "
            "drone_demo.mp4"
        )

        return

    print()
    print("Video found!")

    info = get_video_info(VIDEO_PATH)

    print()
    print("VIDEO INFORMATION")
    print("-----------------")

    print(
        f"Resolution: "
        f"{info['width']} x {info['height']}"
    )

    print(
        f"FPS: "
        f"{info['fps']:.2f}"
    )

    print(
        f"Total frames: "
        f"{info['total_frames']}"
    )

    print(
        f"Duration: "
        f"{info['duration_seconds']:.2f} seconds"
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"Reading every "
        f"{EVERY_N_FRAMES}th frame..."
    )

    saved_frames = 0

    for frame_number, frame in read_sampled_frames(
        VIDEO_PATH,
        every_n=EVERY_N_FRAMES
    ):

        height, width = frame.shape[:2]

        print(
            f"Frame {frame_number}: "
            f"{width} x {height}"
        )

        output_file = (
            OUTPUT_FOLDER
            / f"frame_{frame_number}.jpg"
        )

        success = cv2.imwrite(
            str(output_file),
            frame
        )

        if success:

            print(
                f"Saved -> {output_file}"
            )

        else:

            print(
                f"Could not save "
                f"{output_file}"
            )

        saved_frames += 1

        if saved_frames >= NUMBER_OF_TEST_FRAMES:
            break

    print()
    print("--------------------------------")

    if saved_frames == 0:

        print(
            "No frames were extracted."
        )

    else:

        print(
            f"SUCCESS: Extracted "
            f"{saved_frames} test frames."
        )

    print("--------------------------------")


if __name__ == "__main__":
    main()