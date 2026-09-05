import cv2


def read_video(path):
    """
    Open a video file and return its frames one by one.

    Parameters
    ----------
    path : str
        Path to the video file.

    Yields
    ------
    frame
        One OpenCV image/frame at a time.
    """

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {path}"
        )

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            yield frame

    finally:
        cap.release()


def read_sampled_frames(path, every_n=15):
    """
    Read only every Nth frame from the video.

    Example:
    every_n=15 means:
    frame 15
    frame 30
    frame 45
    ...

    This avoids sending every video frame to the AI.
    """

    if every_n < 1:
        raise ValueError(
            "every_n must be at least 1"
        )

    frame_number = 0

    for frame in read_video(path):

        frame_number += 1

        if frame_number % every_n != 0:
            continue

        yield frame_number, frame


def get_video_info(path):
    """
    Read useful information about a video.
    """

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if fps > 0:
        duration_seconds = total_frames / fps
    else:
        duration_seconds = 0

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }