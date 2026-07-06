
from basketball_vision_analyser.video import (
    InvalidFrameError,
    VideoError,
    VideoOpenError,
    VideoReadError,
    VideoWriteError,
)


def test_video_exceptions_inherit_from_video_error() -> None:
    assert issubclass(VideoOpenError, VideoError)
    assert issubclass(VideoReadError, VideoError)
    assert issubclass(VideoWriteError, VideoError)
    assert issubclass(InvalidFrameError, VideoError)
