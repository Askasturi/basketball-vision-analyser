
"""Video-related exceptions for Basketball Vision Analyser."""


class VideoError(Exception):
    """Base exception for video processing errors."""


class VideoOpenError(VideoError):
    """Raised when a video file cannot be opened."""


class VideoReadError(VideoError):
    """Raised when a video frame cannot be read."""


class VideoWriteError(VideoError):
    """Raised when a video file cannot be written."""


class InvalidFrameError(VideoError):
    """Raised when a frame has an invalid shape or type."""
