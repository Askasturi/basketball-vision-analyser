
"""Simple IoU-based multi-object tracker."""

from __future__ import annotations

from dataclasses import dataclass

from basketball_vision_analyser.detection import (
    Detection,
    DetectionResult,
)
from basketball_vision_analyser.tracking.config import TrackingConfig
from basketball_vision_analyser.tracking.result import TrackingResult
from basketball_vision_analyser.tracking.track import TrackedObject


@dataclass
class _ActiveTrack:
    track_id: int
    detection: Detection
    age: int
    hits: int
    lost_frames: int
    last_frame_index: int


class SimpleTracker:
    """Track detections across frames using greedy IoU matching."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self._tracks: dict[int, _ActiveTrack] = {}
        self._next_track_id = 0

    def reset(self) -> None:
        """Clear all active tracks."""

        self._tracks.clear()
        self._next_track_id = 0

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """Update tracks using detections from one frame."""

        frame_index = detection_result.frame_index
        detections = self._filter_detections(detection_result.detections)

        matches = self._match_detections(detections)
        matched_track_ids = {track_id for track_id, _ in matches}
        matched_detection_indices = {index for _, index in matches}

        tracked_objects: list[TrackedObject] = []

        for track_id, detection_index in matches:
            detection = detections[detection_index].with_track_id(track_id)
            active_track = self._tracks[track_id]
            active_track.detection = detection
            active_track.age += 1
            active_track.hits += 1
            active_track.lost_frames = 0
            active_track.last_frame_index = frame_index

            tracked_objects.append(
                self._tracked_object_from_active_track(active_track, frame_index)
            )

        new_track_ids: set[int] = set()

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue

            active_track = self._start_track(
                detection=detection,
                frame_index=frame_index,
            )
            new_track_ids.add(active_track.track_id)
            tracked_objects.append(
                self._tracked_object_from_active_track(active_track, frame_index)
            )

        self._age_unmatched_tracks(
            matched_track_ids=matched_track_ids,
            new_track_ids=new_track_ids,
        )

        tracked_objects.sort(key=lambda obj: obj.track_id)

        return TrackingResult(
            frame_index=frame_index,
            objects=tuple(tracked_objects),
            metadata={
                "tracker": "simple_iou",
                "active_tracks": len(self._tracks),
            },
        )

    def _filter_detections(
        self,
        detections: tuple[Detection, ...],
    ) -> tuple[Detection, ...]:
        return tuple(
            detection
            for detection in detections
            if detection.class_name in self.config.track_classes
            and detection.confidence >= self.config.min_confidence
        )

    def _match_detections(
        self,
        detections: tuple[Detection, ...],
    ) -> list[tuple[int, int]]:
        candidate_matches: list[tuple[float, int, int]] = []

        for track_id, active_track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                if active_track.detection.class_name != detection.class_name:
                    continue

                score = active_track.detection.box.iou(detection.box)
                if score >= self.config.iou_threshold:
                    candidate_matches.append((score, track_id, detection_index))

        candidate_matches.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _score, track_id, detection_index in candidate_matches:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            matches.append((track_id, detection_index))

        return matches

    def _start_track(
        self,
        detection: Detection,
        frame_index: int,
    ) -> _ActiveTrack:
        track_id = self._next_track_id
        self._next_track_id += 1

        detection = detection.with_track_id(track_id)

        active_track = _ActiveTrack(
            track_id=track_id,
            detection=detection,
            age=1,
            hits=1,
            lost_frames=0,
            last_frame_index=frame_index,
        )
        self._tracks[track_id] = active_track

        return active_track

    def _age_unmatched_tracks(
        self,
        *,
        matched_track_ids: set[int],
        new_track_ids: set[int],
    ) -> None:
        expired_track_ids: list[int] = []

        for track_id, active_track in self._tracks.items():
            if track_id in matched_track_ids or track_id in new_track_ids:
                continue

            active_track.lost_frames += 1
            active_track.age += 1

            if active_track.lost_frames > self.config.max_lost_frames:
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            del self._tracks[track_id]

    @staticmethod
    def _tracked_object_from_active_track(
        active_track: _ActiveTrack,
        frame_index: int,
    ) -> TrackedObject:
        return TrackedObject(
            track_id=active_track.track_id,
            detection=active_track.detection,
            frame_index=frame_index,
            age=active_track.age,
            hits=active_track.hits,
            lost_frames=active_track.lost_frames,
        )

    @property
    def active_track_count(self) -> int:
        """Return number of active tracks."""

        return len(self._tracks)

    @property
    def next_track_id(self) -> int:
        """Return the next track ID that will be assigned."""

        return self._next_track_id
