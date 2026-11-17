"""Bounded asynchronous video capture, ONNX inference, and latency reporting."""

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from surveillance_ml.detection.onnx import DetectionInference, OnnxDetector

VideoSource = str | int | Path


@dataclass(frozen=True)
class _Frame:
    index: int
    captured_at: float
    image: np.ndarray
    fps: float


@dataclass(frozen=True)
class _CaptureFailure:
    message: str


def _is_live_source(source: VideoSource) -> bool:
    return isinstance(source, int) or (
        isinstance(source, str)
        and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
    )


def _open_capture(source: VideoSource) -> cv2.VideoCapture:
    if isinstance(source, Path):
        capture = cv2.VideoCapture(str(source))
    elif isinstance(source, int):
        capture = cv2.VideoCapture(source)
    elif source.lower().startswith(("rtsp://", "rtsps://")):
        parameters = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            5000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            5000,
        ]
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, parameters)
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Unable to open the configured video source")
    return capture


def _annotate(frame: np.ndarray, result: DetectionInference) -> np.ndarray:
    annotated = frame.copy()
    colors = ((35, 115, 205), (230, 130, 35))
    for detection in result.detections:
        color = colors[detection.class_id % len(colors)]
        x1 = int(round(detection.x))
        y1 = int(round(detection.y))
        x2 = int(round(detection.x + detection.width))
        y2 = int(round(detection.y + detection.height))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_video_inference(
    detector: OnnxDetector,
    source: VideoSource,
    *,
    target_fps: float | None = None,
    max_frames: int | None = None,
    annotated_output: Path | None = None,
    reconnect_attempts: int = 3,
    reconnect_backoff_seconds: float = 0.5,
) -> dict[str, Any]:
    """Run bounded video inference; live sources replace stale queued frames."""
    if target_fps is not None and target_fps <= 0:
        raise ValueError("target_fps must be positive when supplied")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when supplied")
    if reconnect_attempts < 0 or reconnect_backoff_seconds < 0:
        raise ValueError("Reconnect attempts and delay must be non-negative")
    live = _is_live_source(source)
    frame_queue: queue.Queue[_Frame | _CaptureFailure | None] = queue.Queue(
        maxsize=1 if live else 8
    )
    stop = threading.Event()
    state_lock = threading.Lock()
    state = {"dropped_frames": 0, "reconnections": 0}
    producer_error: list[BaseException] = []

    def publish(frame: _Frame) -> None:
        if not live:
            while not stop.is_set():
                try:
                    frame_queue.put(frame, timeout=0.1)
                    return
                except queue.Full:
                    continue
            return
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                with state_lock:
                    state["dropped_frames"] += 1
            frame_queue.put_nowait(frame)

    def capture_frames() -> None:
        capture: cv2.VideoCapture | None = None
        frame_index = 0
        attempts = 0
        try:
            while not stop.is_set():
                if capture is None:
                    try:
                        capture = _open_capture(source)
                    except RuntimeError:
                        if not live or attempts >= reconnect_attempts:
                            raise
                        attempts += 1
                        with state_lock:
                            state["reconnections"] += 1
                        stop.wait(reconnect_backoff_seconds * attempts)
                        continue
                read_ok, frame = capture.read()
                if not read_ok:
                    if not live:
                        break
                    capture.release()
                    capture = None
                    if attempts >= reconnect_attempts:
                        raise RuntimeError(
                            "Live video source disconnected after reconnect attempts"
                        )
                    attempts += 1
                    with state_lock:
                        state["reconnections"] += 1
                    stop.wait(reconnect_backoff_seconds * attempts)
                    continue
                attempts = 0
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                packet = _Frame(frame_index, time.perf_counter(), frame, fps)
                frame_index += 1
                publish(packet)
        except BaseException as error:
            producer_error.append(error)
            while not stop.is_set():
                try:
                    frame_queue.put(_CaptureFailure(str(error)), timeout=0.1)
                    break
                except queue.Full:
                    continue
        finally:
            if capture is not None:
                capture.release()
            if not stop.is_set():
                while not stop.is_set():
                    try:
                        frame_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    worker = threading.Thread(target=capture_frames, name="video-capture", daemon=True)
    worker.start()
    latencies: list[float] = []
    inference_latencies: list[float] = []
    processed_frames = 0
    sampled_frames = 0
    first_frame_fps = 0.0
    output: cv2.VideoWriter | None = None
    started = time.perf_counter()
    last_inference_at = 0.0

    try:
        while max_frames is None or processed_frames < max_frames:
            try:
                packet = frame_queue.get(timeout=0.2)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            if packet is None:
                break
            if isinstance(packet, _CaptureFailure):
                raise RuntimeError(packet.message)
            if first_frame_fps <= 0:
                first_frame_fps = packet.fps
            if target_fps is not None and time.perf_counter() - last_inference_at < 1 / target_fps:
                sampled_frames += 1
                continue

            rgb = np.ascontiguousarray(packet.image[:, :, ::-1])
            result = detector.predict_rgb(rgb)
            completed_at = time.perf_counter()
            inference_latencies.append(result.elapsed_seconds)
            latencies.append(completed_at - packet.captured_at)
            processed_frames += 1
            last_inference_at = completed_at

            if annotated_output is not None:
                if output is None:
                    annotated_output.parent.mkdir(parents=True, exist_ok=True)
                    fps = first_frame_fps if first_frame_fps > 0 else target_fps or 25.0
                    codec = cv2.VideoWriter.fourcc(*"mp4v")
                    output = cv2.VideoWriter(
                        str(annotated_output),
                        codec,
                        fps,
                        (packet.image.shape[1], packet.image.shape[0]),
                    )
                    if not output.isOpened():
                        raise RuntimeError("Unable to create the requested annotated video")
                output.write(_annotate(packet.image, result))
    finally:
        stop.set()
        worker.join(timeout=10)
        if worker.is_alive():
            raise RuntimeError("Video capture did not stop within the 10-second shutdown limit")
        if output is not None:
            output.release()

    if producer_error:
        raise RuntimeError("Video capture failed") from producer_error[0]
    if processed_frames == 0:
        raise RuntimeError("Video source produced no frames for inference")
    elapsed = time.perf_counter() - started
    latency_array = np.asarray(latencies, dtype=np.float64) * 1000
    inference_array = np.asarray(inference_latencies, dtype=np.float64) * 1000
    with state_lock:
        dropped_frames = state["dropped_frames"]
        reconnections = state["reconnections"]
    return {
        "processed_frames": processed_frames,
        "dropped_frames": dropped_frames + sampled_frames,
        "reconnections": reconnections,
        "source_fps": first_frame_fps or None,
        "target_fps": target_fps,
        "elapsed_seconds": elapsed,
        "throughput_fps": processed_frames / elapsed if elapsed > 0 else 0.0,
        "end_to_end_latency_ms": {
            "mean": float(np.mean(latency_array)),
            "p50": float(np.percentile(latency_array, 50)),
            "p95": float(np.percentile(latency_array, 95)),
            "max": float(np.max(latency_array)),
        },
        "onnx_latency_ms": {
            "mean": float(np.mean(inference_array)),
            "p50": float(np.percentile(inference_array, 50)),
            "p95": float(np.percentile(inference_array, 95)),
            "max": float(np.max(inference_array)),
        },
        "annotated_video": str(annotated_output) if annotated_output is not None else None,
    }
