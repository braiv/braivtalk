#!/usr/bin/env python3
"""
Parallel I/O utilities for frame processing.
Provides optimized frame writing with threading for improved performance.
"""

import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelFrameWriter:
    """
    Parallel frame writer for improved I/O performance.

    Uses ThreadPoolExecutor to write frames to disk in parallel.
    """

    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self.futures = []
        self.frames_written = 0
        self.start_time = time.time()

    def write_frame_async(self, executor, frame_idx, frame_array, output_path):
        future = executor.submit(self._write_frame, frame_idx, frame_array, output_path)
        self.futures.append(future)
        return future

    def _write_frame(self, frame_idx, frame_array, output_path):
        try:
            filename = f"{output_path}/{str(frame_idx).zfill(8)}.png"
            success = cv2.imwrite(filename, frame_array)
            if success:
                self.frames_written += 1
                return True
            else:
                print(f"❌ Failed to write frame {frame_idx}")
                return False
        except Exception as e:
            print(f"❌ Error writing frame {frame_idx}: {e}")
            return False

    def wait_for_completion(self):
        if not self.futures:
            return

        print(f"⏳ Finalizing {len(self.futures)} frame writes...")

        completed = 0
        for future in as_completed(self.futures):
            try:
                future.result(timeout=30)
                completed += 1
                if completed % 100 == 0:
                    print(f"📊 Completed {completed}/{len(self.futures)} writes")
            except Exception as e:
                print(f"❌ Frame write error: {e}")

        elapsed = time.time() - self.start_time
        fps = self.frames_written / elapsed if elapsed > 0 else 0
        print(f"✅ Parallel I/O: Wrote {self.frames_written} frames in {elapsed:.2f}s ({fps:.1f} FPS)")

    def get_stats(self):
        elapsed = time.time() - self.start_time
        fps = self.frames_written / elapsed if elapsed > 0 else 0
        return {
            "frames_written": self.frames_written,
            "elapsed_time": elapsed,
            "fps": fps,
            "pending_writes": len([f for f in self.futures if not f.done()]),
            "completed_writes": len([f for f in self.futures if f.done()]),
        }


def create_parallel_writer(max_workers=4):
    return ParallelFrameWriter(max_workers=max_workers)


def write_frames_parallel(frames_data, output_path, max_workers=4, show_progress=True):
    writer = ParallelFrameWriter(max_workers=max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for frame_idx, frame_array in frames_data:
            writer.write_frame_async(executor, frame_idx, frame_array, output_path)

        if show_progress:
            writer.wait_for_completion()

    return writer.get_stats()

