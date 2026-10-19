"""Passive CUDA allocator diagnostics, including samples around runner cleanup."""

import json
import os
from pathlib import Path
import sys
import time


class CudaMemoryLogger:
    def __init__(self, output, every, cuda):
        self.output = Path(output)
        self.every = every
        self.cuda = cuda
        self.started = time.perf_counter()
        self.context = {}
        self.allocator = cuda.get_allocator_backend()
        self.allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

    def progress(self, setting, event):
        # Keep only scalars, never an event object that could retain a tensor.
        self.context = {"setting": setting, **{
            key: event[key] for key in ("fold", "seed", "step") if key in event
        }}
        if "step" in event:
            if event["step"] % self.every:
                return
            # The existing training callback runs after the objective, before backward.
            phase = "training_objective"
        else:
            phase = event.get("event", "progress")
        self.record(phase)

    def record(self, phase):
        """Log without synchronizing CUDA or resetting peaks, to limit perturbation."""
        try:
            stats = self.cuda.memory_stats()
            allocated = self.cuda.memory_allocated()
            reserved = self.cuda.memory_reserved()
            record = {
                **self.context, "event": phase,
                "elapsed_seconds": time.perf_counter() - self.started,
                "allocator_backend": self.allocator,
                "pytorch_cuda_alloc_conf": self.allocator_config,
                "allocated_bytes": allocated, "reserved_bytes": reserved,
                "reserved_unallocated_bytes": reserved - allocated,
                "peak_allocated_bytes": self.cuda.max_memory_allocated(),
                "peak_reserved_bytes": self.cuda.max_memory_reserved(),
                "active_bytes": stats.get("active_bytes.all.current"),
                "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
                "allocation_retries": stats.get("num_alloc_retries"),
                "oom_count": stats.get("num_ooms"),
            }
            # Keep allocator statistics even if a CUDA error prevents a driver query.
            try:
                free, total = self.cuda.mem_get_info()
                record.update(device_free_bytes=free, device_total_bytes=total)
            except Exception as error:
                record["device_query_error"] = str(error)
            with (self.output / "cuda_memory.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")

            def gib(key):
                value = record.get(key)
                return "n/a" if value is None else f"{value / 2**30:.3f}"

            location = " ".join(f"{key}={value}" for key, value in self.context.items())
            print(
                f"[CUDA memory] {location} event={phase} GiB: "
                f"allocated={gib('allocated_bytes')} reserved={gib('reserved_bytes')} "
                f"unused={gib('reserved_unallocated_bytes')} "
                f"inactive_split={gib('inactive_split_bytes')} "
                f"peak_allocated={gib('peak_allocated_bytes')} "
                f"peak_reserved={gib('peak_reserved_bytes')} "
                f"device_free={gib('device_free_bytes')}; "
                f"alloc_retries={record['allocation_retries']} ooms={record['oom_count']}",
                flush=True,
            )
        except Exception as error:
            # Diagnostic failures must not replace the training exception.
            print(f"warning: CUDA memory report failed: {error}", file=sys.stderr, flush=True)

    def failure(self):
        # The exception traceback can still own tensors here; this is not a post-cleanup sample.
        self.record("failure")
        try:
            summary = self.cuda.memory_summary(abbreviated=False)
            path = self.output / "cuda_memory_failure.txt"
            path.write_text(summary, encoding="utf-8")
            print(f"CUDA allocator summary: {path}\n{summary}", file=sys.stderr, flush=True)
        except Exception as error:
            print(f"warning: CUDA allocator summary failed: {error}", file=sys.stderr, flush=True)
