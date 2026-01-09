"""
gpu_stress.py
=============
Stress-tests *all* NVIDIA GPUs in a machine with **high compute utilisation (≈80-90 %)**
while keeping the **memory footprint small (<100 MB per GPU)**.

How it works
------------
1.  Waits until every GPU is < `IDLE_THRESHOLD` % busy.
2.  Forks one process **per GPU**.
   • Each process pins itself to a single device.
   • Allocates two tiny fp16 matrices (default 2048×2048 → 32 MiB total).
   • Launches many batched GEMMs on several CUDA streams to flood the SMs.
3.  `Ctrl-C` anywhere cleanly stops all workers.

Requirements
------------
pip install torch pynvml
(Any recent PyTorch build with CUDA support is fine).

Run
---
python gpu_stress.py            # use defaults
python gpu_stress.py --dim 4096 # heavier compute, still small memory (≈64 MiB)
"""
from __future__ import annotations
import argparse
import signal
import sys
import time
from multiprocessing import Event, Process

import torch
from pynvml import (
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)

IDLE_THRESHOLD = 5  # %   – start only when every GPU is below this

# --------------------------------------------------------------------------- #
#  Utility: wait until the whole machine is quiet
# --------------------------------------------------------------------------- #
def wait_for_idle(threshold: int = IDLE_THRESHOLD, interval: int = 30) -> None:
    nvmlInit()
    n_gpu = nvmlDeviceGetCount()
    flag = 0
    while True:
        busy = [
            nvmlDeviceGetUtilizationRates(nvmlDeviceGetHandleByIndex(i)).gpu
            for i in range(n_gpu)
        ]
        if max(busy) < threshold:
            flag += 1
        else:
            flag = 0

        if flag > 10:
            return
        print(f"[monitor] utilisation {busy} - flag {flag} – sleeping {interval}s …", flush=True)
        time.sleep(interval)


# --------------------------------------------------------------------------- #
#  Worker: saturate ONE gpu
# --------------------------------------------------------------------------- #
def hammer(
    device_id: int,
    stop: Event,
    dim: int,
    batch: int,
    n_streams: int,
) -> None:
    torch.cuda.set_device(device_id)
    torch.backends.cudnn.benchmark = True       # let cuDNN pick fastest kernels
    dtype = torch.float16                       # fp16 → tiny memory + tensor-core speed

    # Two input tensors (A, B) – reused forever => constant memory
    a = torch.randn(batch, dim, dim, device=device_id, dtype=dtype)
    b = torch.randn(batch, dim, dim, device=device_id, dtype=dtype)

    streams = [torch.cuda.Stream() for _ in range(n_streams)]

    while not stop.is_set():
        for s in streams:                       # launch work on all streams (async)
            with torch.cuda.stream(s):
                torch.bmm(a, b, out=a)          # overwrite A in-place (keeps mem small)
        torch.cuda.synchronize(device_id)       # force completion before next loop


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2048,
                        help="matrix dimension (memory ~ 2*dim^2*2 bytes)")
    parser.add_argument("--batch", type=int, default=2,
                        help="batch of small matrices per stream")
    parser.add_argument("--streams", type=int, default=4,
                        help="concurrent CUDA streams per GPU")
    args = parser.parse_args()

    wait_for_idle()

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        print("No CUDA devices detected.", file=sys.stderr)
        sys.exit(1)

    stop_event = Event()
    workers: list[Process] = []

    # graceful Ctrl-C: set the event then join children
    def _signal_handler(sig, frame):
        print("\n[main] Caught Ctrl-C – stopping …", flush=True)
        stop_event.set()
        for p in workers:
            p.join()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    for i in range(n_gpu):
        p = Process(
            target=hammer,
            args=(i, stop_event, args.dim, args.batch, args.streams),
            daemon=True,
        )
        p.start()
        workers.append(p)

    print(
        f"[main] launched {n_gpu} worker(s) – util should climb to ≈90 % "
        f"with only {(2*args.batch*args.dim**2*2)/1024/1024:.1f} MiB per GPU."
    )
    # Sleep forever; Ctrl-C handled by signal
    while any(p.is_alive() for p in workers):
        time.sleep(1)


if __name__ == "__main__":
    main()