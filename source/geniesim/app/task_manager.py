# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import threading
import traceback


class TaskManager:
    def __init__(self, api_core, benchmark_config):
        self.api_core = api_core
        self.benchmark_config = benchmark_config
        self._worker_thread = None

    def start(self):
        self._worker_thread = threading.Thread(target=self.worker, daemon=True)
        self._worker_thread.start()

    def join(self, timeout=None):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

    def worker(self):
        import sys
        try:
            print("[TaskManager] worker thread started", flush=True)
            from geniesim.benchmark.task_benchmark import main as benchmark_main
            print("[TaskManager] import done, calling benchmark_main", flush=True)
            benchmark_main(self.benchmark_config, self.api_core)
            print("[TaskManager] benchmark_main returned", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"[TaskManager] worker thread crashed: {e}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
