# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import signal
import sys
import threading
import traceback


def watchdog_thread(timeout: float):
    def print_thread_stacks():
        for thread in threading.enumerate():
            print(f"Thread ID: {thread.ident}")
            print("".join(traceback.format_stack(sys._current_frames()[thread.ident])))
        print("Killing main process", flush=True)
        # Exit the main thread
        os.kill(os.getpid(), signal.SIGINT)

    timer = threading.Timer(timeout, print_thread_stacks)
    timer.start()
