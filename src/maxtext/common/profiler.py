# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dispatch to the chosen profiler."""

from ctypes import cdll
import os
import shutil
import subprocess
import threading
from typing import Any

import jax

try:
  from maxtext.common.gcloud_stub import mldiagnostics_modules
  from maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
  from maxtext.utils import max_logging
except ImportError:
  from maxtext.src.maxtext.common.gcloud_stub import mldiagnostics_modules
  from maxtext.src.maxtext.common.managed_mldiagnostics import ManagedMLDiagnostics
  from maxtext.src.maxtext.utils import max_logging

mldiag, _ = mldiagnostics_modules()


class Profiler:
  """Activate/deactivate a profiler based on the 'profiler' config."""

  def __init__(self, config, offset_step=0):
    self.libcudart = None
    self.mode = config.profiler
    if self.mode != "":
      self.base_output_dir = config.tensorboard_dir
    self.output_path = ""
    self.upload_all_profiler_results = config.upload_all_profiler_results
    self.profile_cleanly = config.profile_cleanly
    self.profile_period = config.profile_periodically_period
    self.start_initial_profile_step = self._set_first_profiler_step(config.skip_first_n_steps_for_profiler, offset_step)
    self.finished_initial_profile_step = self._set_last_profiler_step(config.profiler_steps, config.steps)
    if config.profiler != "" and self.start_initial_profile_step >= config.steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    self.prof = None  # managed mldiagnostics xprof collector.
    self.managed_mldiagnostics = config.managed_mldiagnostics
    if config.managed_mldiagnostics:
      ManagedMLDiagnostics(config)  # Initialize the MLRun instance.

    self._lock = threading.RLock()
    self._is_active = False
    self._active_session_id = None

    self.profiling_options = jax.profiler.ProfileOptions()
    advanced_config = {}

    if self.mode == "xplane" and not self.managed_mldiagnostics and config.profile_power_events:
      advanced_config.update(
          {
              "tpu_power_trace_level": config.xprof_tpu_power_trace_level,
              "e2e_enable_fw_throttle_event": config.xprof_e2e_enable_fw_throttle_event,
              "e2e_enable_fw_power_level_event": config.xprof_e2e_enable_fw_power_level_event,
              "e2e_enable_fw_thermal_event": config.xprof_e2e_enable_fw_thermal_event,
          }
      )

    if self.mode == "xplane" and config.enable_tpu_profiling_options:
      advanced_config.update(
          {
              "tpu_num_chips_to_profile_per_task": config.tpu_num_chips_to_profile_per_task,
              "tpu_num_sparse_core_tiles_to_trace": config.tpu_num_sparse_core_tiles_to_trace,
              "tpu_num_sparse_cores_to_trace": config.tpu_num_sparse_cores_to_trace,
          }
      )

    if advanced_config:
      self.profiling_options.advanced_configuration = advanced_config

  @property
  def is_active(self) -> bool:
    """Returns True if the profiler is currently tracing."""
    with self._lock:
      return self._is_active

  def maybe_activate_profiler(self, step: int, state: Any | None = None) -> None:
    """Conditionally activates the profiler based on the current step.

    This method checks if the current training step matches the step designated
    for starting an initial profile, or if it meets the criteria for
    activating a new periodic profile.
    """
    if self.mode != "" and (step == self.start_initial_profile_step or self.should_activate_periodic_profile(step)):
      optional_postfix = f"step_{step}" if self.profile_period > 0 else ""
      self.activate(blocking_object=state, optional_postfix=optional_postfix)

  def activate(
      self,
      session_id: str | None = None,
      blocking_object: Any | None = None,
      optional_postfix: str = "",
  ) -> None:
    """Start the profiler.

    nsys profiler becomes no-op when libcudart.so is not available on the system.
    """
    if optional_postfix and not session_id:
      session_id = optional_postfix
    if not session_id:
      session_id = "default_session"

    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)

    with self._lock:
      if self._is_active:
        return
      self._is_active = True
      self._active_session_id = session_id

      if self.managed_mldiagnostics and self.mode == "xplane":
        # Handle the special profiling logic for managed_mldiagnostics
        if self.prof is None:
          # Starts xprof collector.
          # Only profiling on the first device, if not upload_all_profiler_results. None is for all devices.
          self.prof = mldiag.xprof(
              process_index_list=None
              if self.upload_all_profiler_results
              else [0]
          )
        self.prof.start(session_id=session_id)
        return

      if not (self.upload_all_profiler_results or jax.process_index() == 0):
        return
      if self.mode != "":
        self.output_path = (
            os.path.join(self.base_output_dir, session_id)
            if session_id
            else self.base_output_dir
        )
      if self.mode == "nsys":
        try:
          self.libcudart = cdll.LoadLibrary("libcudart.so")
        except OSError as e:
          max_logging.warning(
              "Failed to load library for nsys: %s\nprofiler has no effect", e
          )
          return
        self.libcudart.cudaProfilerStart()
      elif self.mode == "xplane":
        options = (
            self.profiling_options
            if self.profiling_options is not None
            else (
                jax.profiler.ProfileOptions()
                if hasattr(jax.profiler, "ProfileOptions")
                else None
            )
        )
        if options is not None:
          options.session_id = session_id
          jax.profiler.start_trace(
              self.base_output_dir, profiler_options=options
          )
        else:
          jax.profiler.start_trace(self.output_path)

  def maybe_deactivate_profiler(self, step: int, state: Any | None = None) -> None:
    """Conditionally deactivates the profiler based on the current step.

    This method checks if the current training step matches the step designated
    for finishing the initial profile, or if it meets the criteria for
    deactivating a periodic profile.
    """
    if self.mode != "" and (step == self.finished_initial_profile_step or self.should_deactivate_periodic_profile(step)):
      self.deactivate(blocking_object=state)

  def deactivate(self, blocking_object: Any | None = None) -> None:
    """End the profiler.

    The result is uploaded to the output bucket.
    """
    if self.profile_cleanly and blocking_object is not None:
      jax.block_until_ready(blocking_object)

    with self._lock:
      if not self._is_active:
        return
      try:
        if self.managed_mldiagnostics and self.mode == "xplane":
          # Handle the special profiling logic for managed_mldiagnostics
          if self.prof is not None:
            self.prof.stop()
          return

        if not (self.upload_all_profiler_results or jax.process_index() == 0):
          return
        if self.mode == "nsys":
          if self.libcudart is not None:
            self.libcudart.cudaProfilerStop()
          else:
            max_logging.warning(
                "Library for nsys was not loaded\nprofiler has no effect"
            )
            return
          # Popen() instead of run() for non-blocking behavior
          if shutil.which("gcloud") is not None:
            subprocess.Popen(["gcloud", "storage", "cp", "*nsys-rep", self.output_path])  # pylint: disable=consider-using-with
          else:
            max_logging.warning(
                "gcloud is not installed or not found in the system's"
                " PATH. Skipping upload..."
            )
        elif self.mode == "xplane":
          jax.profiler.stop_trace()
      finally:
        self._is_active = False
        self._active_session_id = None

  def _set_first_profiler_step(self, skip_steps, start_step):
    return start_step + skip_steps

  def _set_last_profiler_step(self, profiler_steps, last_job_step):
    return min(self.start_initial_profile_step + profiler_steps - 1, last_job_step - 1)

  def should_activate_periodic_profile(self, step):
    return self.profile_period > 0 and (step - self.start_initial_profile_step) % self.profile_period == 0

  def should_deactivate_periodic_profile(self, step):
    return self.profile_period > 0 and (step - self.finished_initial_profile_step) % self.profile_period == 0

  def post_process(self):
    pass
