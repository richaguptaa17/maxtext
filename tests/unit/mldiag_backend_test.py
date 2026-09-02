# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for MLDiagScalarBackend."""

import datetime
from unittest import mock

import unittest

try:
  from absl.testing import absltest
  _TestCase = absltest.TestCase
except ImportError:
  _TestCase = unittest.TestCase

try:
  import jax.numpy as jnp
except ImportError:
  jnp = None
from maxtext.trainers.post_train.rl import mldiag_backend
import numpy as np

_RECORD_TARGET = (
    "maxtext.trainers.post_train.rl.mldiag_backend."
    "mldiag_metrics.record"
)
_METRICS_TARGET = (
    "maxtext.trainers.post_train.rl.mldiag_backend."
    "mldiag_metrics"
)
_MLDIAG_TARGET = (
    "maxtext.trainers.post_train.rl.mldiag_backend.mldiag"
)


class MLDiagScalarBackendTest(_TestCase):

  def test_extract_scalar_valid_types(self):
    self.assertEqual(mldiag_backend._extract_scalar(42), 42)
    self.assertEqual(mldiag_backend._extract_scalar(3.14), 3.14)
    self.assertEqual(mldiag_backend._extract_scalar(np.float32(1.5)), 1.5)
    self.assertEqual(mldiag_backend._extract_scalar(np.int64(10)), 10)
    if jnp is not None:
      val = mldiag_backend._extract_scalar(jnp.array(2.718))
      self.assertIsNotNone(val)
      self.assertAlmostEqual(val, 2.718, places=3)
    self.assertEqual(mldiag_backend._extract_scalar(np.array([5.0])), 5.0)

  def test_extract_scalar_type_preservation(self):
    val_int = mldiag_backend._extract_scalar(42)
    self.assertIsInstance(val_int, int)
    self.assertEqual(val_int, 42)

    val_np_int = mldiag_backend._extract_scalar(np.int64(10))
    self.assertIsInstance(val_np_int, int)
    self.assertEqual(val_np_int, 10)

    val_np_int32 = mldiag_backend._extract_scalar(np.int32(7))
    self.assertIsInstance(val_np_int32, int)
    self.assertEqual(val_np_int32, 7)

    val_float = mldiag_backend._extract_scalar(3.14)
    self.assertIsInstance(val_float, float)
    self.assertEqual(val_float, 3.14)

    val_np_float = mldiag_backend._extract_scalar(np.float32(1.5))
    self.assertIsInstance(val_np_float, float)
    self.assertEqual(val_np_float, 1.5)

  def test_extract_scalar_invalid_types(self):
    self.assertIsNone(mldiag_backend._extract_scalar(None))
    self.assertIsNone(mldiag_backend._extract_scalar(True))
    self.assertIsNone(mldiag_backend._extract_scalar(False))
    self.assertIsNone(mldiag_backend._extract_scalar(np.bool_(True)))
    self.assertIsNone(mldiag_backend._extract_scalar(np.bool_(False)))
    self.assertIsNone(mldiag_backend._extract_scalar(np.array(True)))
    self.assertIsNone(mldiag_backend._extract_scalar(np.array([False])))
    self.assertIsNone(mldiag_backend._extract_scalar([1, 2, 3]))
    self.assertIsNone(mldiag_backend._extract_scalar(np.array([1.0, 2.0])))
    self.assertIsNone(mldiag_backend._extract_scalar("not_a_number"))
    self.assertIsNone(mldiag_backend._extract_scalar("123"))
    self.assertIsNone(mldiag_backend._extract_scalar("3.14"))
    self.assertIsNone(mldiag_backend._extract_scalar(b"123"))
    self.assertIsNone(mldiag_backend._extract_scalar(b"3.14"))
    self.assertIsNone(mldiag_backend._extract_scalar(b"not_a_number"))
    self.assertIsNone(mldiag_backend._extract_scalar(np.array("123")))
    self.assertIsNone(mldiag_backend._extract_scalar(np.array(["123"])))
    self.assertIsNone(mldiag_backend._extract_scalar(np.array(b"123")))
    self.assertIsNone(mldiag_backend._extract_scalar(np.str_("123")))
    self.assertIsNone(mldiag_backend._extract_scalar(np.bytes_(b"123")))

  def test_log_scalar_non_zero_process_index_noop(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=1):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", 0.5, step=1)
        mock_record.assert_not_called()

  def test_log_scalar_nan_inf_none_ignored(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", float("nan"), step=1)
        backend.log_scalar("actor/train/loss", float("inf"), step=1)
        backend.log_scalar("actor/train/loss", -float("inf"), step=1)
        backend.log_scalar("actor/train/loss", None, step=1)
        mock_record.assert_not_called()

  def test_log_scalar_string_and_bytes_ignored(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", "0.5", step=1)
        backend.log_scalar("actor/train/loss", "123", step=1)
        backend.log_scalar("actor/train/loss", b"0.5", step=1)
        backend.log_scalar("actor/train/loss", np.array("0.5"), step=1)
        backend.log_scalar("actor/train/loss", True, step=1)
        mock_record.assert_not_called()

  def test_log_scalar_exact_metric_mapping(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("loss", 0.125, step=10)
        if (
            mldiag_backend.metric_types is not None
            and hasattr(mldiag_backend.metric_types.MetricType, "LOSS")
        ):
          expected_key = mldiag_backend.metric_types.MetricType.LOSS
        else:
          expected_key = "loss"
        mock_record.assert_called_once_with(
            expected_key, 0.125, step=10, timestamp=mock.ANY
        )
        _, kwargs = mock_record.call_args
        timestamp = kwargs.get("timestamp")
        self.assertIsInstance(timestamp, datetime.datetime)
        self.assertEqual(timestamp.tzinfo, datetime.timezone.utc)

  def test_log_scalar_all_exact_metric_mappings(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    for event_name, enum_name in mldiag_backend._EXACT_METRIC_MAP.items():
      with mock.patch("jax.process_index", return_value=0):
        with mock.patch(_RECORD_TARGET) as mock_record:
          backend.log_scalar(event_name, 1.23, step=7)
          if (
              mldiag_backend.metric_types is not None
              and hasattr(mldiag_backend.metric_types.MetricType, enum_name)
          ):
            expected_key = getattr(
                mldiag_backend.metric_types.MetricType, enum_name
            )
          else:
            expected_key = event_name
          mock_record.assert_called_once_with(
              expected_key, 1.23, step=7, timestamp=mock.ANY
          )
          _, kwargs = mock_record.call_args
          timestamp = kwargs.get("timestamp")
          self.assertIsInstance(timestamp, datetime.datetime)
          self.assertEqual(timestamp.tzinfo, datetime.timezone.utc)

  def test_log_scalar_hierarchical_event_preserves_full_name(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", 0.125, step=10)
        mock_record.assert_called_once_with(
            "actor/train/loss", 0.125, step=10, timestamp=mock.ANY
        )

  def test_log_scalar_multi_model_metrics_no_collision(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", 0.25, step=1)
        backend.log_scalar("critic/train/loss", 0.5, step=1)
        backend.log_scalar("rewards/score", 1.0, step=1)
        self.assertEqual(mock_record.call_count, 3)
        self.assertEqual(
            mock_record.call_args_list[0],
            mock.call("actor/train/loss", 0.25, step=1, timestamp=mock.ANY),
        )
        self.assertEqual(
            mock_record.call_args_list[1],
            mock.call("critic/train/loss", 0.5, step=1, timestamp=mock.ANY),
        )
        self.assertEqual(
            mock_record.call_args_list[2],
            mock.call("rewards/score", 1.0, step=1, timestamp=mock.ANY),
        )

  def test_log_scalar_unmapped_metric_passes_full_event_name(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("custom/reward/score", 1.0, step=5)
        mock_record.assert_called_once_with(
            "custom/reward/score", 1.0, step=5, timestamp=mock.ANY
        )

  def test_log_scalar_step_zero_preserved(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", 0.5, step=0)
        mock_record.assert_called_once_with(
            "actor/train/loss", 0.5, step=0, timestamp=mock.ANY
        )

  def test_log_scalar_exception_safe(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(
          _RECORD_TARGET,
          side_effect=RuntimeError("RPC failure"),
      ):
        # Should not raise exception
        backend.log_scalar("actor/train/loss", 0.5, step=1)

  def test_log_scalar_passes_timestamp_kwarg(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar(
            "custom/reward/score",
            1.0,
            step=5,
            timestamp="2026-08-30T12:00:00Z",
        )
        mock_record.assert_called_once_with(
            "custom/reward/score",
            1.0,
            step=5,
            timestamp="2026-08-30T12:00:00Z",
        )

  def test_log_scalar_default_timestamp_utc(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    before = datetime.datetime.now(datetime.timezone.utc)
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", 0.5, step=1)
        mock_record.assert_called_once()
        _, kwargs = mock_record.call_args
        timestamp = kwargs.get("timestamp")
        self.assertIsInstance(timestamp, datetime.datetime)
        self.assertEqual(timestamp.tzinfo, datetime.timezone.utc)
        after = datetime.datetime.now(datetime.timezone.utc)
        self.assertLessEqual(before, timestamp)
        self.assertLessEqual(timestamp, after)

  def test_log_scalar_explicit_datetime_timestamp(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    dt = datetime.datetime(2026, 8, 30, 12, 0, 0, tzinfo=datetime.timezone.utc)
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar(
            "custom/reward/score",
            1.0,
            step=5,
            timestamp=dt,
        )
        mock_record.assert_called_once_with(
            "custom/reward/score",
            1.0,
            step=5,
            timestamp=dt,
        )

  def test_log_scalar_none_step_default_timestamp(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", 0.5, step=None)
        mock_record.assert_called_once_with(
            "actor/train/loss", 0.5, step=None, timestamp=mock.ANY
        )
        _, kwargs = mock_record.call_args
        timestamp = kwargs.get("timestamp")
        self.assertIsInstance(timestamp, datetime.datetime)
        self.assertEqual(timestamp.tzinfo, datetime.timezone.utc)

  def test_log_scalar_explicit_none_timestamp_defaults_to_utc(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    before = datetime.datetime.now(datetime.timezone.utc)
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/train/loss", 0.5, step=1, timestamp=None)
        mock_record.assert_called_once()
        _, kwargs = mock_record.call_args
        timestamp = kwargs.get("timestamp")
        self.assertIsInstance(timestamp, datetime.datetime)
        self.assertEqual(timestamp.tzinfo, datetime.timezone.utc)
        after = datetime.datetime.now(datetime.timezone.utc)
        self.assertLessEqual(before, timestamp)
        self.assertLessEqual(timestamp, after)

  def test_log_scalar_explicit_numeric_timestamp(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar(
            "custom/reward/score",
            1.0,
            step=5,
            timestamp=1700000000.0,
        )
        mock_record.assert_called_once_with(
            "custom/reward/score",
            1.0,
            step=5,
            timestamp=1700000000.0,
        )

  def test_close_callable(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    # Should execute cleanly without error
    backend.close()

  def test_init_with_config_calls_managed_mldiagnostics(self):
    mock_config = mock.MagicMock()
    with mock.patch.object(
        mldiag_backend.managed_mldiagnostics,
        "ManagedMLDiagnostics",
    ) as mock_managed:
      mldiag_backend.MLDiagScalarBackend(mock_config)
      mock_managed.assert_called_once_with(mock_config)

    mock_managed.reset_mock()
    with mock.patch.object(
        mldiag_backend.managed_mldiagnostics,
        "ManagedMLDiagnostics",
    ) as mock_managed:
      mldiag_backend.MLDiagScalarBackend(config=mock_config)
      mock_managed.assert_called_once_with(mock_config)

  def test_init_without_config_does_not_call_managed_mldiagnostics(self):
    with mock.patch.object(
        mldiag_backend.managed_mldiagnostics,
        "ManagedMLDiagnostics",
    ) as mock_managed:
      mldiag_backend.MLDiagScalarBackend()
      mldiag_backend.MLDiagScalarBackend(None)
      mock_managed.assert_not_called()

  def test_normalize_metric_event(self):
    self.assertEqual(
        mldiag_backend._normalize_metric_event("actor/Mode.TRAIN/loss"),
        "actor/train/loss",
    )
    self.assertEqual(
        mldiag_backend._normalize_metric_event("rewards/Mode.EVAL/score/mean"),
        "rewards/eval/score/mean",
    )
    self.assertEqual(
        mldiag_backend._normalize_metric_event("global/Mode.Train/throughput"),
        "global/train/throughput",
    )
    self.assertEqual(
        mldiag_backend._normalize_metric_event("Mode.Eval/latency"),
        "eval/latency",
    )
    self.assertEqual(
        mldiag_backend._normalize_metric_event("actor/train/loss"),
        "actor/train/loss",
    )
    self.assertEqual(
        mldiag_backend._normalize_metric_event("loss"),
        "loss",
    )
    self.assertEqual(
        mldiag_backend._normalize_metric_event(""),
        "",
    )

  def test_log_scalar_normalizes_mode_enum_in_event_name(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        backend.log_scalar("actor/Mode.TRAIN/loss", 0.5, step=10)
        backend.log_scalar("rewards/Mode.EVAL/score/mean", 1.5, step=10)
        backend.log_scalar("global/Mode.Train/throughput", 100.0, step=10)
        backend.log_scalar("Mode.Eval/latency", 20.0, step=10)

        self.assertEqual(mock_record.call_count, 4)
        self.assertEqual(
            mock_record.call_args_list[0],
            mock.call("actor/train/loss", 0.5, step=10, timestamp=mock.ANY),
        )
        self.assertEqual(
            mock_record.call_args_list[1],
            mock.call(
                "rewards/eval/score/mean", 1.5, step=10, timestamp=mock.ANY
            ),
        )
        self.assertEqual(
            mock_record.call_args_list[2],
            mock.call(
                "global/train/throughput", 100.0, step=10, timestamp=mock.ANY
            ),
        )
        self.assertEqual(
            mock_record.call_args_list[3],
            mock.call("eval/latency", 20.0, step=10, timestamp=mock.ANY),
        )

  def test_log_scalar_fallback_when_timestamp_unsupported(self):
    backend = mldiag_backend.MLDiagScalarBackend()
    self.assertTrue(backend._supports_timestamp)
    with mock.patch("jax.process_index", return_value=0):
      with mock.patch(_RECORD_TARGET) as mock_record:
        def side_effect(*args, **kwargs):
          if "timestamp" in kwargs:
            raise TypeError("record() got an unexpected keyword argument 'timestamp'")
          return None

        mock_record.side_effect = side_effect

        # First call catches TypeError, toggles _supports_timestamp, and retries without timestamp
        backend.log_scalar("actor/train/loss", 0.5, step=1)
        self.assertFalse(backend._supports_timestamp)
        self.assertEqual(mock_record.call_count, 2)
        self.assertNotIn("timestamp", mock_record.call_args[1])
        self.assertEqual(mock_record.call_args[1]["step"], 1)

        # Subsequent call immediately bypasses timestamp
        mock_record.reset_mock()
        backend.log_scalar("actor/train/loss", 0.25, step=2)
        self.assertEqual(mock_record.call_count, 1)
        self.assertNotIn("timestamp", mock_record.call_args[1])
        self.assertEqual(mock_record.call_args[1]["step"], 2)


if __name__ == "__main__":
  try:
    from absl.testing import absltest
    absltest.main()
  except ImportError:
    unittest.main()
