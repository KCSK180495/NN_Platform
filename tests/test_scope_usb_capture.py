"""Exercise USB discovery, timeout recovery and waveform transfer through the deployment worker."""

import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from nn_training_studio.scope_runtime.csv_capture import playback_capture
from nn_training_studio.scope_runtime.scope_types import (
    AcquisitionCancelled, AcquisitionTimeout, CaptureSettingsError, Channel,
)
from nn_training_studio.scope_runtime.session import ScopeRecording, ScopeWorker


class RigolUsbCaptureTests(unittest.TestCase):
    address = "USB0::0x1AB1::0x044D::TEST::INSTR"
    identity = "RIGOL TECHNOLOGIES,DHO814,TEST,00.01.04"
    channels = [Channel(1, "U"), Channel(2, "V")]

    def setUp(self):
        self.points = 11
        self.start, self.end = 1, self.points
        self.state = "STOP"
        self.source = 1
        self.instrument = Mock()
        self.instrument.write.side_effect = self.write
        self.instrument.query.side_effect = self.query
        self.instrument.query_ascii_values.side_effect = self.read_waveform
        self.manager = Mock()
        self.manager.open_resource.return_value = self.instrument
        self.manager.list_resources.return_value = (self.address,)
        visa = Mock()
        visa.ResourceManager.return_value = self.manager
        modules = patch.dict(sys.modules, pyvisa=visa)
        modules.start()
        self.addCleanup(modules.stop)
        self.worker = ScopeWorker()
        self.addCleanup(self.worker.executor.shutdown)
        self.addCleanup(self.worker.close)
        self.worker.submit(self.worker.connect, self.address, "@ivi").result()

    def write(self, command):
        if command == ":SING":
            self.state = "WAIT"
        elif command in (":STOP", ":TFOR"):
            self.state = "STOP"
        elif command.startswith(":WAV:SOUR CHAN"):
            self.source = int(command[-1])
        elif command.startswith(":WAV:STAR "):
            self.start = int(command.split()[-1])
            self.assertLessEqual(self.start, self.end)
        elif command.startswith(":WAV:STOP "):
            self.end = int(command.split()[-1])
            self.assertGreaterEqual(self.end, self.start)

    def query(self, command):
        return {
            "*IDN?": self.identity,
            "*OPC?": "0",  # Observed on USB even after a confirmed STOP.
            ":SYST:ERR?": '0,"No error"',
            ":TRIG:MODE?": "EDGE",
            ":TRIG:EDGE:SOUR?": "CHAN1",
            ":TRIG:EDGE:LEV?": "0",
            ":TRIG:STAT?": self.state,
            ":ACQ:MDEP?": str(self.points),
            ":WAV:PRE?": f"2,2,{self.end-self.start+1},1,2e-6,0,0,.01,0,0",
            ":CHAN1:OFFS?": "0",
            ":CHAN2:OFFS?": "0",
        }[command]

    def read_waveform(self, command, container):
        self.assertEqual(command, ":WAV:DATA?")
        return np.arange(self.start, self.end + 1) * self.source

    def capture(self, **options):
        return self.worker.submit(self.worker.capture, self.channels, options, [], False).result()

    def test_rescan_timeout_and_retry_on_same_session(self):
        self.assertEqual(self.worker.submit(self.worker.scan, "@ivi").result(), [self.address])
        self.manager.close.assert_not_called()
        with self.assertRaisesRegex(AcquisitionTimeout, "CHAN1"):
            self.capture(trigger_timeout=.01)
        self.instrument.close.assert_not_called()
        capture, result = self.capture(force_trigger=True, block_points=4)
        samples = np.arange(1, self.points + 1)
        np.testing.assert_array_equal(capture.scope_values, np.column_stack((samples, 2 * samples)))
        self.assertTrue(capture.metadata["full_record_transferred"])
        self.assertTrue(capture.metadata["force_trigger_command_sent"])
        self.assertEqual(result.status, "Acquisition only")
        self.manager.open_resource.assert_called_once_with(self.address)
        self.worker.submit(self.worker.disconnect).result()
        self.manager.close.assert_called_once()

    def test_correct_record_limit_without_reconnecting(self):
        with self.assertRaisesRegex(CaptureSettingsError, "record depth"):
            self.capture(force_trigger=True, max_points=10)
        self.instrument.query_ascii_values.assert_not_called()
        capture, _ = self.capture(force_trigger=True)
        self.assertEqual(len(capture.time_s), self.points)
        self.manager.open_resource.assert_called_once()

    def test_partial_transfer_discards_record_and_closes_session(self):
        self.instrument.query_ascii_values.side_effect = [np.arange(4), TimeoutError("USB timeout")]
        with self.assertRaisesRegex(ConnectionError, "CH1 waveform read failed at samples 5.*8"):
            self.capture(force_trigger=True, block_points=4)
        self.assertIsNone(self.worker.scope)
        self.instrument.close.assert_called_once()

    def test_failed_identity_check_closes_session_on_trigger_timeout(self):
        self.identity = "RIGOL TECHNOLOGIES,DHO814,OTHER,00.01.04"
        with self.assertRaisesRegex(ConnectionError, "identity changed"):
            self.capture(trigger_timeout=.01)
        self.assertIsNone(self.worker.scope)
        self.instrument.query_ascii_values.assert_not_called()

    def test_stop_during_waveform_transfer_allows_another_capture(self):
        def stop_after_first_block(command, container):
            self.worker.cancel.set()
            return self.read_waveform(command, container)

        self.instrument.query_ascii_values.side_effect = stop_after_first_block
        with self.assertRaises(AcquisitionCancelled):
            self.capture(force_trigger=True, block_points=4)
        self.assertEqual(self.state, "STOP")
        self.instrument.close.assert_not_called()
        self.worker.cancel.clear()
        self.instrument.query_ascii_values.side_effect = self.read_waveform
        capture, _ = self.capture(force_trigger=True, block_points=4)
        self.assertTrue(capture.metadata["full_record_transferred"])

    def live(self, recording, **settings):
        return self.worker.submit(self.worker.capture_live, self.channels, settings, recording).result()

    def test_live_view_then_record_across_a_trigger_timeout(self):
        with TemporaryDirectory() as parent:
            preview, _, error = self.live(None, force_trigger=True)
            self.assertIsNone(error)
            recording = ScopeRecording(parent)
            first, _, _ = self.live(recording, force_trigger=True)
            with self.assertRaises(AcquisitionTimeout):
                self.live(recording, trigger_timeout=.01)
            second, _, _ = self.live(recording, force_trigger=True)
            self.assertEqual(recording.saved, 2)
            files = sorted(recording.folder.glob("capture_*/native_samples.csv"))
            self.assertEqual(len(files), 2)
            for sequence, (path, original) in enumerate(zip(files, (first, second)), 1):
                saved = playback_capture(path, pd.read_csv(path, float_precision="round_trip"))
                self.assertEqual(saved.capture_id, original.capture_id)
                self.assertNotEqual(saved.capture_id, preview.capture_id)
                self.assertEqual(saved.metadata["recording_sequence"], sequence)
                self.assertEqual(saved.metadata["recording_session"], recording.folder.name)
                np.testing.assert_array_equal(saved.values, original.values)

    def test_recording_error_keeps_waveform_and_prior_capture(self):
        with TemporaryDirectory() as parent:
            recording = ScopeRecording(parent)
            self.live(recording, force_trigger=True)
            with patch("nn_training_studio.scope_runtime.session.save_capture", side_effect=OSError("disk full")):
                capture, result, error = self.live(recording, force_trigger=True)
            self.assertIn("disk full", error)
            self.assertEqual(capture.values.shape, (self.points, 2))
            self.assertEqual(result.status, "Acquisition only")
            self.assertEqual(recording.saved, 1)
            self.assertEqual(len(list(recording.folder.glob("*/native_samples.csv"))), 1)
            self.instrument.close.assert_not_called()
