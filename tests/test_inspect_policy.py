import base64
import json
import struct
import unittest

import numpy as np

from rl.inspect_policy import decode_replay, frame_stacks, probability_metrics


def replay_html(payload, frames=3, actions=None):
    actions = [0, 1] if actions is None else actions
    return (f"const metadata = {json.dumps(dict(frames=frames, steps=len(actions)))};\n"
            f"const actions = {json.dumps(actions)};\n"
            f"const data = Uint8Array.from(atob('{base64.b64encode(payload).decode()}'), c => c.charCodeAt(0));\n")


class PolicyInspectionTests(unittest.TestCase):
    def test_decode_replay_preserves_deltas_and_unchanged_frames(self):
        payload = bytes([128])*1024 + struct.pack("<HHB", 1, 12, 255) + struct.pack("<H", 0)
        metadata, frames, actions = decode_replay(replay_html(payload))
        self.assertEqual(frames.shape, (3, 16, 64))
        self.assertEqual(frames[0, 0, 12], 128)
        self.assertEqual(frames[1, 0, 12], 255)
        np.testing.assert_array_equal(frames[1], frames[2])
        np.testing.assert_array_equal(actions, [0, 1])

    def test_decode_replay_rejects_invalid_payloads(self):
        for payload in (b"", bytes(1024)+b"\x01", bytes(1024)+struct.pack("<HHB", 1, 1024, 1),
                        bytes(1024)+struct.pack("<H", 0)):
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                decode_replay(replay_html(payload))

    def test_frame_stack_alignment_and_initial_padding(self):
        frames = np.broadcast_to(np.arange(5)[:, None, None], (5, 16, 64))
        stacks = frame_stacks(frames, 0, 4)
        np.testing.assert_array_equal(stacks[:, :, 0, 0],
                                      [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 2, 3]])

    def test_serve_randomness_does_not_count_as_direction_randomness(self):
        result = probability_metrics(np.array([[.5, 0, 0, .5, 0, 0]]), np.array([3]))
        self.assertAlmostEqual(result["action_entropy_nats"], np.log(2))
        self.assertAlmostEqual(result["serve_entropy_nats"], np.log(2))
        self.assertEqual(result["direction_entropy_nats"], 0)
        self.assertEqual(result["mean_non_greedy_direction_probability"], 0)

    def test_uniform_action_entropy(self):
        result = probability_metrics(np.full((6, 6), 1/6), np.arange(6))
        self.assertAlmostEqual(result["action_entropy_nats"], np.log(6))
        self.assertAlmostEqual(result["direction_entropy_nats"], np.log(3))
        self.assertAlmostEqual(result["serve_entropy_nats"], np.log(2))
        self.assertAlmostEqual(result["mean_non_greedy_direction_probability"], 2/3)


if __name__ == "__main__":
    unittest.main()
