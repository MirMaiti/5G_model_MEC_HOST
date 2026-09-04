"""The wire format: framing, packing, and the failures that must be loud."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from signbridge.protocol import (
    HEADER_SIZE,
    MAGIC,
    MAX_PAYLOAD,
    Connection,
    MessageType,
    ProtocolError,
    decode_frame,
    decode_json,
    encode_frame,
    encode_json,
    encode_message,
)
from tests.conftest import make_frame


def test_frame_round_trip_preserves_coordinates(rng):
    landmarks, mask = make_frame(rng)
    encoded = encode_frame(42, 1234.5, landmarks, mask)
    seq, t_ms, decoded, decoded_mask = decode_frame(encoded[HEADER_SIZE:])

    assert seq == 42
    assert t_ms == pytest.approx(1234.5)
    # float32 on the wire, so exact equality is not the right assertion.
    assert np.allclose(decoded, landmarks, atol=1e-6)
    assert np.array_equal(decoded_mask, mask)


def test_frame_is_far_smaller_than_the_json_it_replaces(rng):
    """The packed format is the reason landmarks beat images on the wire."""
    import json

    landmarks, mask = make_frame(rng)
    packed = len(encode_frame(0, 0.0, landmarks, mask))
    as_json = len(json.dumps({"landmarks": landmarks.tolist(), "mask": mask.tolist()}))

    assert packed == 568
    assert packed < as_json / 3


def test_json_round_trip():
    payload = encode_json(MessageType.WELCOME, {"labels": ["a", "b"], "window": 45})
    assert decode_json(payload[HEADER_SIZE:]) == {"labels": ["a", "b"], "window": 45}


def test_reassembles_a_message_split_across_segments(rng):
    """TCP is a stream: a message can arrive in any number of pieces."""
    left, right = socket.socketpair()
    sender, receiver = Connection(left), Connection(right)
    landmarks, mask = make_frame(rng)

    blob = encode_json(MessageType.HELLO, {"layout": "hands"}) + encode_frame(1, 2.0, landmarks, mask)
    for start in range(0, len(blob), 7):  # deliberately awkward chunking
        left.sendall(blob[start : start + 7])

    first_type, first_payload = receiver.read_message()
    second_type, second_payload = receiver.read_message()

    assert first_type is MessageType.HELLO
    assert decode_json(first_payload) == {"layout": "hands"}
    assert second_type is MessageType.FRAME
    assert decode_frame(second_payload)[0] == 1
    sender.close()
    receiver.close()


def test_clean_close_reads_as_none():
    left, right = socket.socketpair()
    sender, receiver = Connection(left), Connection(right)
    sender.close()
    assert receiver.read_message() is None
    receiver.close()


def test_rejects_foreign_magic():
    left, right = socket.socketpair()
    receiver = Connection(right)
    left.sendall(b"XX" + bytes([1, 1]) + (0).to_bytes(4, "big"))
    with pytest.raises(ProtocolError, match="Bad magic"):
        receiver.read_message()
    left.close()
    receiver.close()


def test_rejects_unknown_message_type():
    left, right = socket.socketpair()
    receiver = Connection(right)
    left.sendall(MAGIC + bytes([1, 200]) + (0).to_bytes(4, "big"))
    with pytest.raises(ProtocolError, match="Unknown message type"):
        receiver.read_message()
    left.close()
    receiver.close()


def test_rejects_version_mismatch():
    left, right = socket.socketpair()
    receiver = Connection(right)
    left.sendall(MAGIC + bytes([99, 1]) + (0).to_bytes(4, "big"))
    with pytest.raises(ProtocolError, match="protocol version"):
        receiver.read_message()
    left.close()
    receiver.close()


def test_truncated_payload_raises_rather_than_hanging(rng):
    left, right = socket.socketpair()
    receiver = Connection(right)
    landmarks, mask = make_frame(rng)
    encoded = encode_frame(1, 0.0, landmarks, mask)
    left.sendall(encoded[:-20])  # header promises more than is sent
    left.close()
    with pytest.raises(ProtocolError, match="closed mid-message"):
        receiver.read_message()
    receiver.close()


def test_oversized_payload_is_refused():
    with pytest.raises(ProtocolError, match="exceeds"):
        encode_message(MessageType.FRAME, b"x" * (MAX_PAYLOAD + 1))


def test_mismatched_mask_is_rejected(rng):
    landmarks, _ = make_frame(rng)
    with pytest.raises(ProtocolError, match="does not match"):
        encode_frame(0, 0.0, landmarks, np.ones(7))


def test_frame_claiming_more_landmarks_than_it_carries_is_rejected(rng):
    landmarks, mask = make_frame(rng)
    payload = bytearray(encode_frame(0, 0.0, landmarks, mask)[HEADER_SIZE:])
    payload[12:14] = (99).to_bytes(2, "big")  # lie about the landmark count
    with pytest.raises(ProtocolError, match="claims 99 landmarks"):
        decode_frame(bytes(payload))
