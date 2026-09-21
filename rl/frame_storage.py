"""Lossless in-memory interning of visible frames, with bounded ring references.

Every observation still stores its four exact frame identities. No assumption
about temporal adjacency, worker order, n-step distance or reset history is
needed. Python's byte-key equality resolves hash collisions exactly. This
storage consumes no RNG and never reads emulator/native state.
"""

import operator

import numpy as np


class FramePool:
    def __init__(self):
        self.lookup = {}
        self.frames, self.references, self.free = [], [], []

    def acquire(self, frame):
        index = self.lookup.get(frame)
        if index is not None:
            self.references[index] += 1
            return index
        if self.free:
            index = self.free.pop()
            self.frames[index], self.references[index] = frame, 1
        else:
            index = len(self.frames)
            self.frames.append(frame)
            self.references.append(1)
        self.lookup[frame] = index
        return index

    def release(self, index):
        if self.references[index] < 1:
            raise RuntimeError("releasing an unreferenced frame")
        self.references[index] -= 1
        if not self.references[index]:
            del self.lookup[self.frames[index]]
            self.frames[index] = None
            self.free.append(index)


class FrameTable:
    """Whole-stack assignment and indexing interface used by Replay."""

    def __init__(self, capacity, pool):
        self.pool = pool
        self.indices = np.full((capacity, 4), -1, np.int32)
        self.shape = (capacity, 4, 16, 64)
        self.dtype = np.dtype(np.uint8)

    def __len__(self):
        return len(self.indices)

    def __setitem__(self, index, observation):
        index = operator.index(index)
        old = self.indices[index].copy()  # Validate the index before changing references.
        observation = np.asarray(observation)
        if observation.shape != (4, 16, 64) or observation.dtype != np.uint8:
            raise ValueError("frame storage requires a uint8 [4,16,64] screen stack")
        acquired = []
        try:
            for frame in observation:
                acquired.append(self.pool.acquire(frame.tobytes()))
        except BaseException:
            for item in acquired:
                self.pool.release(item)
            raise
        self.indices[index] = acquired
        # Acquire first: identical frames survive even when this was their
        # final old ring reference, and repeated boot frames count correctly.
        for item in old:
            if item >= 0:
                self.pool.release(int(item))

    def __getitem__(self, selector):
        indices = self.indices[selector]
        if not indices.ndim or indices.shape[-1] != 4:
            raise IndexError("select complete screen stacks")
        if np.any(indices < 0):
            raise IndexError("uninitialized replay screen slot")
        raw = b"".join(self.pool.frames[int(i)] for i in indices.reshape(-1))
        # Independent writable arrays match dense replay indexing; neither a
        # caller modifying a sample nor a later ring overwrite can alias them.
        return np.frombuffer(raw, np.uint8).reshape(*indices.shape, 16, 64).copy()


class FrameStorage:
    def __init__(self, capacity):
        if isinstance(capacity, (bool, np.bool_)):
            raise ValueError("capacity must be a positive integer")
        capacity = operator.index(capacity)
        if not 1 <= capacity <= (np.iinfo(np.int32).max-8)//8:
            raise ValueError("capacity exceeds bounded int32 frame IDs")
        self.capacity = capacity
        self.pool = FramePool()
        self.obs = FrameTable(capacity, self.pool)
        self.next_obs = FrameTable(capacity, self.pool)

    def stats(self):
        return dict(unique_frames=len(self.pool.lookup), frame_slots=len(self.pool.frames),
                    screen_payload_bytes=len(self.pool.lookup)*16*64,
                    screen_index_bytes=self.obs.indices.nbytes+self.next_obs.indices.nbytes,
                    dense_equivalent_screen_bytes=self.capacity*8*16*64)
