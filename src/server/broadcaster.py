"""Queue-backed broadcast for the map pipeline.

Wraps the thread-safe message queue with two emit shapes:

- ``emit``: single payload (text stats/ack or one binary frame). On overflow,
  binary frames are coalesced (drain queued binaries, keep text) so the server
  never lags behind a slow dashboard.
- ``emit_frame``: a whole logical frame (multi-chunk snapshot/delta). Delivered
  atomically; if it cannot fit it is dropped as a unit (never a truncated frame)
  and the caller is told so it can preserve the delta state for retransmit.
"""

import queue

from src.server import ws_protocol


class Broadcaster:
    def __init__(self, maxsize: int = 128, wire_compress: bool = False):
        self.messages: queue.Queue = queue.Queue(maxsize=maxsize)
        self._wire_compress = wire_compress
        self.frames_emitted = 0
        self.frames_dropped = 0

    @property
    def qsize(self) -> int:
        return self.messages.qsize()

    def clear(self):
        while True:
            try:
                self.messages.get_nowait()
            except queue.Empty:
                break

    def emit(self, payload, kind: str = "text"):
        """Queue a single broadcast payload. kind='text' -> JSON dict, 'binary' -> bytes frame."""
        if kind != "text" and self._wire_compress:
            payload = ws_protocol._maybe_compress(payload, enabled=True)
        item = ("T", payload) if kind == "text" else ("B", payload)
        is_binary = kind != "text"
        try:
            self.messages.put_nowait(item)
            self.frames_emitted += 1
        except queue.Full:
            if is_binary:
                kept: list = []
                while True:
                    try:
                        q = self.messages.get_nowait()
                        if q[0] == "T":
                            kept.append(q)
                        else:
                            self.frames_dropped += 1
                    except queue.Empty:
                        break
                for k in kept:
                    try:
                        self.messages.put_nowait(k)
                    except queue.Full:
                        break
            else:
                self._drop_oldest()
                self.frames_dropped += 1
            try:
                self.messages.put_nowait(item)
                self.frames_emitted += 1
            except queue.Full:
                self.frames_dropped += 1

    def emit_frame(self, chunks, kind: str = "binary") -> bool:
        """Enqueue all chunks of one logical frame atomically.

        If the whole frame cannot fit, drop it as a unit (never deliver a
        truncated/partial frame) and drain queued binary frames to make room.
        Returns True if the frame was successfully sent, False if it was dropped
        entirely; the caller may then preserve state (upsert/freed rows) for a
        later merge instead of silently losing them.
        """
        is_binary = kind != "text"
        if is_binary and self._wire_compress:
            chunks = [ws_protocol._maybe_compress(c, enabled=True) for c in chunks]
        items = [("B", c) for c in chunks] if is_binary else [("T", c) for c in chunks]
        q = self.messages
        n = len(items)
        if q.qsize() + n > q.maxsize:
            kept: list = []
            while True:
                try:
                    it = q.get_nowait()
                    if it[0] == "T":
                        kept.append(it)
                    else:
                        self.frames_dropped += 1
                except queue.Empty:
                    break
            for k in kept:
                try:
                    q.put_nowait(k)
                except queue.Full:
                    self.frames_dropped += 1
            return False
        for it in items:
            q.put_nowait(it)
        self.frames_emitted += n
        return True

    def _drop_oldest(self):
        try:
            self.messages.get_nowait()
        except queue.Empty:
            pass