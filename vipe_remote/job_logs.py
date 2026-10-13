"""Incremental UTF-8 log following and a bounded display buffer; no DCC imports."""
import base64
import codecs
import re
import threading

ANSI = re.compile(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')
CONTROLS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


class LogFollower:
    def __init__(self, client, identifier, emit):
        self.client = client
        self.identifier = identifier
        self.emit = emit
        self.offset = -1
        self.generation = None
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.warned = False
        self.disabled = False

    def _emit(self, text):
        try:
            self.emit(text)
        except Exception:
            # A display/terminal callback is optional; never fail the tracking job.
            self.disabled = True

    def poll(self):
        if self.disabled:
            return
        try:
            # Limit catch-up work so cancellation and status checks stay responsive.
            for _ in range(4):
                chunk = self.client.log_chunk(self.identifier, self.offset, self.generation)
                if not chunk['exists']:
                    return
                if chunk['reset'] or chunk['skipped_bytes']:
                    self.decoder.reset()
                    if chunk['reset']:
                        self._emit('\n[Worker log restarted; showing recent output.]\n')
                    if chunk['skipped_bytes']:
                        self._emit(f"\n[Live view omitted {chunk['skipped_bytes']} earlier bytes; the full log remains on the worker.]\n")
                raw = base64.b64decode(chunk['data'], validate=True)
                if len(raw) > 32768 or chunk['next_offset'] != chunk['offset'] + len(raw):
                    raise ValueError('Invalid worker log chunk.')
                self.offset, self.generation = chunk['next_offset'], chunk['generation']
                text = self.decoder.decode(raw)
                if text:
                    self._emit(text)
                self.warned = False
                if self.offset >= chunk['size']:
                    return
        except Exception as exc:
            if not self.warned:
                self._emit(f'\n[Live log unavailable: {exc}. Tracking continues.]\n')
                self.warned = True
            # Old workers return plain text instead of the cursor protocol.
            if isinstance(exc, (ValueError, KeyError, TypeError)) or getattr(exc, 'status', None) == 404:
                self.disabled = True

    def finish(self):
        self.poll()
        remaining = self.decoder.decode(b'', final=True)
        if remaining:
            self._emit(remaining)


class LogBuffer:
    """Network thread writes; Blender's main thread reads snapshots."""
    def __init__(self, limit=256 * 1024):
        self.limit = limit
        self.text = ''
        self.revision = 0
        self.trimmed = False
        self.lock = threading.Lock()
        self.trailing_cr = False

    def append(self, text):
        # Render tqdm's carriage-return updates as readable log lines.
        text = CONTROLS.sub('', ANSI.sub('', text))
        with self.lock:
            if self.trailing_cr and text.startswith('\n'):
                text = text[1:]
            self.trailing_cr = text.endswith('\r')
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            if not text:
                return ''
            combined = self.text + text
            if len(combined) > self.limit:
                self.trimmed = True
                combined = combined[-self.limit:]
            self.text = combined
            self.revision += 1
        return text

    def snapshot(self):
        with self.lock:
            prefix = '[Blender keeps the most recent log output; the full log remains on the worker.]\n' if self.trimmed else ''
            return self.revision, prefix + self.text
