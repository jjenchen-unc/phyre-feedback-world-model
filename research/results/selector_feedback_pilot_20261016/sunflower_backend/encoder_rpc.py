"""Authenticated array-only TCP RPC; no NFS polling and no pickle deserialization."""
import argparse
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import secrets
import socket
import socketserver
import struct
import sys
import threading
import time
import zlib

import numpy as np

B = Path(__file__).resolve().parent
R = B.parent
MAX_BYTES = 128 * 2**20
TIMEOUT = 1800


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def pack(array):
    stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
    assert stream.tell() <= MAX_BYTES
    return zlib.compress(stream.getvalue(), level=1)


def unpack(payload):
    dec = zlib.decompressobj()
    raw = dec.decompress(payload, MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or dec.unconsumed_tail or dec.unused_data or not dec.eof:
        raise ValueError('Invalid or oversized array payload')
    value = np.load(io.BytesIO(raw), allow_pickle=False)
    if value.dtype.hasobject:
        raise ValueError('Object arrays are forbidden')
    return value


def receive_exact(sock, n):
    parts = []; left = n
    while left:
        part = sock.recv(min(left, 1024 * 1024))
        if not part:
            raise ConnectionError('Encoder RPC connection closed mid-message')
        parts.append(part); left -= len(part)
    return b''.join(parts)


def send(sock, header, payload, key):
    header = {**header, 'payload_sha256': hashlib.sha256(payload).hexdigest()}
    data = json.dumps(header, sort_keys=True, separators=(',', ':')).encode()
    signature = hmac.digest(key, data, 'sha256')
    assert len(data) <= 65536 and len(payload) <= MAX_BYTES
    sock.sendall(struct.pack('!IQ', len(data), len(payload)) + signature + data + payload)


def receive(sock, key):
    nh, nb = struct.unpack('!IQ', receive_exact(sock, 12))
    if nh > 65536 or nb > MAX_BYTES:
        raise ValueError('Oversized RPC message')
    signature = receive_exact(sock, 32); data = receive_exact(sock, nh)
    if not hmac.compare_digest(signature, hmac.digest(key, data, 'sha256')):
        raise ValueError('Encoder RPC authentication failed')
    header = json.loads(data); payload = receive_exact(sock, nb)
    if not hmac.compare_digest(header['payload_sha256'], hashlib.sha256(payload).hexdigest()):
        raise ValueError('Encoder RPC checksum mismatch')
    return header, payload


class Client:
    def __init__(self, endpoint=None):
        path = Path(endpoint or os.environ.get('PHYRE_ENCODER_ENDPOINT', B / 'rpc/ENDPOINT.json'))
        self.endpoint = json.loads(path.read_text())
        self.key = Path(self.endpoint['auth_file']).read_bytes()
        self.plan = sha(R / 'PLAN.json'); self.code = sha(__file__)
        assert self.endpoint['plan'] == self.plan and self.endpoint['code'] == self.code

    def request(self, method, frames):
        assert method in ['initial', 'rollout', 'scene_measure']
        assert frames.dtype == np.uint8 and frames.ndim == 3 and frames.shape[1:] == (256, 256) and 0 < len(frames) <= 1024
        if method == 'rollout': assert len(frames) == 128
        if method == 'scene_measure': assert len(frames) == 1
        nonce = secrets.token_hex(16); start = time.perf_counter()
        payload = pack(frames)
        with socket.create_connection((self.endpoint['host'], self.endpoint['port']), timeout=30) as sock:
            sock.settimeout(TIMEOUT)
            send(sock, {'method': method, 'nonce': nonce, 'plan': self.plan, 'code': self.code}, payload, self.key)
            answer, payload = receive(sock, self.key)
        if not answer.get('ok'):
            raise RuntimeError(answer.get('error', 'Encoder RPC failed'))
        assert answer['nonce'] == nonce and answer['plan'] == self.plan and answer['code'] == self.code
        result = unpack(payload)
        assert np.isfinite(result).all()
        if method == 'rollout': assert result.shape == (32, 196, 128) and result.dtype == np.float16
        if method == 'initial': assert result.shape == (len(frames), 196, 128) and result.dtype == np.float16
        answer['client_s'] = time.perf_counter() - start
        answer['transport_and_codec_s'] = answer['client_s'] - answer['encode_s'] - answer['queue_s']
        return result, answer


_client = None
def request(method, frames):
    global _client
    if _client is None: _client = Client()
    return _client.request(method, frames)


def serve(host, port, advertised_host, lifetime):
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    os.environ.pop('DISABLE_ADDMM_CUDA_LT', None)
    import torch
    sys.path.insert(0, str(R / 'code'))
    import selector_feedback_data as pools
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 1 and 'A6000' in torch.cuda.get_device_name(0)
    pools.data.checked()
    encoder = pools.data.runtime().m.Encoder(pools.data.runtime().m.PCA)
    plan = sha(R / 'PLAN.json'); code = sha(__file__)
    folder = B / 'rpc'; folder.mkdir(exist_ok=True)
    auth = folder / 'auth.key'
    if not auth.exists():
        fd = os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(secrets.token_bytes(32))
    assert auth.stat().st_mode & 0o077 == 0
    key = auth.read_bytes(); lock = threading.Lock(); count = 0

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            nonlocal count
            self.request.settimeout(TIMEOUT)
            spec = None
            try:
                spec, payload = receive(self.request, key)
                assert spec['plan'] == plan and spec['code'] == code
                method = spec['method']; assert method in ['initial', 'rollout', 'scene_measure']
                frames = unpack(payload)
                assert frames.dtype == np.uint8 and frames.ndim == 3 and frames.shape[1:] == (256, 256) and 0 < len(frames) <= 1024
                if method == 'rollout': assert len(frames) == 128
                if method == 'scene_measure': assert len(frames) == 1
                before = time.perf_counter()
                with lock:
                    queue_s = time.perf_counter() - before
                    torch.cuda.synchronize(); start = time.perf_counter()
                    if method == 'scene_measure':
                        with torch.no_grad():
                            pixels = encoder.pixels(frames)
                            encoder.model(pixel_values=torch.stack([pixels, pixels], 2)).last_hidden_state
                        result = np.array(0)
                    else:
                        result = getattr(encoder, method)(frames)
                    torch.cuda.synchronize(); encode_s = time.perf_counter() - start
                    count += 1
                payload = pack(result)
                send(self.request, {'ok': True, 'nonce': spec['nonce'], 'plan': plan, 'code': code,
                                   'encode_s': encode_s, 'queue_s': queue_s}, payload, key)
                write(folder / 'STATUS.json', {'stage': 'serving', 'pid': os.getpid(), 'completed': count,
                      'time': time.time(), 'last_encode_s': encode_s, 'last_queue_s': queue_s})
            except Exception as exc:
                # Only array/JSON messages are accepted; never execute client-specified paths/code.
                if spec is not None:
                    try: send(self.request, {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}, b'', key)
                    except OSError: pass
                print('RPC_ERROR', type(exc).__name__, str(exc), flush=True)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with Server((host, port), Handler) as server:
        server.timeout = 1
        write(folder / 'ENDPOINT.json', {'host': advertised_host, 'port': server.server_address[1],
              'auth_file': str(auth), 'plan': plan, 'code': code, 'pid': os.getpid(),
              'gpu': str(torch.cuda.get_device_properties(0).uuid), 'max_runtime_s': lifetime})
        print('ENCODER_READY', server.server_address[1], flush=True)
        start = time.monotonic()
        while time.monotonic() - start < lifetime and not (folder / 'STOP').exists():
            server.handle_request()
    write(folder / 'STATUS.json', {'stage': 'stopped', 'pid': os.getpid(), 'completed': count, 'time': time.time()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--host', default='0.0.0.0'); ap.add_argument('--port', type=int, default=0)
    ap.add_argument('--advertised-host', default='152.2.134.113'); ap.add_argument('--lifetime', type=int, default=4*3600)
    a = ap.parse_args(); serve(a.host, a.port, a.advertised_host, a.lifetime)
