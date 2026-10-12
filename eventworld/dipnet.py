# -*- coding: utf-8 -*-
"""DipNet (Paquette et al., NeurIPS 2019) as a frozen, callable Diplomacy agent.

Public API
----------
    orders = dipnet_orders(state_dict, power, map_name='standard', rules=[])
    all_orders = dipnet_orders_batch(state_dict, powers=None, ...)   # {power: [orders]}

`state_dict` is the dict returned by `diplomacy.Game.get_state()` (keys used:
name, units, centers, homes, retreats, builds; extra keys are ignored).

Optional kwargs (both functions):
    model        'sl' (default, neurips2019-sl_model) or 'rl' (neurips2019-rl_model)
    temperature  softmax temperature (default 0.1, as in the paper's benchmark player);
                 0.0 = greedy
    seed         int player seed forwarded to the model (deterministic sampling)
    use_beam     use beam search instead of sampling (default False)
    prev_state, prev_orders
                 optional: the state dict of the *previous movement phase* and a
                 {power: [orders]} dict of the orders played there.  DipNet's input
                 includes the last movement phase's orders; without this the feature
                 is all-zeros (the model still plays fine, but it is not exactly what
                 it saw in training).

How it runs
-----------
* The network itself is served by TensorFlow Serving 1.13 (CPU) inside the
  Apptainer image containers/tfserving.sif; it is auto-started if port 9501 (sl)
  / 9502 (rl) is closed.
* Feature building / order decoding uses the authors' `diplomacy_research`
  package, which needs Python 3.7 + TF 1.13 + diplomacy==1.1.0 (env37/).
  - If this module is imported under env37, everything runs in-process.
  - Otherwise (e.g. from a Python 3.11 venv with diplomacy==1.1.2) it spawns
    one persistent env37 worker (`python eventworld/dipnet.py --worker`) and talks
    JSON lines to it over a pipe.  The worker is started lazily on first call
    and reused.

Legality: the model decodes over the engine's `get_all_possible_orders()`
candidates for each orderable location, so every emitted order is legal for
the given state (verified by adjudicating with the engine).
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time

# DIPNET_HOME holds env37/ (Python 3.7 + TF 1.13 + diplomacy_research), research/ (github.com/diplomacy/research), containers/tfserving.sif and
# models/{sl,rl} (see third_party/dipnet/README.md). Default: third_party/dipnet next to this package.
DIPNET_ROOT = os.path.abspath(os.environ.get('DIPNET_HOME', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'dipnet')))
ENV37_PYTHON = os.path.join(DIPNET_ROOT, 'env37', 'bin', 'python')
RESEARCH_DIR = os.path.join(DIPNET_ROOT, 'research')
SIF_PATH = os.path.join(DIPNET_ROOT, 'containers', 'tfserving.sif')
LOG_DIR = os.path.join(DIPNET_ROOT, 'logs')

MODELS = {
    'sl': {'port': 9501, 'dir': 'models/sl'},
    'rl': {'port': 9502, 'dir': 'models/rl'},
}
SERVING_THREADS = int(os.environ.get('DIPNET_SERVING_THREADS', '8'))
SERVING_START_TIMEOUT = 300  # seconds

__all__ = ['dipnet_orders', 'dipnet_orders_batch', 'ensure_serving', 'shutdown']


# --------------------------------------------------------------------------- #
# TF Serving management (works from any python)
# --------------------------------------------------------------------------- #
def is_port_open(port, host='127.0.0.1'):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def _container_runtime():
    for name in ('apptainer', 'singularity'):
        for path in os.environ.get('PATH', '').split(os.pathsep):
            cand = os.path.join(path, name)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    raise RuntimeError('Neither apptainer nor singularity found on PATH; cannot start TF Serving.')


def ensure_serving(model='sl', timeout=SERVING_START_TIMEOUT, verbose=True):
    """Start TF Serving for `model` in the background if its port is closed. Returns the port."""
    cfg = MODELS[model]
    port = cfg['port']
    if is_port_open(port):
        return port
    if not os.path.isfile(SIF_PATH):
        raise RuntimeError('TF Serving image not found: %s' % SIF_PATH)
    model_dir = os.path.join(DIPNET_ROOT, cfg['dir'])
    if not os.path.isdir(model_dir):
        raise RuntimeError('Model directory not found: %s' % model_dir)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, 'serving_%s_%d.log' % (model, port))
    cmd = [_container_runtime(), 'exec',
           '-B', '%s:/work_dir' % DIPNET_ROOT,
           SIF_PATH,
           'tensorflow_model_server',
           '--port=%d' % port,
           '--model_name=player',
           '--model_base_path=/work_dir/%s' % cfg['dir'],
           '--tensorflow_session_parallelism=%d' % SERVING_THREADS]
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''  # CPU only, as in the authors' benchmark player
    if verbose:
        sys.stderr.write('[dipnet] starting TF Serving (%s) on port %d, log: %s\n' % (model, port, log_path))
    with open(log_path, 'a') as log:
        subprocess.Popen(cmd, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                         env=env, start_new_session=True, cwd=DIPNET_ROOT)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_open(port):
            time.sleep(1.0)  # give the model a moment to finish loading
            return port
        time.sleep(1.0)
    raise RuntimeError('TF Serving did not open port %d within %ds; see %s' % (port, timeout, log_path))


# --------------------------------------------------------------------------- #
# Game construction (engine-version agnostic; used by the in-process backend)
# --------------------------------------------------------------------------- #
_STATE_KEYS = ('name', 'units', 'centers', 'homes', 'retreats', 'builds', 'influence', 'civil_disorder')


def _clean_state(state):
    """Keep only the keys `Game.set_state` understands (drops map/rules/timestamps etc.)."""
    return {k: state[k] for k in _STATE_KEYS if k in state}


def build_game(state, map_name='standard', rules=None, prev_state=None, prev_orders=None):
    """Build a `diplomacy.Game` positioned at `state` (optionally with one movement phase of history)."""
    from diplomacy import Game
    if not str(map_name).startswith('standard'):
        raise ValueError('DipNet was trained on the standard map only (got map_name=%r).' % map_name)
    kwargs = {'map_name': map_name}
    if rules:
        kwargs['rules'] = list(rules)
    game = Game(**kwargs)
    if prev_state is not None and prev_orders:
        game.set_state(_clean_state(prev_state))
        for power_name, orders in prev_orders.items():
            if orders:
                game.set_orders(power_name, list(orders))
        game.process()
        game.set_state(_clean_state(state), clear_history=False)
    else:
        game.set_state(_clean_state(state))
    return game


# --------------------------------------------------------------------------- #
# In-process backend (requires env37: py3.7 + TF1.13 + diplomacy_research)
# --------------------------------------------------------------------------- #
def _in_process_available():
    if sys.version_info[:2] != (3, 7):
        return False
    try:
        import diplomacy_research  # noqa: F401
        return True
    except Exception:  # pylint: disable=broad-except
        if RESEARCH_DIR not in sys.path:
            sys.path.insert(0, RESEARCH_DIR)
            try:
                import diplomacy_research  # noqa: F401
                return True
            except Exception:  # pylint: disable=broad-except
                return False
        return False


class _InProcessBackend(object):
    def __init__(self):
        os.environ.setdefault('WORKING_DIR', DIPNET_ROOT)
        self._players = {}
        from tornado import ioloop
        self._loop = ioloop.IOLoop.current()

    def _player(self, model, use_beam):
        """Build (once) the benchmark player. Must be called from inside the running IOLoop."""
        key = (model, bool(use_beam))
        if key not in self._players:
            ensure_serving(model)
            from diplomacy_research.players import benchmark_player
            cls = {'sl': benchmark_player.DipNetSLPlayer, 'rl': benchmark_player.DipNetRLPlayer}[model]
            self._players[key] = cls(temperature=0.1, use_beam=bool(use_beam), port=MODELS[model]['port'])
        return self._players[key]

    def orders(self, state, powers, map_name='standard', rules=None, model='sl', temperature=0.1,
               seed=None, use_beam=False, prev_state=None, prev_orders=None):
        from tornado import gen
        game = build_game(state, map_name, rules, prev_state, prev_orders)
        powers = list(powers) if powers else list(game.powers.keys())
        kwargs = {'temperature': float(temperature)}
        if seed is not None:
            kwargs['player_seed'] = int(seed)

        @gen.coroutine
        def _run():
            # The adapter requires a running IO loop at construction time
            player = self._player(model, use_beam)
            result = yield player.get_orders(game, powers, **kwargs)
            return result

        result = self._loop.run_sync(_run, timeout=300)
        return {p: list(o) for p, o in zip(powers, result)}


# --------------------------------------------------------------------------- #
# Subprocess backend (any python -> env37 worker)
# --------------------------------------------------------------------------- #
class _SubprocessBackend(object):
    def __init__(self):
        if not os.path.isfile(ENV37_PYTHON):
            raise RuntimeError('env37 python not found at %s' % ENV37_PYTHON)
        self._proc = None
        self._lock = threading.Lock()
        self._next_id = 0

    def _start(self):
        env = dict(os.environ)
        env['PYTHONPATH'] = RESEARCH_DIR + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        env['WORKING_DIR'] = DIPNET_ROOT
        env.setdefault('CUDA_VISIBLE_DEVICES', '')
        # Do not inherit the caller's venv settings
        for key in ('VIRTUAL_ENV', 'PYTHONHOME', 'PYTHONNOUSERSITE'):
            env.pop(key, None)
        os.makedirs(LOG_DIR, exist_ok=True)
        self._log = open(os.path.join(LOG_DIR, 'worker.log'), 'a')
        self._proc = subprocess.Popen([ENV37_PYTHON, '-u', os.path.abspath(__file__), '--worker'],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                      env=env, cwd=DIPNET_ROOT, universal_newlines=True, bufsize=1)
        ready = self._proc.stdout.readline()
        if not ready.startswith('{"ready"'):
            raise RuntimeError('DipNet worker failed to start (see %s/worker.log): %r' % (LOG_DIR, ready))

    def _request(self, payload):
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            self._next_id += 1
            payload['id'] = self._next_id
            self._proc.stdin.write(json.dumps(payload) + '\n')
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                self._proc = None
                raise RuntimeError('DipNet worker died (see %s/worker.log)' % LOG_DIR)
            resp = json.loads(line)
        if 'error' in resp:
            raise RuntimeError('DipNet worker error: %s' % resp['error'])
        return resp

    def orders(self, state, powers, **kwargs):
        resp = self._request({'cmd': 'orders', 'state': state, 'powers': list(powers) if powers else None,
                              'kwargs': kwargs})
        return resp['orders']

    def close(self):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(json.dumps({'cmd': 'quit'}) + '\n')
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=10)
                except Exception:  # pylint: disable=broad-except
                    self._proc.kill()
            self._proc = None


def _worker_main():
    """Entry point of the env37 worker: JSON lines in (stdin) -> JSON lines out (stdout)."""
    import logging
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    out = sys.stdout
    sys.stdout = sys.stderr  # anything the libraries print must not corrupt the protocol
    if not _in_process_available():
        out.write(json.dumps({'error': 'diplomacy_research not importable in worker'}) + '\n')
        out.flush()
        return
    backend = _InProcessBackend()
    out.write(json.dumps({'ready': True}) + '\n')
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get('cmd') == 'quit':
                break
            orders = backend.orders(req['state'], req.get('powers'), **req.get('kwargs', {}))
            resp = {'id': req.get('id'), 'orders': orders}
        except Exception as exc:  # pylint: disable=broad-except
            import traceback
            traceback.print_exc(file=sys.stderr)
            resp = {'id': req.get('id') if isinstance(req, dict) else None,
                    'error': '%s: %s' % (type(exc).__name__, exc)}
        out.write(json.dumps(resp) + '\n')
        out.flush()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
_BACKEND = None
_BACKEND_LOCK = threading.Lock()


def _backend():
    global _BACKEND  # pylint: disable=global-statement
    with _BACKEND_LOCK:
        if _BACKEND is None:
            _BACKEND = _InProcessBackend() if _in_process_available() else _SubprocessBackend()
        return _BACKEND


def dipnet_orders_batch(state_dict, powers=None, map_name='standard', rules=None, model='sl',
                        temperature=0.1, seed=None, use_beam=False, prev_state=None, prev_orders=None):
    """Orders for several powers from one state. Returns {power_name: [orders]}.
    `powers=None` means all powers of the map."""
    if model not in MODELS:
        raise ValueError('model must be one of %s' % sorted(MODELS))
    return _backend().orders(state_dict, powers, map_name=map_name, rules=list(rules or []), model=model,
                             temperature=temperature, seed=seed, use_beam=bool(use_beam),
                             prev_state=prev_state, prev_orders=prev_orders)


def dipnet_orders(state_dict, power, map_name='standard', rules=None, **kwargs):
    """Orders DipNet would submit for `power` in `state_dict` (a `Game.get_state()` dict).
    Returns a list of order strings, e.g. ['F BRE - MAO', 'A PAR - BUR', 'A MAR - SPA'].
    An empty list means the power has nothing to order this phase."""
    power = str(power).upper()
    return dipnet_orders_batch(state_dict, [power], map_name=map_name, rules=rules, **kwargs)[power]


def shutdown():
    """Stop the env37 worker (if any). TF Serving keeps running; kill it separately if needed."""
    global _BACKEND  # pylint: disable=global-statement
    with _BACKEND_LOCK:
        if isinstance(_BACKEND, _SubprocessBackend):
            _BACKEND.close()
        _BACKEND = None


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        _worker_main()
    elif len(sys.argv) > 1 and sys.argv[1] == '--start-serving':
        model_name = sys.argv[2] if len(sys.argv) > 2 else 'sl'
        print('TF Serving (%s) ready on port %d' % (model_name, ensure_serving(model_name)))
    else:
        # Smoke test: opening orders for all powers
        from diplomacy import Game
        _game = Game()
        _t0 = time.time()
        _orders = dipnet_orders_batch(_game.get_state())
        print('%.2fs' % (time.time() - _t0))
        for _p, _o in _orders.items():
            print(_p, _o)
        shutdown()
