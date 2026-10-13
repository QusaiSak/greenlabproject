import argparse
import csv
import json
import random
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_SIZES = {
    "small": "prompt_small.txt",
    "medium": "prompt_medium.txt",
    "large": "prompt_large.txt",
}

DEFAULT_RUNS = 10
DEFAULT_OUTPUT_TOKENS = 1536
DEFAULT_ORDER_SEED = 42
DEFAULT_AGENT_PORT = os.getenv("DEFAULT_AGENT_PORT")
DEFAULT_LLAMA_PORT = os.getenv("DEFAULT_LLAMA_PORT")
ENERGY_INTERVAL_MS = 200
WARMUP_STABILIZATION_SECONDS = 30
RUN_COOLDOWN_SECONDS = 10
DEFAULT_REMOTE_HOST = os.getenv("DEFAULT_REMOTE_HOST")
DEFAULT_REMOTE_MODEL_DIR = os.getenv("DEFAULT_REMOTE_MODEL_DIR")
DEFAULT_REMOTE_PROMPT_DIR = os.getenv("DEFAULT_REMOTE_PROMPT_DIR")
DEFAULT_REMOTE_RESULTS_DIR = os.getenv("DEFAULT_REMOTE_RESULTS_DIR")
DEFAULT_REMOTE_SCRIPT = os.getenv("DEFAULT_REMOTE_SCRIPT")
DEFAULT_AGENT_DELAY_SCRIPT = os.getenv("DEFAULT_AGENT_DELAY_SCRIPT", "./agent-delay")
DEFAULT_NETWORK_DELAY_IP = os.getenv("DEFAULT_NETWORK_DELAY_IP")
DEFAULT_NETWORK_DELAY_MS = 0


# ============================================================
# SHARED HTTP HELPERS
# ============================================================


def http_json(url, payload=None, timeout=900):
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_command(command, check=True):
    result = subprocess.run(
        [str(value) for value in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): "
            f"{' '.join(str(value) for value in command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )

    return result


def ssh_command(host, command, user=None):
    target = host or DEFAULT_REMOTE_HOST
    if user:
        target = f"{user}@{target}"
    return ["ssh", target, command]


def scp_from_remote(host, remote_path, local_path, user=None):
    target = host or DEFAULT_REMOTE_HOST
    run_command([
        "scp","-O",
        f"{target}:{remote_path}",
        str(local_path),
    ])


def wait_for_http(url, timeout=120):
    """Wait for an HTTP endpoint, printing useful diagnostics on failure."""
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            http_json(url, timeout=5)
            print(f"[health] OK: {url}", flush=True)
            return
        except Exception as error:
            last_error = error
            time.sleep(1)
    raise TimeoutError(
        f"Timed out after {timeout}s waiting for {url}. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


def read_remote_file(host, path, user=None):
    result = run_command(
        ssh_command(
            host,
            f"cat {remote_shell_path(path)}",
            user,
        )
    )
    return result.stdout


# ============================================================
# REMOTE ONE-SHOT MEASUREMENT AGENT
# ============================================================


def llama_request(llama_port, payload):
    return http_json(
        f"http://127.0.0.1:{llama_port}/v1/chat/completions",
        payload,
    )


def get_server_metrics(llama_port):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{llama_port}/metrics",
            timeout=10,
        ) as response:
            metrics_text = response.read().decode("utf-8")
    except Exception:
        return {}

    metrics = {}

    for line in metrics_text.splitlines():

        if not line or line.startswith("#"):
            continue

        parts = line.split()

        if len(parts) == 2:
            try:
                metrics[parts[0].split("{")[0]] = float(parts[1])
            except ValueError:
                continue

    return metrics


def collect_system_metrics():
    metrics = {}

    try:
        import psutil

        metrics["cpu_usage_avg_pct"] = float(psutil.cpu_percent(interval=0.1))
        memory = psutil.virtual_memory()
        metrics["memory_avg_mb"] = memory.used / 1024 / 1024
        metrics["memory_avg_pct"] = float(memory.percent)
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        gpu_usage, gpu_memory = result.stdout.strip().split(",", 1)
        metrics["gpu_usage_avg_pct"] = float(gpu_usage.strip())
        metrics["gpu_memory_mb"] = float(gpu_memory.strip())
    except Exception:
        pass

    return metrics


class OneShotHandler(BaseHTTPRequestHandler):
    server_version = "RemoteBenchmarkAgent/1.0"

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return

        self.send_error(404)

    def do_POST(self):
        if self.path != "/infer":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        request_body = self.rfile.read(length)
        payload = json.loads(request_body.decode("utf-8"))
        metrics_before = get_server_metrics(self.server.llama_port)
        system_metrics = collect_system_metrics()

        try:
            response = llama_request(self.server.llama_port, payload)
            metrics_after = get_server_metrics(self.server.llama_port)
            result = {
                "ok": True,
                "response": response,
                "request_bytes": len(request_body),
                "server_metrics_before": metrics_before,
                "server_metrics_after": metrics_after,
                "system_metrics": system_metrics,
            }
        except Exception as error:
            result = {"ok": False, "error": str(error)}

        response_body = json.dumps(result).encode("utf-8")
        self.send_response(200 if result.get("ok") else 500)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

        self.server.result_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        self.server.output_path.write_text(
            ((result.get("response", {}).get("choices") or [{}])[0]
             .get("message") or {}).get("content", ""),
            encoding="utf-8",
        )

        threading.Thread(
            target=self.server.shutdown,
            daemon=True,
        ).start()

    def log_message(self, format_string, *args):
        return


def run_agent(args):
    server = ThreadingHTTPServer(("0.0.0.0", args.agent_port), OneShotHandler)
    server.llama_port = args.llama_port
    server.result_path = Path(args.result_file)
    server.output_path = Path(args.output_file)
    server.serve_forever()
    server.server_close()


def run_remote_request(args):
    payload = json.loads(args.request_payload_json)
    started = time.perf_counter()
    try:
        response = http_json(
            args.request_url,
            payload,
            timeout=args.request_timeout,
        )
        result = {
            "ok": True,
            "response": response,
            "request_wall_time_s": time.perf_counter() - started,
        }
    except Exception as error:
        result = {
            "ok": False,
            "error": str(error),
            "request_wall_time_s": time.perf_counter() - started,
        }

    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 2


def calculate_energy(path):
    try:
        with path.open(newline="", encoding="utf-8") as energy_file:
            rows = csv.DictReader(energy_file)
            total = 0.0
            found_sample = False
            for row in rows:
                power = row.get("SYSTEM_POWER (Watts)") or row.get("SYSTEM_POWER")
                delta = row.get("Delta") or row.get("delta")
                try:
                    power_value = float(power)
                    delta_value = float(delta)
                except (TypeError, ValueError):
                    continue
                if power_value >= 0 and delta_value >= 0:
                    total += power_value * delta_value / 1000
                    found_sample = True
            return total if found_sample else None
    except (OSError, csv.Error):
        return None


# ============================================================
# REMOTE ORCHESTRATOR
# ============================================================


def remote_shell_path(path):
    return shlex.quote(str(path))


def remote_ssh_peer_ports(args):
    """Return the delay-IP-side ports of SSH connections on the remote host,
    excluding the connection used to run this query."""
    result = run_command(
        ssh_command(
            args.remote_host,
            (
                'echo "self $SSH_CONNECTION"; '
                "ss -Htn state established '( sport = :22 )' "
                f"dst {remote_shell_path(args.network_delay_ip)}"
            ),
            args.remote_user,
        )
    )
    own_port = None
    ports = set()
    ip_pattern = re.escape(args.network_delay_ip)
    for line in result.stdout.splitlines():
        if line.startswith("self "):
            fields = line.split()
            if len(fields) >= 3:
                own_port = int(fields[2])
            continue
        match = re.search(rf"{ip_pattern}:(\d+)\s*$", line.strip())
        if match:
            ports.add(int(match.group(1)))
    ports.discard(own_port)
    return ports


def find_tunnel_port(args, ports_before):
    # Inside the tunnel the agent port only exists on the remote loopback, so
    # agent-delay (which filters on enp6s0) must target the tunnel's SSH
    # connection instead: remote:22 -> <delay ip>:<ephemeral port>.
    new_ports = remote_ssh_peer_ports(args) - ports_before
    if len(new_ports) != 1:
        raise RuntimeError(
            "Could not identify the SSH tunnel connection on the remote host "
            f"(new connections to {args.network_delay_ip}: {sorted(new_ports)}). "
            "Another SSH session may have opened at the same time; retry."
        )
    return new_ports.pop()


def apply_network_delay(args, delay_port):

    if args.network_delay_ms <= 0:
        return None

    # Clear any delay left over from an interrupted run; agent-delay-setup
    # fails with "File exists" if its tc filter is already installed.
    run_command(
        ssh_command(
            args.remote_host,
            "sudo -n /usr/local/bin/agent-delay-clear",
            args.remote_user,
        ),
        check=False,
    )

    command = ssh_command(
        args.remote_host,
        (
            f"{remote_shell_path(args.agent_delay_script)} "
            f"{remote_shell_path(args.network_delay_ip)} "
            f"{delay_port} {args.network_delay_ms}"
        ),
        args.remote_user,
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # agent-delay reports its status on stderr; skip the SSH login banner.
    output = ""
    match = None
    for line in process.stderr:
        output = line
        match = re.search(r"Delay active\. PID: (\d+)", line)
        if match:
            break
    if not match:
        process.terminate()
        _, stderr = process.communicate(timeout=5)
        raise RuntimeError(
            "agent-delay did not report an active PID. "
            f"Output: {output.strip()} Error: {stderr.strip()}"
        )

    pid = int(match.group(1))
    print(
        f"Network delay active on SSH tunnel to {args.network_delay_ip}:{delay_port}; PID: {pid}"
    )
    return process, pid


def remove_network_delay(args, delay_process, delay_pid):

    if delay_process is None or delay_pid is None:
        return

    run_command(
        ssh_command(
            args.remote_host,
            f"kill -TERM {delay_pid}",
            args.remote_user,
        ),
        check=False,
    )
    try:
        delay_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        delay_process.terminate()
        delay_process.wait(timeout=5)


def open_ssh_tunnel(args):
    """Forward the llama-server and agent ports from the remote host to localhost."""
    target = args.remote_host or DEFAULT_REMOTE_HOST
    if args.remote_user:
        target = f"{args.remote_user}@{target}"
    process = subprocess.Popen(
        [
            "ssh", "-N",
            "-o", "ExitOnForwardFailure=yes",
            "-L", f"{args.llama_port}:127.0.0.1:{args.llama_port}",
            "-L", f"{args.agent_port}:127.0.0.1:{args.agent_port}",
            target,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SSH tunnel to {target} exited: {process.stderr.read().strip()}")
        try:
            socket.create_connection(("127.0.0.1", args.llama_port), timeout=1).close()
            print(
                f"[tunnel] localhost:{args.llama_port},{args.agent_port} -> {target}",
                flush=True,
            )
            return process
        except OSError:
            time.sleep(0.5)
    process.terminate()
    raise TimeoutError(f"SSH tunnel to {target} did not open within 30s")


def close_ssh_tunnel(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def remote_start_server(args):
    """Start llama-server detached from SSH and verify that its process survives."""
    binary = "/home/gl_greenai/llama.cpp/build-cuda/bin/llama-server"
    log_path = args.remote_root / "llama-server.log"
    pid_path = args.remote_root / "llama-server.pid"

    # Quote every remote path. setsid detaches the process from the SSH session;
    # the PID file and startup diagnostics make failures inspectable.
    remote_command = (
        f"set -eu; "
        f"mkdir -p {remote_shell_path(args.remote_root)}; "
        f"test -x {remote_shell_path(binary)} || "
        f"{{ echo '[server-start] ERROR: executable missing/not executable: {binary}' >&2; exit 10; }}; "
        f"test -r {remote_shell_path(args.remote_model)} || "
        f"{{ echo '[server-start] ERROR: model missing/unreadable: {args.remote_model}' >&2; exit 11; }}; "
        # Pre-flight: catch truncated/corrupt GGUFs and missing GPUs before llama-server segfaults.
        f"magic=$(head -c 4 {remote_shell_path(args.remote_model)}); "
        f"[ \"$magic\" = GGUF ] || "
        f"{{ echo \"[preflight] ERROR: bad GGUF header ($magic): {args.remote_model}\" >&2; exit 13; }}; "
        f"size=$(stat -c %s {remote_shell_path(args.remote_model)}); "
        f"echo \"[preflight] model size=$size bytes\"; "
        f"[ \"$size\" -ge 104857600 ] || "
        f"{{ echo '[preflight] ERROR: model file < 100 MB, likely truncated' >&2; exit 14; }}; "
        f"if [ -n {shlex.quote(args.remote_model_sha256 or '')} ]; then "
        f"  echo '[preflight] verifying sha256 (may take a moment)'; "
        f"  echo {shlex.quote((args.remote_model_sha256 or '') + '  ' + str(args.remote_model))} | sha256sum -c - >&2 || "
        f"  {{ echo '[preflight] ERROR: sha256 mismatch, re-download the model' >&2; exit 15; }}; "
        f"fi; "
        f"nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader || "
        f"{{ echo '[preflight] ERROR: nvidia-smi failed, no usable GPU for -ngl 99' >&2; exit 16; }}; "
        f"if command -v setsid >/dev/null 2>&1; then "
        f"  setsid nohup {remote_shell_path(binary)} "
        f"  -m {remote_shell_path(args.remote_model)} "
        f"  --host 0.0.0.0 --port {args.llama_port} --metrics "
        f"  -ngl 99 -c 4096 -b 512 -ub 256 "
        f"  --reasoning off --no-cache-prompt --cache-ram 0 "
        f"  > {remote_shell_path(log_path)} 2>&1 < /dev/null & "
        f"else "
        f"  nohup {remote_shell_path(binary)} "
        f"  -m {remote_shell_path(args.remote_model)} "
        f"  --host 0.0.0.0 --port {args.llama_port} --metrics "
        f"  -ngl 99 -c 4096 -b 512 -ub 256 "
        f"  --reasoning off --no-cache-prompt --cache-ram 0 "
        f"  > {remote_shell_path(log_path)} 2>&1 < /dev/null & "
        f"fi; "
        f"pid=$!; echo \"$pid\" > {remote_shell_path(pid_path)}; "
        f"echo \"[server-start] launched PID=$pid\"; "
        f"sleep 2; "
        f"if kill -0 \"$pid\" 2>/dev/null; then "
        f"  echo \"[server-start] process alive after 2s (PID=$pid)\"; "
        f"else "
        f"  echo \"[server-start] ERROR: process exited within 2s (PID=$pid)\" >&2; "
        f"  echo '[server-start] Last 80 lines of server log:' >&2; "
        f"  tail -80 {remote_shell_path(log_path)} >&2 || true; exit 12; "
        f"fi"
    )
    print(f"[server-start] Host={args.remote_host} port={args.llama_port}", flush=True)
    print(f"[server-start] Binary={binary}", flush=True)
    print(f"[server-start] Model={args.remote_model}", flush=True)
    print(f"[server-start] Log={log_path}", flush=True)
    result = run_command(ssh_command(args.remote_host, remote_command, args.remote_user))
    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.stderr.strip():
        print("[server-start remote stderr]\\n" + result.stderr.rstrip(), flush=True)
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""


def remote_stop_server(args):
    """Stop only this script's llama-server executable, without matching the SSH shell."""
    command = (
        "pids=$(pgrep -f '^/home/gl_greenai/llama.cpp/build-cuda/bin/llama-server' || true); "
        "if [ -n \"$pids\" ]; then "
        "  echo \"[server-stop] Stopping llama-server PID(s): $pids\"; "
        "  kill $pids 2>/dev/null || true; "
        "  sleep 2; "
        "  remaining=$(pgrep -f '^/home/gl_greenai/llama.cpp/build-cuda/bin/llama-server' || true); "
        "  if [ -n \"$remaining\" ]; then "
        "    echo \"[server-stop] Force-stopping remaining PID(s): $remaining\"; "
        "    kill -KILL $remaining 2>/dev/null || true; "
        "  fi; "
        "else echo '[server-stop] No matching llama-server process found'; fi"
    )
    result = run_command(
        ssh_command(args.remote_host, command, args.remote_user),
        check=False,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.stderr.strip():
        print("[server-stop remote stderr]\\n" + result.stderr.rstrip(), flush=True)


def remote_run_one(args, run_dir, prompt, model_name, prompt_size, run_number, schedule_index):
    remote_run_dir = args.remote_root / model_name / args.quantization / prompt_size / f"run_{run_number:02d}"
    run_command(
        ssh_command(
            args.remote_host,
            f"mkdir -p {remote_shell_path(remote_run_dir)}",
            args.remote_user,
        )
    )

    remote_energy = remote_run_dir / "energy.csv"
    remote_result = remote_run_dir / "agent_result.json"
    remote_output = remote_run_dir / "output.txt"
    remote_agent_log = remote_run_dir / "agent.log"
    client_energy = run_dir / "client_energy.csv"
    client_result_tmp = run_dir / "client_energy_request.json"

    agent_port = args.agent_port
    remote_venv = Path(args.remote_script).parent.parent / "venv" / "bin" / "activate"
    # Kill any agent/energibridge left over from the previous run that still holds the port.
    # Must be a separate ssh call: if it shared a shell with the launch command below, pkill -f
    # would match that shell's own command line (which contains "--agent --agent-port N") and kill it.
    cleanup_agent = (
        f"fuser -k {agent_port}/tcp >/dev/null 2>&1 || true; "
        f"pkill -f -- '--agent --agent-por[t] {agent_port}' >/dev/null 2>&1 || true; "
        f"sleep 1"
    )
    start_agent = (
        # Source the venv as its own statement: with "A && B ... &" the whole and-or list is
        # backgrounded in a subshell that keeps ssh's stdout open, so ssh never returns.
        f". {remote_shell_path(remote_venv)} || exit 1; "
        f"setsid nohup energibridge "
        f"-o {remote_shell_path(remote_energy)} "
        f"-c {remote_shell_path(remote_run_dir / 'client_result.json')} "
        f"-i {ENERGY_INTERVAL_MS} -g -- "
        f"python3 {remote_shell_path(args.remote_script)} --agent "
        f"--agent-port {agent_port} --llama-port {args.llama_port} "
        f"--result-file {remote_shell_path(remote_result)} "
        f"--output-file {remote_shell_path(remote_output)} "
        f"< /dev/null > {remote_shell_path(remote_agent_log)} 2>&1 & echo $!"
    )

    print(f"  [agent] remote run dir: {remote_run_dir}", flush=True)
    print(f"  [agent] starting agent + energibridge on {args.remote_host}:{agent_port}", flush=True)
    step_start = time.perf_counter()
    print(f"  [agent] cleanup: killing leftovers on port {agent_port}", flush=True)
    run_command(ssh_command(args.remote_host, cleanup_agent, args.remote_user), check=False)
    print(f"  [agent] cleanup done in {time.perf_counter() - step_start:.1f}s", flush=True)
    step_start = time.perf_counter()
    print(f"  [agent] launch command: {start_agent}", flush=True)
    agent_pid = run_command(ssh_command(args.remote_host, start_agent, args.remote_user)).stdout.strip()
    print(f"  [agent] launched PID={agent_pid} in {time.perf_counter() - step_start:.1f}s; log={remote_agent_log}", flush=True)
    step_start = time.perf_counter()
    try:
        wait_for_http(f"http://127.0.0.1:{agent_port}/health")
        print(f"  [agent] healthy after {time.perf_counter() - step_start:.1f}s", flush=True)
    except Exception:
        print(f"  [agent] ERROR: agent did not become healthy. Remote agent log:", flush=True)
        log = run_command(
            ssh_command(args.remote_host, f"tail -40 {remote_shell_path(remote_agent_log)}", args.remote_user),
            check=False,
        )
        print(log.stdout.rstrip() or "  (agent log empty)", flush=True)
        raise

    network_probe_start = time.perf_counter()
    http_json(f"http://127.0.0.1:{agent_port}/health")
    network_probe_end = time.perf_counter()
    network_rtt_ms = (network_probe_end - network_probe_start) * 1000.0
    print(f"  [agent] health RTT {network_rtt_ms:.1f} ms", flush=True)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "seed": 42,
        "max_tokens": args.output_tokens,
        "stream": False,
        "reasoning_effort": "none",
    }

    request_body = json.dumps(payload).encode("utf-8")
    request_url = f"http://127.0.0.1:{agent_port}/infer"
    client_energy_command = [
        "energibridge",
        "-o", str(client_energy),
        "-c", str(client_result_tmp),
        "-i", str(ENERGY_INTERVAL_MS),
        "-g",
        "--",
        sys.executable,
        str(Path(__file__).resolve()),
        "--remote-request",
        "--request-url", request_url,
        "--request-payload-json", json.dumps(payload),
        "--request-timeout", "900",
    ]
    print(f"  [infer] sending {prompt_size} prompt ({len(prompt)} chars, max_tokens={args.output_tokens})", flush=True)
    infer_start = time.perf_counter()
    run_command(client_energy_command)
    print(f"  [infer] done in {time.perf_counter() - infer_start:.1f}s", flush=True)

    client_result = json.loads(
        client_result_tmp.read_text(encoding="utf-8")
    )
    if not client_result.get("ok"):
        raise RuntimeError(client_result.get("error", "Remote inference failed"))

    response = client_result["response"]

    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Remote inference failed"))

    response_body_bytes = len(json.dumps(response).encode("utf-8"))
    request_wall_time_s = client_result["request_wall_time_s"]
    print(f"  [scp] copying energy.csv, client_result.json, output.txt from {args.remote_host}", flush=True)

    for remote_path, local_name in [
        (remote_energy, "energy.csv"),
        (remote_run_dir / "client_result.json", "client_result.json"),
        (remote_output, "output.txt"),
    ]:
        scp_from_remote(
            args.remote_host,
            remote_path,
            run_dir / local_name,
            args.remote_user,
        )

    server_response = response["response"]
    usage = server_response.get("usage") or {}
    choice = (server_response.get("choices") or [{}])[0]

    metadata = {
        "model": model_name,
        "architecture": args.architecture,
        "model_path": args.remote_model,
        "quantization": args.quantization,
        "prompt_size": prompt_size,
        "prompt_file": str(
            args.remote_prompt_dir / f"prompt_{prompt_size}.txt"
        ),
        "run": run_number,
        "schedule_index": schedule_index,
        "order_seed": args.order_seed,
        "requested_output_tokens": args.output_tokens,
        "actual_output_tokens": usage.get("completion_tokens"),
        "deployment": "remote",
        "runtime": "llama-server",
        "llama_metrics": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "request_wall_time_s": request_wall_time_s,
            "finish_reason": choice.get("finish_reason"),
            "output_token_count_source": "remote llama-server response usage",
            "server_metrics_before": response.get("server_metrics_before", {}),
            "server_metrics_after": response.get("server_metrics_after", {}),
        },
        "system_metrics": response.get("system_metrics", {}),
        "client_energy_j": calculate_energy(client_energy),
        "network": {
            "condition": args.network_condition,
            "delay_script": args.agent_delay_script,
            "delay_target_ip": args.network_delay_ip,
            "delay_target_port": args.delay_target_port,
            "configured_delay_ms": args.network_delay_ms,
            "network_rtt_ms": network_rtt_ms,
            "inference_request_wall_time_ms": request_wall_time_s * 1000.0,
            "client_energy_file": str(client_energy),
            "request_bytes": len(request_body),
            "response_bytes": response_body_bytes,
            "total_bytes": len(request_body) + response_body_bytes,
        },
        "energy_measurement": {
            "source": "EnergiBridge on remote server",
            "method": "Remote sampled system power during one remote HTTP inference request.",
            "unit": "J",
        },
        "client_energy_measurement": {
            "source": "EnergiBridge on the orchestrator MacBook",
            "method": "Local sampled system power during the complete remote HTTP request and response wait.",
            "file": str(client_energy),
            "unit": "J",
        },
        "configuration": {
            "architecture": args.architecture,
            "deployment": "remote",
            "runtime": "llama-server",
            "output_tokens": args.output_tokens,
            "temperature": 0.6,
            "seed": 42,
            "reasoning": "off",
            "prompt_cache_enabled": False,
            "cache_ram_mb": 0,
            "context_size": 4096,
            "batch_size": 512,
            "ubatch_size": 256,
            "gpu_layers": 99,
        },
    }

    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    client_result_tmp.unlink(missing_ok=True)


def run_remote(args):
    args.remote_root = Path(args.remote_root)
    args.remote_prompt_dir = Path(args.remote_prompt_dir)
    local_root = Path(args.results_dir).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser().resolve()
    model_name = model_path.stem

    if args.remote_model is None:
        args.remote_model = str(
            Path(args.remote_model_dir) / f"{model_name}.gguf"
        )

    prompts = {
        size: read_remote_file(
            args.remote_host,
            args.remote_prompt_dir / filename,
            args.remote_user,
        ).strip()
        for size, filename in PROMPT_SIZES.items()
    }

    print(f"[remote] Loaded prompts from {args.remote_host}; stopping old llama-server", flush=True)
    remote_stop_server(args)
    delay_process = None
    delay_pid = None
    tunnel_process = None
    try:
        ports_before = None
        if args.network_delay_ms > 0:
            if not args.network_delay_ip:
                raise ValueError(
                    "A delay target IP is required when --network-delay-ms is non-zero. "
                    "Set --network-delay-ip or DEFAULT_NETWORK_DELAY_IP."
                )
            ports_before = remote_ssh_peer_ports(args)
        tunnel_process = open_ssh_tunnel(args)
        if ports_before is not None:
            args.delay_target_port = find_tunnel_port(args, ports_before)
        delay_state = apply_network_delay(args, args.delay_target_port)
        if delay_state is not None:
            delay_process, delay_pid = delay_state
        print(f"[remote] Starting llama-server with {args.remote_model}", flush=True)
        remote_start_server(args)
        health_url = f"http://127.0.0.1:{args.llama_port}/health"
        try:
            wait_for_http(health_url, timeout=120)
        except Exception:
            print(
                f"[remote] ERROR: llama-server health check failed. "
                f"Fetching remote log: {args.remote_root / 'llama-server.log'}",
                flush=True,
            )
            diagnostic_command = (
                f"echo '--- process check ---'; "
                f"pgrep -af '^/home/gl_greenai/llama.cpp/build-cuda/bin/llama-server' || true; "
                f"echo '--- listening port ---'; "
                f"(ss -ltnp 2>/dev/null | grep -E ':{args.llama_port}\\b' || true); "
                f"echo '--- llama-server log (last 120 lines) ---'; "
                f"tail -120 {remote_shell_path(args.remote_root / 'llama-server.log')} 2>&1 || true"
            )
            diagnostic = run_command(
                ssh_command(args.remote_host, diagnostic_command, args.remote_user),
                check=False,
            )
            if diagnostic.stdout.strip():
                print(diagnostic.stdout.rstrip(), flush=True)
            if diagnostic.stderr.strip():
                print(diagnostic.stderr.rstrip(), flush=True)
            raise
        print("[remote] llama-server healthy", flush=True)

        rng = random.Random(args.order_seed)
        warmup_order = list(PROMPT_SIZES)
        rng.shuffle(warmup_order)

        for prompt_size in warmup_order:
            print(f"[warmup] {prompt_size} prompt", flush=True)
            warmup_start = time.perf_counter()
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompts[prompt_size]}],
                "temperature": 0.6,
                "seed": 42,
                "max_tokens": args.output_tokens,
                "stream": False,
                "reasoning_effort": "none",
            }
            http_json(
                f"http://127.0.0.1:{args.llama_port}/v1/chat/completions",
                payload,
            )
            print(f"[warmup] {prompt_size} done in {time.perf_counter() - warmup_start:.1f}s", flush=True)

        print(f"[warmup] Done; stabilizing for {WARMUP_STABILIZATION_SECONDS}s", flush=True)
        time.sleep(WARMUP_STABILIZATION_SECONDS)

        schedule = []
        for run_number in range(1, args.runs + 1):
            block = list(PROMPT_SIZES)
            rng.shuffle(block)
            schedule.extend((run_number, prompt_size) for prompt_size in block)

        for schedule_index, (run_number, prompt_size) in enumerate(schedule):
            run_dir = local_root / model_name / args.quantization / prompt_size / f"run_{run_number:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run {schedule_index + 1}/{len(schedule)}] {prompt_size} run_{run_number:02d}", flush=True)
            remote_run_one(
                args,
                run_dir,
                prompts[prompt_size],
                model_name,
                prompt_size,
                run_number,
                schedule_index,
            )

            print(f"[run {schedule_index + 1}/{len(schedule)}] saved to {run_dir}", flush=True)
            if schedule_index < len(schedule) - 1:
                print(f"[cooldown] {RUN_COOLDOWN_SECONDS}s", flush=True)
                time.sleep(RUN_COOLDOWN_SECONDS)
    finally:
        print("[cleanup] stopping llama-server, removing delay, closing tunnel", flush=True)
        remote_stop_server(args)
        remove_network_delay(args, delay_process, delay_pid)
        close_ssh_tunnel(tunnel_process)


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the benchmark against a remote llama-server."
    )
    parser.add_argument("--model", required=False)
    parser.add_argument("--quantization", required=False)
    parser.add_argument("--architecture", choices=["Dense", "MoE"], required=False)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-user", default=None)
    parser.add_argument("--remote-model", default=None)
    parser.add_argument(
        "--remote-model-sha256",
        default=None,
        help="Optional expected sha256 of the remote model; verified before server start.",
    )
    parser.add_argument(
        "--remote-model-dir",
        default=DEFAULT_REMOTE_MODEL_DIR,
    )
    parser.add_argument(
        "--remote-prompt-dir",
        default=DEFAULT_REMOTE_PROMPT_DIR,
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_RESULTS_DIR,
    )
    parser.add_argument(
        "--remote-script",
        default=DEFAULT_REMOTE_SCRIPT,
    )
    parser.add_argument("--results-dir", default="results/raw/remote")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--llama-port", type=int, default=DEFAULT_LLAMA_PORT)
    parser.add_argument("--agent-port", type=int, default=DEFAULT_AGENT_PORT)
    parser.add_argument("--remote-request", action="store_true")
    parser.add_argument("--request-url", default=None)
    parser.add_argument("--request-payload-json", default="{}")
    parser.add_argument("--request-timeout", type=int, default=900)
    parser.add_argument(
        "--agent-delay-script",
        default=DEFAULT_AGENT_DELAY_SCRIPT,
        help="Remote wrapper invoked as ./agent-delay <ip> <port> <delay_ms>.",
    )
    parser.add_argument(
        "--network-delay-ip",
        default=DEFAULT_NETWORK_DELAY_IP,
        help="IP address passed to the remote agent-delay wrapper.",
    )
    parser.add_argument(
        "--network-delay-ms",
        type=int,
        default=DEFAULT_NETWORK_DELAY_MS,
        help="Temporary delay applied only to benchmark-agent response ports.",
    )
    parser.add_argument("--network-condition", default="baseline")
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--result-file", default="agent_result.json")
    parser.add_argument("--output-file", default="output.txt")
    return parser.parse_args()


def main():
    args = parse_args()
    args.delay_target_port = None

    if args.remote_request:
        return run_remote_request(args)

    if args.agent:
        run_agent(args)
        return

    required = [
        "model",
        "quantization",
        "architecture",
        "remote_host",
    ]

    missing = [name for name in required if getattr(args, name) is None]

    if missing:
        raise SystemExit(
            "Missing required orchestrator arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )

    run_remote(args)


if __name__ == "__main__":
    main()
