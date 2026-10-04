import argparse
import json
import random
import shlex
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
DEFAULT_REMOTE_INTERFACE = os.getenv("DEFAULT_REMOTE_INTERFACE")
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
    return ["ssh", target, command]


def scp_from_remote(host, remote_path, local_path, user=None):
    target = host or DEFAULT_REMOTE_HOST
    run_command([
        "scp","-O",
        f"{target}:{remote_path}",
        str(local_path),
    ])


def wait_for_http(url, timeout=120):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            http_json(url, timeout=5)
            return
        except Exception:
            time.sleep(1)

    raise TimeoutError(f"Timed out waiting for {url}")


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


# ============================================================
# REMOTE ORCHESTRATOR
# ============================================================


def remote_shell_path(path):
    return shlex.quote(str(path))


def apply_network_delay(args):

    if args.network_delay_ms <= 0:
        return False

    interface = remote_shell_path(args.network_interface)
    agent_port = args.agent_port

    tc_commands = [
        f"sudo -n /usr/sbin/tc qdisc replace dev {interface} "
        "root handle 1: prio bands 3",
        f"sudo -n /usr/sbin/tc qdisc replace dev {interface} "
        "parent 1:3 handle 30: netem "
        f"delay {args.network_delay_ms}ms",
        f"sudo -n /usr/sbin/tc filter replace dev {interface} "
        "protocol ip parent 1:0 prio 10 flower ip_proto tcp "
        f"src_port {agent_port} flowid 1:3",
    ]

    for command in tc_commands:
        run_command(
            ssh_command(
                args.remote_host,
                command,
                args.remote_user,
            )
        )

    return True


def remove_network_delay(args):

    if args.network_delay_ms <= 0:
        return

    run_command(
        ssh_command(
            args.remote_host,
            f"sudo -n /usr/sbin/tc qdisc del dev "
            f"{remote_shell_path(args.network_interface)} root",
            args.remote_user,
        ),
        check=False,
    )


def remote_start_server(args):
    run_command(
        ssh_command(
            args.remote_host,
            f"mkdir -p {remote_shell_path(args.remote_root)}",
            args.remote_user,
        )
    )

    command = (
        f"nohup llama-server "
        f"-m {remote_shell_path(args.remote_model)} "
        f"--host 0.0.0.0 --port {args.llama_port} --metrics "
        f"-ngl 99 -c 4096 -b 512 -ub 256 "
        f"--reasoning off --no-cache-prompt --cache-ram 0 "
        f"> {remote_shell_path(args.remote_root / 'llama-server.log')} "
        "2>&1 & echo $!"
    )
    result = run_command(ssh_command(args.remote_host, command, args.remote_user))
    return result.stdout.strip().splitlines()[-1]


def remote_stop_server(args):
    run_command(
        ssh_command(
            args.remote_host,
            "pkill -f 'llama-server.*--port' || true",
            args.remote_user,
        ),
        check=False,
    )


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

    agent_port = args.agent_port
    start_agent = (
        f"nohup energibridge "
        f"-o {remote_shell_path(remote_energy)} "
        f"-c {remote_shell_path(remote_run_dir / 'client_result.json')} "
        f"-i {ENERGY_INTERVAL_MS} -g -- "
        f"python3 {remote_shell_path(args.remote_script)} --agent "
        f"--agent-port {agent_port} --llama-port {args.llama_port} "
        f"--result-file {remote_shell_path(remote_result)} "
        f"--output-file {remote_shell_path(remote_output)} "
        f"> {remote_shell_path(remote_agent_log)} 2>&1 & echo $!"
    )

    run_command(ssh_command(args.remote_host, start_agent, args.remote_user))
    wait_for_http(f"http://{args.remote_host}:{agent_port}/health")

    network_probe_start = time.perf_counter()
    http_json(f"http://{args.remote_host}:{agent_port}/health")
    network_probe_end = time.perf_counter()
    network_rtt_ms = (network_probe_end - network_probe_start) * 1000.0

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
    request_start = time.perf_counter()
    response = http_json(f"http://{args.remote_host}:{agent_port}/infer", payload)
    request_end = time.perf_counter()

    if not response.get("ok"):
        raise RuntimeError(response.get("error", "Remote inference failed"))

    response_body_bytes = len(json.dumps(response).encode("utf-8"))
    request_wall_time_s = request_end - request_start

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
        "network": {
            "condition": args.network_condition,
            "interface": args.network_interface,
            "configured_delay_ms": args.network_delay_ms,
            "delayed_source_port": args.agent_port,
            "network_rtt_ms": network_rtt_ms,
            "inference_request_wall_time_ms": request_wall_time_s * 1000.0,
            "request_bytes": len(request_body),
            "response_bytes": response_body_bytes,
            "total_bytes": len(request_body) + response_body_bytes,
        },
        "energy_measurement": {
            "source": "EnergiBridge on remote server",
            "method": "Remote sampled system power during one remote HTTP inference request.",
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

    remote_stop_server(args)
    delay_applied = False
    try:
        delay_applied = apply_network_delay(args)
        remote_start_server(args)
        wait_for_http(f"http://{args.remote_host}:{args.llama_port}/health")

        rng = random.Random(args.order_seed)
        warmup_order = list(PROMPT_SIZES)
        rng.shuffle(warmup_order)

        for prompt_size in warmup_order:
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
                f"http://{args.remote_host}:{args.llama_port}/v1/chat/completions",
                payload,
            )

        time.sleep(WARMUP_STABILIZATION_SECONDS)

        schedule = []
        for run_number in range(1, args.runs + 1):
            block = list(PROMPT_SIZES)
            rng.shuffle(block)
            schedule.extend((run_number, prompt_size) for prompt_size in block)

        for schedule_index, (run_number, prompt_size) in enumerate(schedule):
            run_dir = local_root / model_name / args.quantization / prompt_size / f"run_{run_number:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            remote_run_one(
                args,
                run_dir,
                prompts[prompt_size],
                model_name,
                prompt_size,
                run_number,
                schedule_index,
            )

            if schedule_index < len(schedule) - 1:
                time.sleep(RUN_COOLDOWN_SECONDS)
    finally:
        remote_stop_server(args)
        if delay_applied:
            remove_network_delay(args)


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
    parser.add_argument("--network-interface", default=DEFAULT_REMOTE_INTERFACE)
    parser.add_argument(
        "--network-delay-ms",
        type=int,
        choices=[0, 25, 50, 100],
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
