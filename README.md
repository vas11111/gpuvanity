# gpuvanity

GPU-accelerated Solana vanity address generator. Finds Ed25519 keypairs whose Base58 address starts or ends with the patterns you want.

## Quick Start

```bash
# install
apt install -y python3-full python3-venv ocl-icd-libopencl1
apt install -y libnvidia-compute-570   # match your driver version

git clone <repo-url> gpuvanity && cd gpuvanity
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# find an address starting with "Sol"
python3 main.py --prefix Sol

# find an address ending with "bonk"
python3 main.py --suffix bonk

# find addresses matching any of several patterns
python3 main.py --prefix Dead,Beef --suffix bonk,pump

# run continuously, saving every hit
python3 main.py --suffix bonk --count 0
```

## How It Works

Every GPU generates random Ed25519 keypairs, encodes the public key as a Base58 address, and checks if it matches your prefix/suffix patterns -- all on-device. Only matching keypairs are sent back to the host and saved to disk.

When you specify both `--prefix` and `--suffix`, the search uses **OR logic**: an address that matches *any* listed prefix or *any* listed suffix is a hit.

## Benchmarks (9x RTX 4090)

| Metric | Value |
|---|---|
| Per-GPU throughput | ~80 MH/s |
| Aggregate (9 GPUs) | ~720 MH/s |

### Expected search times

| Pattern Length | Combinations | 1 GPU (~80 MH/s) | 9 GPUs (~720 MH/s) |
|:-:|:-:|:-:|:-:|
| 3 chars | ~195K | instant | instant |
| 4 chars | ~11.3M | < 1s | < 1s |
| 5 chars | ~656M | ~8s | ~1s |
| 6 chars | ~38B | ~8 min | ~53s |
| 7 chars | ~2.2T | ~7.5 hr | ~50 min |
| 8 chars | ~128T | ~18 days | ~2 days |

Times are averages. Actual results are probabilistic -- you may get lucky or unlucky.

## CLI Reference

```
python3 main.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--prefix` | | Comma-separated prefix targets |
| `--suffix` | | Comma-separated suffix targets |
| `--count N` | `1` | Keys to find per pattern. `0` = run forever |
| `--output-dir PATH` | `./keys` | Where to save found keypairs |
| `--case-sensitive / --no-case-sensitive` | on | Case-sensitive matching |
| `--batch-exp N` | `28` | Keys per GPU per iteration = 2^N |
| `--select-device` | off | Interactive GPU picker |
| `--devices` | | List detected GPUs and exit |

### Examples

```bash
# case-insensitive search for multiple prefixes, find 5 of each
python3 main.py --prefix Alpha,Beta,Omega --count 5 --no-case-sensitive

# list available GPUs
python3 main.py --devices

# run via Docker
docker build -t gpuvanity .
docker run --rm --gpus all gpuvanity --prefix Sol --count 3
```

## Benchmark Tool

A standalone single-GPU benchmark is included:

```bash
python3 bench.py                  # default: GPU 0, 10 iterations
python3 bench.py --gpu 2          # test a specific GPU
python3 bench.py --iters 20       # more iterations for stable numbers
```

## Output

Keypairs are saved as Solana CLI-compatible JSON files, organized by pattern:

```
keys/
  pfx_Sol/
    So1abc...xyz.json
  sfx_bonk/
    7Kqabc...bonk.json
```

Import directly with the Solana CLI:

```bash
solana-keygen pubkey keys/pfx_Sol/So1abc...xyz.json
```

## Docker

```bash
docker build -t gpuvanity .
docker run --rm --gpus all gpuvanity --suffix bonk --count 10
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU passthrough.

## Installation Notes

- **NVIDIA driver**: must include the OpenCL ICD. Install the `libnvidia-compute-XXX` package matching your driver version.
- **First run**: kernel compilation takes 15-30 seconds per GPU. All subsequent iterations are instant.
- **Batch exponent**: `28` (268M keys/iter) works well on all modern NVIDIA GPUs. Lower to `26` if you run into memory issues on smaller cards.
- **Longer patterns**: each additional character multiplies search time by ~58x. Patterns of 7+ characters are long-running jobs.
