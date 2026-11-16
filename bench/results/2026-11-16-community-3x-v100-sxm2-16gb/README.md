# Community benchmark: 3x Tesla V100-SXM2-16GB (sm_70), Xeon E5-2630 v4, 125.7 GiB RAM

Measured on 2026-11-16 on a three-card SXM2 board running the V100 fork's
engine 0.1.45.7 (local sm_70 CUDA 12.6 build, source `27c357d5`), model
`Qwen3.8-Flash-Next-GSQ-RCO` IQ3_S at the 262,144-token native window. The
published sweep ran with all three cards at a 300 W power limit; companion
sweeps at the factory 250 W and at 200 W show long-context decode collapses,
described under "Power limit and decode collapse".

At 300 W, median decode was **68.5 / 73.6 / 79.1 / 61.0 / 60.0 tok/s** at
~4.1k, ~8.8k, 29.5k, 117.8k and 256k prompt tokens, and median prompt
throughput was **946 / 1,277 / 2,199 / 2,606 / 2,512 tok/s** over the same
rows. Recall was 9 of 9 needles, including at 256k. These are synthetic
repeated-text prompts with a 64-token output cap, measured by the engine's own
`/metrics`; they say nothing about answer quality or about other workloads.

## Hardware and software

- 3x Tesla V100-SXM2-16GB on one SXM2 baseboard: PCI bus IDs
  `0000:01:00.0`, `0000:02:00.0`, `0000:03:00.0`. `nvidia-smi nvlink -s`
  reports `all links are inActive` on every card, so inter-stage transfers go
  over the host rather than an NVLink path.
- Intel Xeon E5-2630 v4, 10 cores / 20 threads, no AVX-512 (the engine reports
  its expert kernels run on AVX-2).
- 125.7 GiB RAM. The expert arena is 46.84 GiB read through the file cache at
  startup (0.27 GiB/s). `vm.nr_hugepages=0`; transparent huge pages requested.
- Ubuntu 24.04.5, kernel 6.8.0-147-generic, NVIDIA driver 580.178.04, CUDA 12.6.
- Engine 0.1.45.7, `src 27c357d517399187` per `engine-cuda12/BUILD.json`,
  arch 70, vision engine built for CPU.
- Clocks were not fixed. Power limits were set with `nvidia-smi -pl`
  ([power-set.log](power-set.log) for 300 W,
  [power200/power-set.log](power200/power-set.log) for 200 W); the 250 W sweeps
  ran at the factory limit, which the 300 W log records as the previous value.

## Model and configuration

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `IQ3_S`:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (native pack source)
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (PLE table)
- MTP drafter, cyrillic draft vocabulary

Engine settings (service config, none overridden per request):

```text
--pack iq3_s --native <...00001-of-00002.gguf> --ple-gguf <...00002-of-00002.gguf>
--expert-profile <expert-profile.bin> --expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.5 --mtp <rt>
--max-context 262144 --kv int8 --ple-io ram --kv-resident 32768
--vision --conversation-cache-mib 24576 --conversation-cache-slots 8
--conversation-cache-min-tokens 4096 --vram-reserve-mib 1500
```

`layer_split: auto`, `parallel: 2`, GPU list `[0, 1, 2]`. What the engine
reported at startup on this layout:

- layer split: CUDA0 `[0, 16)`, CUDA1 `[16, 32)`, CUDA2 `[32, 48)` and the head;
  11.08 and 10.59 GiB free on the later stages
- VRAM plan on CUDA0: expert cache 8.48 GiB / 4,847 slots (PROFILE policy,
  ranked by routing frequency, no eviction); 8,192-token prefill window
  borrowing 2,495 cache slots; 1,500 MiB user reserve
- two batch slot sessions per card (0.36 GiB each), idle during this test
- a 262k window does not fit these 16 GiB cards' KV plan, so `--kv-resident
  32768` keeps 32k tokens resident and the engine streams the rest from RAM

## Method

`bench/run_v100_bench.py --repeats 3`, unmodified, against
`http://127.0.0.1:18201` with `--model qwen3.8-flash-next-iq3_s`. Every row is
a separate request with a fresh unique prefix (`reused=0` in every row) and a
64-token output cap; prompt and decode times come from the server's `/metrics`,
not client wall clock. Four sweeps ran between 01:00 and 06:03 local time, one
power limit at a time; the seeds in the sweep table below fix the prompt texts.
Scheduled host jobs were stopped for each sweep and the gateway's production
engine served zero requests in the sweep windows, so the engine under test was
the only load on the box. A first attempt the previous evening was discarded
because ordinary traffic shared the machine during it.

Raw rows for the published 300 W sweep: [matrix.json](matrix.json); the
harness's own stdout: [bench_stdout.log](bench_stdout.log).

| row | prompt tokens | reused | prompt tok/s | decode tok/s |
|---|---:|---:|---:|---:|
| ~4K#0 | 4,143 | 0 | 882.9 | 62.2 |
| ~4K#1 | 4,143 | 0 | 926.5 | 76.5 |
| ~4K#2 | 4,145 | 0 | 927.4 | 66.0 |
| ~4K#0 | 4,376 | 0 | 966.0 | 70.7 |
| ~4K#1 | 4,375 | 0 | 965.2 | 69.0 |
| ~4K#2 | 4,376 | 0 | 965.9 | 68.0 |
| ~8K#0 | 8,784 | 0 | 1,281.8 | 73.6 |
| ~8K#1 | 8,788 | 0 | 1,273.2 | 71.9 |
| ~8K#2 | 8,784 | 0 | 1,277.2 | 86.6 |
| 32K#0 | 29,496 | 0 | 2,209.2 | 79.1 |
| 32K#1 | 29,493 | 0 | 2,199.2 | 91.2 |
| 32K#2 | 29,492 | 0 | 2,196.6 | 71.8 |
| 128K#0 | 117,839 | 0 | 2,615.2 | 61.0 |
| 128K#1 | 117,835 | 0 | 2,604.7 | 57.5 |
| 128K#2 | 117,837 | 0 | 2,605.8 | 64.0 |
| 256K#0 | 256,078 | 0 | 2,514.8 | 58.7 |
| 256K#1 | 256,080 | 0 | 2,511.6 | 62.6 |
| 256K#2 | 256,077 | 0 | 2,508.2 | 60.0 |

Medians by row size: 946 / 1,277 / 2,199 / 2,606 / 2,512 tok/s prompt
and 68.5 / 73.6 / 79.1 / 61.0 / 60.0 tok/s decode at ~4.1k / ~8.8k / 29.5k /
117.8k / 256k.

Decode here is speculative decode with the MTP drafter at depth 4. A run
without `--mtp` would not be comparable.

## Sweep comparison (fixed prompts per seed, different power limits)

| sweep | seed | decode tok/s medians, 4K/8K/32K/128K/256K | prompt tok/s medians, same order |
|---|---|---|---|
| 300 W (published) | 20261119 | 68.5 / 73.6 / 79.1 / 61.0 / 60.0 | 946 / 1,277 / 2,199 / 2,606 / 2,512 |
| 250 W run 1 (factory) | 20261117 | 68.8 / 67.0 / 72.0 / 48.3 / 44.0 | 946 / 1,224 / 2,164 / 2,548 / 2,436 |
| 250 W run 2 (factory) | 20261118 | 68.4 / 65.1 / 76.1 / 48.4 / 40.0 | 947 / 1,253 / 2,194 / 2,580 / 2,481 |
| 200 W | 20261120 | 51.8 / 47.8 / 51.2 / 46.7 / 39.9 | 938 / 1,217 / 2,148 / 2,528 / 2,405 |

## Power limit and decode collapse

At the factory 250 W limit and at 200 W, some long-context runs decoded
at 2.9-10.0 tok/s instead of the 40-62 tok/s of their neighbouring repeats:

| limit | seed | row | decode tok/s | wall s | decode expert hit rate |
|---:|---|---|---:|---:|---:|
| 250 W | 20261117 | 128K#1 | 7.9 | 55.6 | 0.996 |
| 250 W | 20261117 | 256K#0 | 3.0 | 132.0 | 0.967 |
| 250 W | 20261118 | 128K#1 | 10.0 | 53.0 | 0.996 |
| 250 W | 20261118 | 256K#0 | 2.9 | 127.1 | 0.996 |
| 200 W | 20261120 | 256K#2 | 7.7 | 117.5 | 0.958 |

At 300 W none of the 18 long-context runs (128k and 256k) collapsed, and the
128k and 256k decode medians rise from 40-48 to 60-61 tok/s. Three notes on the
mechanism, before any root cause is claimed:

- every collapsed run still reports a 0.967-0.997 decode expert-cache hit rate
  in the engine's per-request log, so the stall is not an expert-miss storm;
- in the 200 W sweep, GPU 0 shows the SW Power Cap bit in 98 of 817 one-second samples, p95 220 W while the GPU was active; GPU 1 shows the SW Power Cap bit in 102 of 818 one-second samples, p95 219 W while the GPU was active; GPU 2 shows the SW Power Cap bit in 110 of 818 one-second samples, p95 221 W while the GPU was active. The bit appears in prefill
  phases rather than in the collapsed decode phases, so a power-cap clock clamp
  alone does not explain the decode stalls either;
- these 16 GiB cards run `--kv-resident 32768`, so 128k and 256k KV is streamed
  from host RAM during decode - the other candidate, which I have not isolated.

The correlation (raise the limit, collapses gone, long-context decode up
25-50%) is what this report claims; the artifacts are here so the mechanism can
be chased from them.

Raw 200 W sweep: [power200/matrix.json](power200/matrix.json), with a
one-second `nvidia-smi` log ([power200/throttle.csv](power200/throttle.csv))
and the run timeline ([power200/timeline.log](power200/timeline.log)). Raw
250 W sweeps: [power250-run1/matrix.json](power250-run1/matrix.json) and
[power250-run2/matrix.json](power250-run2/matrix.json).

## Recall checks

`tools/needle_bench.py --lengths 32k,128k,256k --depths 10,50,90` against
the 300 W engine: **9 of 9 found** at 128k and 256k and 32k
prompt tokens. Raw: [needles.json](needles.json). The 256k rows sit inside the
262k window with KV streamed from RAM, and recall is unaffected at every
depth.

## What to take from this

- Three SXM2 V100 16 GB cards hold this IQ3_S model at the full 256k
  window; with the profiled expert cache the decode expert hit rate stays at
  0.958-0.997, and decode lands close to the published 3x V100 32 GB PCIe
  report (60.0 vs 66.9 tok/s at 256k) on half the VRAM.
- Prefill on this SXM2 board is markedly faster than on that PCIe box: 2,393-
  2,515 tok/s at 256k vs 1,208-1,751 tok/s there, which fits the SXM2 host path
  against a Gen 3 x8 slot in that rig.
- Long-context decode on this board was power-limited at 250 W: setting the
  SXM2 factory limit to 300 W raises the 128k and 256k decode medians from
  40-48 to 60-61 tok/s and removes every collapse in 18 long-context runs.
  Check a card's enforced power limit before comparing long-context decode:
  `nvidia-smi --query-gpu=power.limit,enforced.power.limit --format=csv` plus
  the per-request decode spread is enough to see it.
