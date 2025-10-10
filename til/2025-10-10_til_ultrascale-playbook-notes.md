# Notes on The Ultra-Scale Playbook

## high level overview
- Goal: train large language models efficiently on many GPUs.
- Two recurring challenges: fit in memory and keep GPUs busy.
- Memory owners during training: parameters, gradients, optimizer states,
  activations. Activations depend on batch size and sequence length.
- Tools to scale: data parallelism (DP), tensor/sequence/context parallelism
  (TP/SP/CP), pipeline parallelism (PP), expert parallelism (EP), and ZeRO.
- Profile first. Overlap compute with communication whenever possible.
- No silver bullet. Combine methods based on model size, sequence length,
  network bandwidth, and available GPUs.

## first steps: training on one gpu
- Training step: forward → backward → optimizer step.
- Batch size matters for convergence and throughput. Report both samples and
  tokens (batch tokens = batch samples × sequence length).
- Memory spikes differ between the first step and later steps (cache allocator
  warms up; optimizer state appears after step 1).
- To estimate parameter count N for a standard transformer: N ≈ h × v +
  L × (12 h^2 + 13 h) + 2 h; the h^2 term dominates at scale.
- Mixed precision (typically BF16 compute + FP32 master weights) shifts where
  memory is used; it improves speed and reduces activation memory bandwidth
  pressure but does not magically remove total memory needs.

### memory usage in transformers
- Stored items: model weights, gradients, optimizer states, activations.
- Precision affects memory: FP32=4B, BF16/FP16=2B, FP8≈1B. Many setups keep
  FP32 master weights and/or FP32 grads for stability.
- Activations dominate at long sequences and moderate batch sizes; parameters
  dominate for very large models.

### activation recomputation (checkpointing)
- Trade compute for memory by recomputing parts of the forward during backward.
- Full vs selective recomputation: selective reduces big tensors (e.g. attention
  matrices) and is often enough (FlashAttention already does this internally).
- Recomputation increases ops (affects HFU) but often improves overall runtime
  because memory traffic is the real bottleneck.

Step‑by‑step improvements
- Naive: keep all activations; simplest but often OOM.
- Selective checkpointing: checkpoint only the heaviest pieces (attention, MLP
  blocks); big memory win with small extra FLOPs; works great with FlashAttn.
- Full checkpointing: checkpoint most layers; maximum memory reduction; ~30%
  extra compute; useful for very long context.
- Validate with memory profiler: ensure peaks drop where expected.

### gradient accumulation
- Split the global batch into micro‑batches processed sequentially.
- Formula: global_batch = micro_batch × grad_accum_steps.
- Keeps peak memory lower (fewer activations live at once) but increases wall
  time per optimizer step. Good when memory‑bound.

Step‑by‑step improvements
- Naive: run backward + all‑reduce for every micro‑batch—wastes bandwidth.
- Better: wrap non‑final micro‑batches in `no_sync()` so reductions happen only
  on the last accumulation step.
- Tune micro‑batch size to balance kernel efficiency (too small hurts math
  throughput) vs memory headroom.

### profiling gpu compute and communication
- Use PyTorch profiler + TensorBoard/Chrome trace to see CPU launch, CUDA
  streams, kernels, and NCCL comms.
- Look for: overlapped comm/compute, idle gaps, CPU launch stalls, host↔device
  transfers, and non‑contiguous buffers causing extra copies.

Step‑by‑step improvements
- Start with a short schedule (wait/warmup/active) and capture a handful of
  steps to avoid gigantic traces.
- Add NVTX ranges around forward/backward/optimizer to orient yourself in the
  timeline.
- Verify NCCL calls appear interleaved with backward kernels (true overlap).
- If comm bunches at the end, reduce bucket size or reorder buckets; if too many
  tiny NCCL calls, increase bucket size and ensure contiguity.
- Confirm multiple CUDA streams are active (compute vs comm streams).

## data parallelism (dp)
- What it is: replicate the model across GPUs; each GPU gets a different
  micro‑batch; average gradients to keep replicas identical.

Step‑by‑step improvements
- Naive sync (wait at the end):
  - Do full backward on each GPU, build all gradients, then run one big
    all‑reduce for every parameter. Simple but leaves GPUs idle during comm.
  - Analogy: everyone writes their essay, then all gather in a room to agree on
    the average—lots of waiting at the end.
- Autograd all‑reduce hooks (streaming sync):
  - Attach a reduction hook per parameter/bucket so reduction starts as soon as
    that gradient is ready. This overlaps backward compute with network comm.
  - In PyTorch DDP this is the default behavior using gradient buckets; you can
    further customize with `register_comm_hook`.
  - Analogy: as soon as a page is done, you send it to be averaged while you
    keep writing the next page.
- Gradient bucketing (reduce overhead and copies):
  - Group many small grads into larger contiguous buffers (“buckets”) to avoid
    per‑tensor kernel and NCCL call overhead. Reduces small memcpy and launch
    costs; improves overlap.
  - Tune bucket size (`bucket_cap_mb`) to match network and compute balance; too
    small = overhead; too large = late overlap.
  - Keep bucket order aligned with backward order so reductions start early; in
    PyTorch DDP use `static_graph=True` when the graph doesn’t change to avoid
    re‑bucketing every step.
- Fewer copies and contiguity:
  - Use contiguous flat buffers for params/grads (DDP does this); reduces extra
    copies for comm. Consider `gradient_as_bucket_view=True`.
- Skip sync when accumulating:
  - Wrap accumulation steps in `with model.no_sync(): ...`; only sync on the last
    micro‑batch each optimizer step.
- Optional compression hooks:
  - `register_comm_hook` can quantize grads (e.g., 8‑bit) before all‑reduce to
    save bandwidth at the cost of potential noise.

Key formulas and tips
- Batch sizing with DP and accumulation: global_batch = micro_batch × grad_acc ×
  dp_degree.
- Prefer increasing DP before grad_acc when possible (DP is parallel; grad_acc
  is sequential). Use grad_acc to hit exact batch sizes when GPUs are limited.
- At very large DP (≥512) the ring latency/bandwidth becomes visible—overlap
  breaks down; combine with TP/PP to keep scaling.

## zero: ze ro redundancy optimizer (deepSpeed/fsdp)
- Motivation: in DP, each rank holds full params, grads, and optimizer states →
  large redundancy. ZeRO shards these across DP ranks.
- Stages:
  - ZeRO‑1: shard optimizer states.
  - ZeRO‑2: shard optimizer states + gradients (reduce‑scatter during backward).
  - ZeRO‑3 (aka FSDP): shard optimizer states + gradients + parameters
    (all‑gather weights just‑in‑time, prefetch next layer; discard when done).
- Communication cost rises from ZeRO‑1 → ZeRO‑2 → ZeRO‑3, but you save memory.
- Activations are not sharded by ZeRO; use recomputation/accumulation/TP/SP/CP
  to reduce activation memory.
- Practical tips:
  - Enable prefetch (gather next layer’s weights while computing current layer).
  - Keep DP within a few hundred ranks for good overlap (rule of thumb ≤512).
  - Measure end‑to‑end throughput, not only MFU/HFU.

Step‑by‑step improvements
- Start: naive DP duplicates params, grads, optimizer states on every rank.
- ZeRO‑1: shard Adam moments across ranks; no change to grad/param layout.
- ZeRO‑2: additionally shard gradients; use reduce‑scatter in backward and
  all‑gather for optimizer updates; less memory, modest extra comm.
- ZeRO‑3/FSDP: shard parameters too; just‑in‑time all‑gather per layer with
  prefetch to overlap comm with compute; discard after use; biggest memory win.
- Offload variants: optionally offload optimizer states (and even params) to CPU
  or NVMe for extreme memory pressure; expect slower steps.

Analogy
- Think of a class set of textbooks: instead of everyone buying full sets (naive
  DP), you split chapters among students (ZeRO‑1/2), and for ZeRO‑3 you borrow
  the needed chapter right before the lesson and return it after.

## tensor parallelism (tp)
- Shard linear layers across GPUs by rows or columns.
  - Column‑parallel: broadcast inputs, split weights by columns, all‑gather
    outputs.
  - Row‑parallel: scatter inputs, split weights by rows, all‑reduce outputs.
- In transformers:
  - MLP: column‑parallel first, then row‑parallel to avoid an extra sync.
  - Attention: split Q/K/V heads by columns; output projection row‑parallel.
- Pros: shards activations and weights; reduces per‑GPU memory substantially.
- Cons: needs high‑bandwidth links; hard to overlap comm; best kept within a
  node (e.g., TP ≤ GPUs per node like 8). Cross‑node TP often hurts throughput.
- Head count limits TP degree (need whole heads per rank; stricter with GQA/MQA).

Step‑by‑step improvements
- Naive split order: row‑then‑column forces an extra sync in the middle.
- Better: column‑parallel first, then row‑parallel; you skip an intermediate
  all‑reduce.
- Preallocate comm buffers for AG/AR to avoid per‑op allocation overhead.
- Keep TP within a node (NVLink/NVSwitch) for bandwidth; cross‑node TP often
  drops utilization sharply.
- Sequence Parallel (below) complements TP by sharding the parts TP leaves full.

Analogy
- Assembly line: each station (GPU) builds a slice of a wide part; you only bolt
  them together at the end rather than between every sub‑step.

### sequence parallelism (sp)
- Complement to TP: shard along the sequence dimension for the parts not already
  sharded by TP (LayerNorm, residuals, etc.).
- Replaces some all‑reduce ops with reduce‑scatter/all‑gather pairs; comm volume
  is similar to TP, but activation memory drops further.
- Great for longer sequences; still bandwidth‑bound; still best within a node.

Step‑by‑step improvements
- Naive: keep full‑sequence activations through non‑TP ops (LN, residuals) and
  run an all‑reduce later—wastes memory.
- Better: reduce‑scatter to sharded‑sequence right after TP row‑linear exit; run
  SP region on sharded sequences; all‑gather only when needed.
- Ensure LN/bias grads are all‑reduced across TP ranks in backward (small cost).

Analogy
- Instead of passing the full book around, pass only your chapter slices between
  stations; only compile the full book at the very end when required.

## context parallelism (cp)
- Shard the full model along the sequence dimension, including TP regions.
- Most layers (MLP, LayerNorm) are independent per token and need no extra comm.
- Attention needs communication because tokens attend to all previous tokens.
- Use ring attention to stream K/V between ranks while computing on local Q.
- CP is valuable for very long sequences (128k+), even with full recomputation.
- Combine CP with TP/SP and DP. Consider CP across nodes; keep TP within nodes.

### discovering ring attention (and zig‑zag variants)
- Each GPU holds a slice of the sequence (its K/V and Q subset).
- GPUs exchange K/V slices in a ring, computing partial attention as data arrives
  (overlap comm + compute). Repeat until all slices are processed.
- Zig‑zag variants balance compute across devices to reduce idle time.

Step‑by‑step improvements
- Naive: all‑gather all K/V across GPUs before attention—prohibitive memory and
  comm cost.
- Better: ring attention—stream K/V slices around; compute partial scores/output
  per slice; overlap send/recv with compute; keep memory bounded.
- Advanced: zig‑zag/2‑ring schemes to balance compute and link utilization; tune
  chunk sizes to match network bandwidth and matmul time.

Analogy
- Four cooks sharing ingredients: instead of each cook fetching everything, you
  pass baskets around the table while chopping; by the time the next basket
  arrives you’re ready to chop again.

## pipeline parallelism (pp)
- Split layers across GPUs as stages; micro‑batches flow through the pipeline.
- Pipeline bubble: idle time while the pipe fills/drains. Bubble fraction ≈
  (p−1)/m where p=stages, m=micro‑batches in flight.
- Schedules:
  - AFAB (all‑forward‑all‑backward): simple, but stores many activations → high
    memory.
  - 1F1B (one‑forward‑one‑backward): start backward early; store ~p activations
    only; same bubble as AFAB; more complex scheduling.
  - Interleaved stages (v>1 chunks per GPU): reduces bubble to ≈ (p−1)/(v·m), at
    cost of more comm and more complex scheduling (depth‑ vs breadth‑first).
  - Zero‑bubble/DualPipe: advanced schedules that nearly eliminate the bubble
    (complex; used in state‑of‑the‑art systems).
- PP often scales better than TP across nodes (less bandwidth requirement).
- Combine PP with TP=8 within nodes, and DP/ZeRO on top.

Step‑by‑step improvements
- Naive split: divide layers across GPUs; push one micro‑batch through end‑to‑
  end—huge bubble (most GPUs idle).
- AFAB: run all forwards then all backwards; easy to implement; but stores
  activations for all micro‑batches → memory heavy.
- 1F1B: begin backward as soon as possible; stores only ~p micro‑batches; same
  bubble, but now you can crank m higher to shrink bubble.
- Interleaved (v>1): split each GPU’s layers into v chunks; alternates chunks so
  the bubble shrinks by ~v; extra comm/scheduling complexity.
- Zero‑bubble/DualPipe: sophisticated scheduling to nearly remove bubble; more
  engineering; best at scale.

Analogy
- A multi‑station car wash: with no scheduling only one car moves while others
  wait; AFAB runs all cars through wash then all through dry; 1F1B starts drying
  as soon as the first car exits wash; interleaving adds more smaller stations to
  keep everyone busy.

## expert parallelism (ep)
- For Mixture‑of‑Experts (MoE): shard experts across GPUs; route tokens with
  all‑to‑all.
- Pros: huge capacity with sparse activation per token; good scaling for large
  expert counts.
- Cons: routing adds comm; needs MoE architecture and balancing strategies.

Step‑by‑step improvements
- Naive dense MLP: every token goes through the same big FFN—expensive.
- MoE with top‑k routing: each token picks k experts; compute only those paths;
  capacity factors prevent overload; introduce load‑balancing loss to spread
  traffic.
- Distributed experts (EP): place experts across GPUs; use all‑to‑all to route
  token batches; overlap routing with local expert compute.
- Tune: adjust top‑k (1 vs 2), capacity factor, and expert parallel group size;
  keep token packing contiguous to minimize copies.

Analogy
- A hospital triage: instead of sending every patient to every specialist, route
  to the two best specialists; clinics (experts) are in different buildings, so
  you schedule shuttles (all‑to‑all) while doctors are already seeing patients.

## 5d parallelism in a nutshell (how parts fit together)
- DP: shard by batch; sync grads with all‑reduce; simple; limited by comm at
  large world sizes.
- TP/SP: shard weights/activations along hidden/sequence; bandwidth‑heavy; keep
  within node.
- CP: shard activations across sequence globally; attention needs ring comm.
- PP: shard layers into stages; watch bubble; interleave to reduce bubble.
- EP: shard experts; route tokens via all‑to‑all.
- ZeRO: shard states across DP ranks; does not shard activations.
- Combine to fit memory first, then to reach target batch, then to maximize
  throughput given network/GPU limits.

## finding the best training configuration

### step 1: fit a training step in memory
- GPU‑rich (many GPUs):
  - <10B params: single technique often enough (TP or ZeRO‑3 with full recompute
    on up to 8 GPUs).
  - 10B–100B: mix TP=8 with PP or with DP (ZeRO‑3), or pure ZeRO‑3.
  - 512+ GPUs: pure DP/ZeRO‑3 becomes inefficient; combine with TP or PP.
  - 1024+ GPUs: a common recipe is TP=8 (intra‑node) + DP (ZeRO‑2) + PP.
  - Very long sequences: add CP (often across nodes). With MoE: add EP.
- GPU‑poor (few GPUs):
  - Enable full recomputation; increase grad accumulation; consider ZeRO‑3.

### step 2: reach the target global batch size
- Increase DP or grad accumulation; for long sequences also scale CP.
- If batch is too large (validation issues, instability), reduce DP and/or CP.

### step 3: optimize throughput
- Keep TP within nodes to use high‑bandwidth links (NVLink/NVSwitch).
- Balance PP stages; use interleaving if helpful; measure the bubble.
- Overlap comm with compute (prefetch weights; overlap all‑reduce with backward).
- Use fused kernels and FlashAttention; avoid host↔device syncs.
- Track MFU/HFU, tokens/sec, idle time in traces; validate overlap actually
  happens.

Decision checklist (quick rules)
- Fit memory first: combine ZeRO‑3, recomputation, TP/SP; add CP for long
  sequences; add PP when model depth is large and TP hits bandwidth limits.
- Then hit the batch: push DP up to where comm still overlaps; fill the rest
  with grad accumulation; consider CP to raise batch tokens without OOM.
- Then go fast: keep TP intra‑node; use 1F1B or interleaved PP to reduce bubble;
  tune DDP bucket size and number; preallocate comm buffers; use FlashAttn.
- Always validate with traces; change one variable at a time and log tokens/sec.

### benchmarking at scale — lessons
- Thousands of configs show sharp drops when TP crosses nodes; prefer TP≤8.
- DP scales well until ring latency dominates; expect diminishing returns past
  a few hundred ranks.
- Interleaved PP reduces bubble but adds comm; gains depend on micro‑batches.
- SP meaningfully increases max batch/sequence, with similar comm volume to TP.
- ZeRO‑3 enables giant models but adds gather/scatter traffic; prefetch is key.

## diving into gpus: fusing, threading, mixing

### gpu primer (very short)
- Hierarchy: registers/shared memory (fast/small) → L2 → HBM (slow/large).
- Threads form warps (32), warps form blocks, blocks run on SMs. Occupancy and
- memory access patterns drive performance.

### writing faster kernels without writing cuda
- torch.compile captures graphs and emits triton kernels; often big wins with a
  one‑line decorator.
- Inspect generated Triton with TORCH_LOGS=output_code; refine if needed.
- If still not enough, write Triton directly; drop to CUDA for full control.

### memory coalescing, shared memory, tiling, thread coarsening
- Coalesce global memory loads/stores so threads in a warp touch consecutive
  addresses; reduces memory transactions.
- Use shared memory for reused data; tile to fit into shared memory; adjust
  block sizes for occupancy and reuse.
- Thread coarsening: have each thread do more work if it improves reuse and
  reduces overhead.

### fused kernels
- Combine multiple pointwise ops into one kernel to avoid bouncing data between
  SM and HBM.
- Reduces launch overhead and global memory traffic; great for LayerNorm chains
  and activation stacks.

### flash attention (v1–v3)
- Compute attention in tiles in shared memory; avoid materializing large S/P
  matrices in HBM.
- Big speedups and lower memory; now the default for transformer attention.
- Newer versions reduce non‑matmul ops, tune for Hopper, and support FP8 well.

### mixed precision training
- Formats: FP32, FP16, BF16, FP8 (e4m3/e5m2). BF16 keeps FP32 range (bigger
  exponent), FP16 keeps more mantissa but smaller range.
- Common practice: BF16 compute + FP32 master weights (and often FP32 grads).
- Benefits: higher throughput, lower memory bandwidth pressure. Risks: numerical
  issues if ranges are exceeded (more so with FP16/FP8).
- For FP8 training: requires scaling/calibration (per‑tensor/channel/group), and
  careful kernel support; can deliver large throughput gains on H100+.

Step‑by‑step improvements
- Start with FP32: stable but slow and memory heavy.
- Move to BF16 compute + FP32 master weights (and often FP32 grads): fast, wide
  dynamic range; the 2025 default.
- Add loss scaling if using FP16 to avoid underflow.
- For FP8: add per‑tensor/channel scaling and calibration windows; ensure kernels
  and attention paths are FP8‑aware; monitor divergence early.

Analogy
- Measuring with different rulers: FP32 is a long, precise ruler; BF16 is a
  shorter ruler with more range marks where you need them; FP8 is compact but
  needs careful rescaling to avoid reading errors.

## conclusion
- Recipe in short:
  1) Make it fit: recomputation, ZeRO‑3, TP/SP/CP, PP as needed.
  2) Hit the batch: DP and/or grad accumulation (and CP for long contexts).
  3) Go fast: keep TP intra‑node; overlap comm; reduce bubble; fuse ops; use
     FlashAttention and mixed precision.
- Measure tokens/sec, utilization, and comm overlap; adjust based on traces.
- Expect to iterate. Hardware, network, and model shape dictate the mix.

## appendix highlights

### a0: parallel programming crash course
- Collectives and patterns you will see: barrier, broadcast, reduce,
  all‑reduce, gather, all‑gather, scatter, reduce‑scatter, all‑to‑all.
- Ring all‑reduce = reduce‑scatter + all‑gather; bandwidth‑optimal, latency‑
  sensitive at large world sizes.

### a1: distributed training profiling
- Use Nsight Systems/Compute, PyTorch profiler. Look for overlapped streams,
  comm streams, kernel timelines, and memory usage over a step.
- Validate assumptions (e.g., grad all‑reduce overlap) with traces, not guesses.

### a2: typical scales
- Modern pretraining: batch tokens in the millions; sequence lengths commonly
  2–8k during main training; later phases may add longer sequences.

### a3: math for overlap
- Overlap works when compute time per bucket exceeds comm latency/bandwidth
  time; increase bucket size, fuse ops, or reduce world size to regain overlap.

### references
- Original playbook: https://huggingface.co/spaces/nanotron/ultrascale-playbook
- See also: DeepSpeed ZeRO, Megatron‑LM, PyTorch FSDP, FlexAttention,
  torchtitan, ColossalAI, and the cited papers.
