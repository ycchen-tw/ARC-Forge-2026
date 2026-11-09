// Host-mapped collectives for GPUs without working P2P (NCCL would use SHM).
//
// Every rank maps one shared, pinned host region. A collective is one kernel:
// each block copies its chunk into this rank's host slot, publishes a per-block
// flag (system-scope release), waits for the same block's flag from all peers,
// then reads the peers' chunks back over PCIe. No grid-wide sync, no host thread,
// graph-capturable. Reduction order is rank order on every rank, so all ranks
// produce bitwise-identical outputs. A stuck peer traps after a timeout.
//
// Payload kinds: bf16 (exact data movement) or fp8 e4m3 with one fp32 scale per
// 128 elements (quantized on write, dequantized + fp32-accumulated on read).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cassert>
#include <cstdio>
#include <cstring>

// Spin timeout before __trap(). Default ~10 s at 2 GHz; a cold Triton JIT on one DP can stall its ranks
// longer than that while the other DP waits in a TP4 collective (Kaggle v15 v4), so builds may raise it.
#ifndef HC_TIMEOUT_CYCLES
#define HC_TIMEOUT_CYCLES 20000000000LL
#endif

// Optional back-off between flag polls (ns, __nanosleep). 0 = spin (the original behaviour). Every waiting thread
// issues one system-scope PCIe read per poll; a back-off trades wake-up latency for less read traffic on the links
// the payload shares.
#ifndef HC_POLL_NS
#define HC_POLL_NS 0
#endif

#define MAX_RANKS 4
#define QBLOCK 128

struct Layout {
  // host (mapped) pointers, already device-addressable under UVA
  uint32_t* flags;   // [2 parity][MAX_RANKS][max_blocks] (32-byte stride)
  uint8_t* data;     // [2 parity][MAX_RANKS][slot_bytes]
  uint8_t* data2;    // [2 parity][MAX_RANKS][slot_bytes] second stage (two-shot)
  uint32_t* flags2;  // like flags, second stage
  int64_t slot_bytes;
  int max_blocks;
  // per-region options (hc_region_opts, looked up by flags in region_opts()):
  uint32_t* done = nullptr;             // device arrival counter: the kernels that support it advance the epoch
                                        // themselves (last block to finish) instead of the separate bump kernel
  unsigned long long* trace = nullptr;  // HC_TRACE builds: per-call, per-block phase timestamps (globaltimer ns)
};

// Per-region options, keyed by the region's flags pointer (one HostComm = one region). Host-side table: entry points
// copy the options into the Layout they pass by value.
struct RegionOpts {
  void* flags;
  uint32_t* done;
  unsigned long long* trace;
};
static RegionOpts g_region_opts[16];
static int g_region_opts_n = 0;

static void region_opts(Layout& L) {
  for (int i = 0; i < g_region_opts_n; ++i)
    if (g_region_opts[i].flags == (void*)L.flags) {
      L.done = g_region_opts[i].done;
      L.trace = g_region_opts[i].trace;
      return;
    }
}

// HC_TRACE: TRACE_CALLS calls (epoch mod) x TRACE_BLOCKS blocks x 8 timestamps per region trace buffer
#define TRACE_CALLS 512
#define TRACE_BLOCKS 384
#ifdef HC_TRACE
__device__ __forceinline__ void trace_at(const Layout& L, uint32_t e, int k) {
  if (L.trace == nullptr || threadIdx.x != 0 || blockIdx.x >= TRACE_BLOCKS) return;
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  L.trace[((int64_t)(e % TRACE_CALLS) * TRACE_BLOCKS + blockIdx.x) * 8 + k] = t;
}
#else
__device__ __forceinline__ void trace_at(const Layout&, uint32_t, int) {}
#endif

// Folded epoch advance (L.done != nullptr): the last block to arrive advances *epoch (and the optional per-call
// counter ctr). Every block read *epoch at its start, before its arrival, so no block of this call sees the new
// value; the next kernel in the stream does. Every block of a kernel that folds calls this exactly once, last.
__device__ __forceinline__ void finish_epoch(const Layout& L, const uint32_t* epoch, uint32_t e,
                                             unsigned long long* ctr) {
  if (L.done == nullptr) return;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (atomicAdd(L.done, 1u) == gridDim.x - 1) {
      *L.done = 0;
      *const_cast<uint32_t*>(epoch) = e + 1;
      if (ctr != nullptr) *ctr += 1;
    }
  }
}

__device__ __forceinline__ uint32_t* flag_at(uint32_t* base, const Layout& L, int parity, int rank, int block) {
  return base + (((int64_t)parity * MAX_RANKS + rank) * L.max_blocks + block) * 8;
}
__device__ __forceinline__ uint8_t* slot_at(uint8_t* base, const Layout& L, int parity, int rank) {
  return base + ((int64_t)parity * MAX_RANKS + rank) * L.slot_bytes;
}

__device__ __forceinline__ void publish(uint32_t* f, uint32_t tag) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(f), "r"(tag) : "memory");
  }
}

__device__ __forceinline__ void wait_peers(uint32_t* base, const Layout& L, int parity, int rank, int nranks,
                                           int block, uint32_t tag, unsigned mask = 0xffffffffu) {
  if (threadIdx.x < nranks && threadIdx.x != rank && ((mask >> threadIdx.x) & 1u)) {
    uint32_t* f = flag_at(base, L, parity, threadIdx.x, block);
    long long start = clock64();
    uint32_t v;
    while (true) {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
      if (v == tag) break;
      if (HC_POLL_NS > 0) __nanosleep(HC_POLL_NS);
      if (clock64() - start > (long long)HC_TIMEOUT_CYCLES) {
        printf("hostcomm: timeout rank %d waiting peer %d block %d tag %u saw %u\n", rank, threadIdx.x, block, tag, v);
        __trap();
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ int4 ld_cv(const int4* p) { return __ldcv(p); }
__device__ __forceinline__ void st_wt(int4* p, int4 v) { __stwt(p, v); }

// ---------------- bf16 ----------------
// out may alias x for all-reduce.
__global__ void ar_bf16_oneshot(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, Layout L, int rank, int nranks,
                                const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  // chunk of 8-element vectors per block
  const int64_t nvec = n / 8;
  const int64_t per = (nvec + gridDim.x - 1) / gridDim.x;
  const int64_t lo = per * blockIdx.x, hi = min(nvec, lo + per);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  const int4* xv = reinterpret_cast<const int4*>(x);
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) st_wt(mine + i, xv[i]);
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, nranks, blockIdx.x, tag);
  int4* ov = reinterpret_cast<int4*>(out);
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < MAX_RANKS; ++j) {
      if (j >= nranks) break;
      int4 v = (j == rank) ? xv[i] : ld_cv(reinterpret_cast<const int4*>(slot_at(L.data, L, parity, j)) + i);
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
    }
    int4 r;
    __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
    for (int k = 0; k < 8; ++k) rb[k] = __float2bfloat16(acc[k]);
    ov[i] = r;
  }
}

// Two-shot: stage 1 every rank writes x; rank r reduces sub-chunk r of each block
// chunk and writes it to stage-2 slot; stage 2 everyone reads the reduced sub-chunks.
__global__ void ar_bf16_twoshot(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, Layout L, int rank, int nranks,
                                const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int64_t nvec = n / 8;
  const int64_t per = (nvec + gridDim.x - 1) / gridDim.x;
  const int64_t lo = per * blockIdx.x, hi = min(nvec, lo + per);
  const int64_t len = max((int64_t)0, hi - lo);
  const int64_t sub = (len + nranks - 1) / nranks;
  const int4* xv = reinterpret_cast<const int4*>(x);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  // stage 1: publish only the sub-chunks other ranks reduce
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    if ((i - lo) / sub != rank) st_wt(mine + i, xv[i]);
  }
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, nranks, blockIdx.x, tag);
  const int64_t slo = lo + sub * rank, shi = min(hi, slo + sub);
  int4* red = reinterpret_cast<int4*>(slot_at(L.data2, L, parity, rank));
  int4* ov = reinterpret_cast<int4*>(out);
  for (int64_t i = slo + threadIdx.x; i < shi; i += blockDim.x) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < MAX_RANKS; ++j) {
      if (j >= nranks) break;
      int4 v = (j == rank) ? xv[i] : ld_cv(reinterpret_cast<const int4*>(slot_at(L.data, L, parity, j)) + i);
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
    }
    int4 r;
    __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
    for (int k = 0; k < 8; ++k) rb[k] = __float2bfloat16(acc[k]);
    st_wt(red + i, r);
    ov[i] = r;
  }
  publish(flag_at(L.flags2, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags2, L, parity, rank, nranks, blockIdx.x, tag);
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const int owner = (int)((i - lo) / sub);
    if (owner != rank) ov[i] = ld_cv(reinterpret_cast<const int4*>(slot_at(L.data2, L, parity, owner)) + i);
  }
}

// All-gather: out = concat over ranks of x (each n elements).
__global__ void ag_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, Layout L, int rank, int nranks,
                        const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int64_t nvec = n / 8;
  const int64_t per = (nvec + gridDim.x - 1) / gridDim.x;
  const int64_t lo = per * blockIdx.x, hi = min(nvec, lo + per);
  const int4* xv = reinterpret_cast<const int4*>(x);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  int4* ov = reinterpret_cast<int4*>(out);
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    int4 v = xv[i];
    st_wt(mine + i, v);
    ov[rank * nvec + i] = v;
  }
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, nranks, blockIdx.x, tag);
  for (int j = 0; j < nranks; ++j) {
    if (j == rank) continue;
    const int4* src = reinterpret_cast<const int4*>(slot_at(L.data, L, parity, j));
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) ov[j * nvec + i] = ld_cv(src + i);
  }
}

// ---------------- fp8 (e4m3, per-128 scale) ----------------
// Slot layout per rank: [nq bytes payload][nq/128 floats scales].
// A warp quantizes 128 elements (4 per lane): amax via shuffles.
__device__ __forceinline__ void quant128(const __nv_bfloat16* src, uint8_t* qdst, float* sdst, int lane) {
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = __bfloat162float(src[lane * 4 + k]);
  float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, o));
  const float scale = fmaxf(a, 1e-12f) / 448.0f;
  const float inv = 1.0f / scale;
  uint32_t packed = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    __nv_fp8_e4m3 q(v[k] * inv);
    packed |= (uint32_t)(*reinterpret_cast<uint8_t*>(&q)) << (8 * k);
  }
  __stwt(reinterpret_cast<unsigned int*>(qdst) + lane, packed);
  if (lane == 0) __stwt(sdst, scale);
}

__device__ __forceinline__ void dequant128_add(const uint8_t* qsrc, const float* ssrc, float acc[4], int lane) {
  const uint32_t packed = __ldcv(reinterpret_cast<const unsigned int*>(qsrc) + lane);
  const float scale = __ldcv(ssrc);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    __nv_fp8_e4m3 q;
    *reinterpret_cast<uint8_t*>(&q) = (packed >> (8 * k)) & 0xff;
    acc[k] += float(q) * scale;
  }
}

// One-shot fp8 all-reduce. Own contribution is also quantized (so every rank sums
// identical values in identical order -> bitwise-identical outputs).
__global__ void ar_fp8_oneshot(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, Layout L, int rank, int nranks,
                               const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int64_t ng = n / QBLOCK;  // groups of 128
  const int64_t per = (ng + gridDim.x - 1) / gridDim.x;
  const int64_t lo = per * blockIdx.x, hi = min(ng, lo + per);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  uint8_t* q = slot_at(L.data, L, parity, rank);
  float* s = reinterpret_cast<float*>(q + n);
  for (int64_t g = lo + warp; g < hi; g += nw) quant128(x + g * QBLOCK, q + g * QBLOCK, s + g, lane);
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, nranks, blockIdx.x, tag);
  for (int64_t g = lo + warp; g < hi; g += nw) {
    float acc[4] = {0, 0, 0, 0};
    for (int j = 0; j < nranks; ++j) {
      const uint8_t* qj = slot_at(L.data, L, parity, j);
      dequant128_add(qj + g * QBLOCK, reinterpret_cast<const float*>(qj + n) + g, acc, lane);
    }
    __nv_bfloat16* o = out + g * QBLOCK + lane * 4;
#pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = __float2bfloat16(acc[k]);
  }
}

// fp8 all-gather (lossy).
__global__ void ag_fp8(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, Layout L, int rank, int nranks,
                       const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int64_t ng = n / QBLOCK;
  const int64_t per = (ng + gridDim.x - 1) / gridDim.x;
  const int64_t lo = per * blockIdx.x, hi = min(ng, lo + per);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  uint8_t* q = slot_at(L.data, L, parity, rank);
  float* s = reinterpret_cast<float*>(q + n);
  for (int64_t g = lo + warp; g < hi; g += nw) quant128(x + g * QBLOCK, q + g * QBLOCK, s + g, lane);
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, nranks, blockIdx.x, tag);
  for (int j = 0; j < nranks; ++j) {
    const uint8_t* qj = slot_at(L.data, L, parity, j);
    for (int64_t g = lo + warp; g < hi; g += nw) {
      float acc[4] = {0, 0, 0, 0};
      dequant128_add(qj + g * QBLOCK, reinterpret_cast<const float*>(qj + n) + g, acc, lane);
      __nv_bfloat16* o = out + (int64_t)j * n + g * QBLOCK + lane * 4;
#pragma unroll
      for (int k = 0; k < 4; ++k) o[k] = __float2bfloat16(acc[k]);
    }
  }
}

// ---------------- vectorized fp8 (16 elements / thread, 8 threads / 128-group) ----------------
// Payload: n bytes (item i -> bytes [16i,16i+16)); scales: n/128 floats after payload.
__device__ __forceinline__ void quant16(const __nv_bfloat16* src, int4& q, float& scale) {
  const int4* s = reinterpret_cast<const int4*>(src);
  int4 a = s[0], b = s[1];
  const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&a);
  const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&b);
  float v[16];
#pragma unroll
  for (int k = 0; k < 8; ++k) { v[k] = __bfloat162float(pa[k]); v[8 + k] = __bfloat162float(pb[k]); }
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) m = fmaxf(m, fabsf(v[k]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
  scale = fmaxf(m, 1e-12f) / 448.0f;
  const float inv = 1.0f / scale;
  __nv_fp8x2_storage_t* qq = reinterpret_cast<__nv_fp8x2_storage_t*>(&q);
#pragma unroll
  for (int k = 0; k < 8; ++k) qq[k] = __nv_cvt_float2_to_fp8x2(make_float2(v[2 * k] * inv, v[2 * k + 1] * inv), __NV_SATFINITE, __NV_E4M3);
}

// quant16 on fp32 inputs (the stage-2 FP8 pair exchange quantizes the owner's fp32 sums): same scale rule
__device__ __forceinline__ void quant16f(const float v[16], int4& q, float& scale) {
  float m = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) m = fmaxf(m, fabsf(v[k]));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
  m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
  scale = fmaxf(m, 1e-12f) / 448.0f;
  const float inv = 1.0f / scale;
  __nv_fp8x2_storage_t* qq = reinterpret_cast<__nv_fp8x2_storage_t*>(&q);
#pragma unroll
  for (int k = 0; k < 8; ++k) qq[k] = __nv_cvt_float2_to_fp8x2(make_float2(v[2 * k] * inv, v[2 * k + 1] * inv), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ void dequant16_add(const int4& q, float scale, float acc[16]) {
  const __nv_fp8x2_storage_t* qq = reinterpret_cast<const __nv_fp8x2_storage_t*>(&q);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(qq[k], __NV_E4M3);
    float2 f = __half22float2(*reinterpret_cast<__half2*>(&h));
    acc[2 * k] += f.x * scale;
    acc[2 * k + 1] += f.y * scale;
  }
}

__device__ __forceinline__ void store16(__nv_bfloat16* dst, const float acc[16]) {
  int4 a, b;
  __nv_bfloat16* pa = reinterpret_cast<__nv_bfloat16*>(&a);
  __nv_bfloat16* pb = reinterpret_cast<__nv_bfloat16*>(&b);
#pragma unroll
  for (int k = 0; k < 8; ++k) { pa[k] = __float2bfloat16(acc[k]); pb[k] = __float2bfloat16(acc[8 + k]); }
  reinterpret_cast<int4*>(dst)[0] = a;
  reinterpret_cast<int4*>(dst)[1] = b;
}

// mode 0: one-shot all-reduce (out = sum of dequantized fp8 of all ranks, own included)
// mode 1: all-gather (out[j*n..] = dequant(rank j))
// mode 2: mode 1 that also keeps the payload: qout[j*n..] = rank j's e4m3 bytes, sout[j*n/128..] = its scales (an
//         FP8 GEMM can take the gathered rows as they travelled, ARC_HC_AG_KIND=ag_fp8q)
// P2P (mode 0, an attention pair over CUDA peer access, PairP2P): L.data / L.flags are this rank's DEVICE region
// (cudaMalloc, IPC-shared; same [parity][rank] layout as the host region) and pdata / pflags the sibling's IPC
// mapping. A rank writes its payload and flag into the sibling's region at its own rank index (posted peer writes:
// the bytes go up and down the PCIe link at once, no host staging) and waits and reads in its own region (local
// memory). The reduction is this kernel's, so the sum is bitwise ar1_fp8v. Two-parity reuse as the host kernels: a
// rank writes the sibling's parity-p slot in call k+2 only after call k+1's wait, i.e. after the sibling finished
// call k (its reads of that slot).
template <int MODE, bool P2P = false>
__global__ void fp8v_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, Layout L, int rank, int nranks,
                            const uint32_t* epoch, uint8_t* qout, float* sout, uint8_t* pdata = nullptr,
                            uint32_t* pflags = nullptr) {
  static_assert(!P2P || MODE == 0, "P2P is the pair all-reduce");
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  trace_at(L, e, 0);
  const int64_t items = n / 16;
  int64_t per = (items + gridDim.x - 1) / gridDim.x;
  per = (per + 7) / 8 * 8;
  const int64_t lo = per * blockIdx.x, hi = min(items, lo + per);
  uint8_t* myq = slot_at(P2P ? pdata : L.data, L, parity, rank);
  float* mys = reinterpret_cast<float*>(myq + n);
  const int lane8 = threadIdx.x & 7;
  for (int64_t base = lo; base < max(hi, lo + 1); base += blockDim.x) {
    const int64_t i = base + threadIdx.x;
    const bool valid = i < hi;
    int4 q = make_int4(0, 0, 0, 0);
    float sc = 0.f;
    // all lanes of a warp must reach the shuffles: invalid lanes quantize a dummy zero row
    __align__(16) __nv_bfloat16 zero[16];
    if (valid) quant16(x + i * 16, q, sc);
    else {
#pragma unroll
      for (int k = 0; k < 16; ++k) zero[k] = __float2bfloat16(0.f);
      quant16(zero, q, sc);
    }
    if (valid) {
      st_wt(reinterpret_cast<int4*>(myq) + i, q);
      if (lane8 == 0) __stwt(mys + i / 8, sc);
      if (MODE >= 1) {
        float acc[16] = {0};
        dequant16_add(q, sc, acc);
        store16(out + (int64_t)rank * n + i * 16, acc);
      }
      if (MODE == 2) {
        reinterpret_cast<int4*>(qout + (int64_t)rank * n)[i] = q;
        if (lane8 == 0) sout[(int64_t)rank * (n / QBLOCK) + i / 8] = sc;
      }
    }
    if (base >= hi) break;
  }
  trace_at(L, e, 1);
  publish(flag_at(P2P ? pflags : L.flags, L, parity, rank, blockIdx.x), tag);
  trace_at(L, e, 2);
  wait_peers(L.flags, L, parity, rank, nranks, blockIdx.x, tag);
  trace_at(L, e, 3);
  for (int64_t base = lo; base < hi; base += blockDim.x) {
    const int64_t i = base + threadIdx.x;
    const bool valid = i < hi;
    if (MODE == 0) {
      // Own contribution: requantize from device memory (warp-uniform shuffles), identical
      // to the bytes peers read from our host slot.
      int4 own;
      float own_sc;
      __align__(16) __nv_bfloat16 zero[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) zero[k] = __float2bfloat16(0.f);
      quant16(valid ? x + i * 16 : zero, own, own_sc);
      if (!valid) continue;
      float acc[16] = {0};
      for (int j = 0; j < nranks; ++j) {
        if (j == rank) {
          dequant16_add(own, own_sc, acc);
          continue;
        }
        const uint8_t* qj = slot_at(L.data, L, parity, j);
        int4 q = ld_cv(reinterpret_cast<const int4*>(qj) + i);
        float sc = __ldcv(reinterpret_cast<const float*>(qj + n) + i / 8);
        dequant16_add(q, sc, acc);
      }
      store16(out + i * 16, acc);
    } else {
      if (!valid) break;
      for (int j = 0; j < nranks; ++j) {
        if (j == rank) continue;
        const uint8_t* qj = slot_at(L.data, L, parity, j);
        int4 q = ld_cv(reinterpret_cast<const int4*>(qj) + i);
        float sc = __ldcv(reinterpret_cast<const float*>(qj + n) + i / 8);
        float acc[16] = {0};
        dequant16_add(q, sc, acc);
        store16(out + (int64_t)j * n + i * 16, acc);
        if (MODE == 2) {
          reinterpret_cast<int4*>(qout + (int64_t)j * n)[i] = q;
          if ((threadIdx.x & 7) == 0) sout[(int64_t)j * (n / QBLOCK) + i / 8] = sc;
        }
      }
    }
  }
  trace_at(L, e, 7);
  finish_epoch(L, epoch, e, nullptr);
}

// Reduce-to-local-rows (MoE combine under DP attention): every rank holds a partial
// sum of all G global rows; rank r needs the full sum of rows [start, start+count)
// (its attention-DP rows, read from device scalars so CUDA graphs replay correctly),
// written into out[0:out_rows) with rows >= count zero-filled (dp_scatter semantics).
// Writers publish all their rows; readers wait for every block of every peer.
__device__ __forceinline__ void wait_all_blocks(uint32_t* base, const Layout& L, int parity, int rank, int nranks,
                                                int nblocks, uint32_t tag) {
  for (int t = threadIdx.x; t < nranks * nblocks; t += blockDim.x) {
    const int j = t / nblocks, b = t % nblocks;
    if (j == rank) continue;
    uint32_t* f = flag_at(base, L, parity, j, b);
    long long start = clock64();
    uint32_t v;
    while (true) {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
      if (v == tag) break;
      if (HC_POLL_NS > 0) __nanosleep(HC_POLL_NS);
      if (clock64() - start > (long long)HC_TIMEOUT_CYCLES) {
        printf("hostcomm rs_rows: timeout rank %d peer %d block %d\n", rank, j, b);
        __trap();
      }
    }
  }
  __syncthreads();
}

template <typename IdxT>
__global__ void rs_rows_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t g_rows, int64_t out_rows, int64_t hidden,
                             const IdxT* start_p, const IdxT* count_p, Layout L, int rank, int nranks,
                             const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int64_t vpr = hidden / 8;  // int4 vectors per row
  const int64_t nvec = g_rows * vpr;
  const int64_t per = (nvec + gridDim.x - 1) / gridDim.x;
  const int64_t lo = per * blockIdx.x, hi = min(nvec, lo + per);
  const int4* xv = reinterpret_cast<const int4*>(x);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) st_wt(mine + i, xv[i]);
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_all_blocks(L.flags, L, parity, rank, nranks, gridDim.x, tag);
  const int64_t start = (int64_t)*start_p, count = (int64_t)*count_p;
  const int64_t ovec = out_rows * vpr;
  const int64_t oper = (ovec + gridDim.x - 1) / gridDim.x;
  const int64_t olo = oper * blockIdx.x, ohi = min(ovec, olo + oper);
  int4* ov = reinterpret_cast<int4*>(out);
  for (int64_t i = olo + threadIdx.x; i < ohi; i += blockDim.x) {
    const int64_t row = i / vpr;
    if (row >= count) {
      ov[i] = make_int4(0, 0, 0, 0);
      continue;
    }
    const int64_t src = (start + row) * vpr + (i % vpr);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int j = 0; j < MAX_RANKS; ++j) {
      if (j >= nranks) break;
      int4 v = (j == rank) ? xv[src] : ld_cv(reinterpret_cast<const int4*>(slot_at(L.data, L, parity, j)) + src);
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(b[k]);
    }
    int4 r;
    __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
    for (int k = 0; k < 8; ++k) rb[k] = __float2bfloat16(acc[k]);
    ov[i] = r;
  }
}

// ---------------- DP-aware (4 ranks = 2 attention-DP x attention-TP 2; pairs (0,1), (2,3)) ----------------
// MoE dispatch gather. xfull = this rank's full DP-local rows [2n] (replicated bitwise within the pair after the
// attention all-reduce); out = [4n] = the 4 rank shards in rank order. Own DP's two shards come from xfull, only
// the other DP's two shards are read from host memory (reads 2n instead of 3n). Every rank still publishes its
// shard and waits for all three peers, so the "a kernel passes its wait only after every peer finished the
// previous collective" invariant that makes the two-parity slots safe holds. mismatch != nullptr: also read the
// sibling's shard and count 32-bit words that differ from the local copy (replication check).
__global__ void ag_dp_bf16(const __nv_bfloat16* xfull, __nv_bfloat16* out, int64_t n, Layout L, int rank,
                           const uint32_t* epoch, unsigned long long* mismatch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int sib = rank ^ 1, d = rank >> 1;
  const int64_t nvec = n / 8;
  const int64_t per = (nvec + gridDim.x - 1) / gridDim.x;
  const int64_t lo = per * blockIdx.x, hi = min(nvec, lo + per);
  const int4* xv = reinterpret_cast<const int4*>(xfull);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  int4* ov = reinterpret_cast<int4*>(out);
  const int64_t own_off = (int64_t)(rank & 1) * nvec, sib_off = (int64_t)(sib & 1) * nvec;
  for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    int4 v = xv[own_off + i];
    st_wt(mine + i, v);
    ov[(int64_t)rank * nvec + i] = v;
    ov[(int64_t)sib * nvec + i] = xv[sib_off + i];
  }
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, 4, blockIdx.x, tag);
  for (int j = 0; j < 4; ++j) {
    if ((j >> 1) == d) continue;
    const int4* src = reinterpret_cast<const int4*>(slot_at(L.data, L, parity, j));
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) ov[(int64_t)j * nvec + i] = ld_cv(src + i);
  }
  if (mismatch != nullptr) {
    const int4* src = reinterpret_cast<const int4*>(slot_at(L.data, L, parity, sib));
    unsigned long long bad = 0;
    for (int64_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      int4 a = ld_cv(src + i), b = xv[sib_off + i];
      bad += (a.x != b.x) + (a.y != b.y) + (a.z != b.z) + (a.w != b.w);
    }
    if (bad) atomicAdd(mismatch, bad);
  }
}

// MoE combine. x = this rank's partial sum over all g_rows global rows (DP d at rows [d*Ld, (d+1)*Ld), Ld =
// g_rows/2, MAX_LEN layout). Rank r = 2d+h owns quarter q=r: rows [d*Ld + h*Q0, ...) with Q0 = ceil(Ld/2).
// Stage 1: publish the three quarters the peers own, reduce own quarter in rank order (fp32, like ar_bf16_twoshot,
// so the sums are bitwise identical to AR4). Stage 2: publish the reduced quarter; read only the sibling's.
// out = this DP's Ld rows; rows >= count are zero (dp_scatter semantics). start must equal d*Ld (trap otherwise).
// Per rank: writes S, reads S (S = g_rows*hidden*2 bytes) vs two-shot AR4 S / 1.5S, and no dp_scatter copy.
template <typename IdxT>
__global__ void rs_owner_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t g_rows, int64_t hidden,
                              const IdxT* start_p, const IdxT* count_p, Layout L, int rank, const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int d = rank >> 1, sib = rank ^ 1;
  const int64_t vpr = hidden / 8;
  const int64_t Ld = g_rows / 2, Q0 = (Ld + 1) / 2;
  const int64_t start = (int64_t)*start_p, count = (int64_t)*count_p;
  if (start != d * Ld) {
    if (threadIdx.x == 0) printf("hostcomm rs_owner: rank %d start %lld != %lld (not MAX_LEN layout)\n", rank,
                                 (long long)start, (long long)(d * Ld));
    __trap();
  }
  // quarter q: global vector range [qs(q), qe(q)); every block takes the same relative slice of every quarter
  auto qs = [&](int q) { return ((q >> 1) * Ld + (q & 1) * Q0) * vpr; };
  auto qe = [&](int q) { return ((q >> 1) * Ld + ((q & 1) ? Ld : Q0)) * vpr; };
  const int64_t per = (Q0 * vpr + gridDim.x - 1) / gridDim.x;
  const int64_t rlo = per * blockIdx.x, rhi = rlo + per;
  const int4* xv = reinterpret_cast<const int4*>(x);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  for (int q = 0; q < 4; ++q) {
    if (q == rank) continue;
    const int64_t a = qs(q) + rlo, b = min(qe(q), qs(q) + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) st_wt(mine + i, xv[i]);
  }
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, 4, blockIdx.x, tag);
  int4* red = reinterpret_cast<int4*>(slot_at(L.data2, L, parity, rank));
  int4* ov = reinterpret_cast<int4*>(out);
  const int64_t base = d * Ld * vpr;  // first global vector of this DP
  const int4 zero = make_int4(0, 0, 0, 0);
  {
    const int64_t a = qs(rank) + rlo, b = min(qe(rank), qs(rank) + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) {
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int4 v = (j == rank) ? xv[i] : ld_cv(reinterpret_cast<const int4*>(slot_at(L.data, L, parity, j)) + i);
        const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(bv[k]);
      }
      int4 r;
      __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
      for (int k = 0; k < 8; ++k) rb[k] = __float2bfloat16(acc[k]);
      st_wt(red + i, r);
      ov[i - base] = ((i - base) / vpr < count) ? r : zero;
    }
  }
  publish(flag_at(L.flags2, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags2, L, parity, rank, 4, blockIdx.x, tag);
  {
    const int4* sred = reinterpret_cast<const int4*>(slot_at(L.data2, L, parity, sib));
    const int64_t a = qs(sib) + rlo, b = min(qe(sib), qs(sib) + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x)
      ov[i - base] = ((i - base) / vpr < count) ? ld_cv(sred + i) : zero;
  }
}

// MoE combine with an FP8 first stage (ARC_HC_MOE_RS=owner_fp8). Same quarters, flags and stage 2 as rs_owner_bf16,
// but the three quarters a rank publishes for its peers go out as e4m3 with one fp32 scale per 128 values (quant16,
// as fp8v; slot layout [g_rows*hidden payload bytes][one float per 128]). The owner adds its own quarter exactly
// (bf16) and the peers' dequantized partials in rank order in fp32. Stage 2 (the owner's reduced quarter for the
// sibling) stays bf16, so the two ranks of a pair still hold the same rows bitwise. Stage-1 bytes per rank:
// 3/4 S (bf16) -> 3/8 S + scales; stage 2 S/4 unchanged. hidden % 128 == 0 (a group never crosses a row).
// S2FP8 (ARC_HC_RS_STAGE2=fp8, lossy): the owner quantizes its fp32 sums (quant16f, scale per 128) for the sibling
// and writes the dequantized values as its own rows too, so both ranks of the pair still hold the same bytes; stage 2
// moves S/8 + scales instead of S/4. SIBWAIT (ARC_HC_RS_SIBWAIT=1): stage 2 waits for the sibling's flag only (it
// reads only the sibling's quarter). Safe for the two-parity slots: stage 1 still waits for every peer, so a rank
// leaves a collective only after every peer entered it, and data2 / flags2 of a rank are read by its sibling only.
template <typename IdxT, bool S2FP8, bool SIBWAIT>
__global__ void rs_owner_fp8(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t g_rows, int64_t hidden,
                             const IdxT* start_p, const IdxT* count_p, Layout L, int rank, const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  trace_at(L, e, 0);
  const int d = rank >> 1, sib = rank ^ 1;
  const int64_t ipr = hidden / 16;  // 16-element items per row; 8 items = one 128-group
  const int64_t n = g_rows * hidden;
  const int64_t Ld = g_rows / 2, Q0 = (Ld + 1) / 2;
  const int64_t start = (int64_t)*start_p, count = (int64_t)*count_p;
  if (start != d * Ld) {
    if (threadIdx.x == 0) printf("hostcomm rs_owner_fp8: rank %d start %lld != %lld (not MAX_LEN layout)\n", rank,
                                 (long long)start, (long long)(d * Ld));
    __trap();
  }
  // quarter q: global item range [qs(q), qe(q)); every block takes the same relative slice of every quarter, in
  // whole 128-groups (the 8 lanes of a group reduce its absmax together)
  auto qs = [&](int q) { return ((q >> 1) * Ld + (q & 1) * Q0) * ipr; };
  auto qe = [&](int q) { return ((q >> 1) * Ld + ((q & 1) ? Ld : Q0)) * ipr; };
  int64_t per = (Q0 * ipr + gridDim.x - 1) / gridDim.x;
  per = (per + 7) / 8 * 8;
  const int64_t rlo = per * blockIdx.x, rhi = rlo + per;
  uint8_t* myq = slot_at(L.data, L, parity, rank);
  float* mys = reinterpret_cast<float*>(myq + n);
  __align__(16) __nv_bfloat16 zero16[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) zero16[k] = __float2bfloat16(0.f);
  for (int q = 0; q < 4; ++q) {
    if (q == rank) continue;
    const int64_t a = qs(q) + rlo, b = min(qe(q), qs(q) + rhi);
    // block-uniform trip count: every lane reaches quant16's shuffles; groups are wholly in or out of [a, b)
    for (int64_t base = a; base < b; base += blockDim.x) {
      const int64_t i = base + threadIdx.x;
      int4 qv;
      float sc;
      quant16(i < b ? x + i * 16 : zero16, qv, sc);
      if (i < b) {
        st_wt(reinterpret_cast<int4*>(myq) + i, qv);
        if ((i & 7) == 0) __stwt(mys + i / 8, sc);
      }
    }
  }
  trace_at(L, e, 1);
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  trace_at(L, e, 2);
  wait_peers(L.flags, L, parity, rank, 4, blockIdx.x, tag);
  trace_at(L, e, 3);
  int4* red = reinterpret_cast<int4*>(slot_at(L.data2, L, parity, rank));  // bf16, indexed like x
  uint8_t* red8 = slot_at(L.data2, L, parity, rank);                      // S2FP8: e4m3 by item, then scales
  float* reds = reinterpret_cast<float*>(red8 + n);
  int4* ov = reinterpret_cast<int4*>(out);
  const int64_t vpr = hidden / 8;
  const int64_t base_v = d * Ld * vpr;  // first 8-element vector of this DP
  const int4 zero = make_int4(0, 0, 0, 0);
  {
    const int64_t a = qs(rank) + rlo, b = min(qe(rank), qs(rank) + rhi);
    // S2FP8: block-uniform trip count (quant16f's shuffles; a and b are whole 128-groups)
    for (int64_t ib = a; S2FP8 ? ib < b : ib + threadIdx.x < b; ib += blockDim.x) {
      const int64_t i = ib + threadIdx.x;
      const bool valid = i < b;
      float acc[16] = {0};
      if (valid) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if (j == rank) {
            const int4* xv = reinterpret_cast<const int4*>(x) + 2 * i;
            int4 v0 = xv[0], v1 = xv[1];
            const __nv_bfloat16* b0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
            const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
              acc[k] += __bfloat162float(b0[k]);
              acc[8 + k] += __bfloat162float(b1[k]);
            }
          } else {
            const uint8_t* qj = slot_at(L.data, L, parity, j);
            const int4 qv = ld_cv(reinterpret_cast<const int4*>(qj) + i);
            const float sc = __ldcv(reinterpret_cast<const float*>(qj + n) + i / 8);
            dequant16_add(qv, sc, acc);
          }
        }
      }
      const bool live = (2 * i - base_v) / vpr < count;
      if (S2FP8) {
        int4 qv;
        float sc;
        quant16f(acc, qv, sc);
        if (valid) {
          st_wt(reinterpret_cast<int4*>(red8) + i, qv);
          if ((i & 7) == 0) __stwt(reds + i / 8, sc);
          // own rows = the dequantized payload, what the sibling computes from the slot. Dequantized from the
          // register copy (reloading the host slot would be a PCIe read); a ptxas forwarding bug like the 12.6 one
          // in fp8v_pipe MODE 3 would break the pair's bitwise agreement, which pipe_collectives_test checks.
          float o[16] = {0};
          dequant16_add(qv, sc, o);
          if (live) store16(out + (2 * i - base_v) * 8, o);
          else {
            ov[2 * i - base_v] = zero;
            ov[2 * i + 1 - base_v] = zero;
          }
        }
      } else {
        int4 r0, r1;
        __nv_bfloat16* p0 = reinterpret_cast<__nv_bfloat16*>(&r0);
        __nv_bfloat16* p1 = reinterpret_cast<__nv_bfloat16*>(&r1);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          p0[k] = __float2bfloat16(acc[k]);
          p1[k] = __float2bfloat16(acc[8 + k]);
        }
        st_wt(red + 2 * i, r0);
        st_wt(red + 2 * i + 1, r1);
        ov[2 * i - base_v] = live ? r0 : zero;
        ov[2 * i + 1 - base_v] = live ? r1 : zero;
      }
    }
  }
  trace_at(L, e, 4);
  publish(flag_at(L.flags2, L, parity, rank, blockIdx.x), tag);
  trace_at(L, e, 5);
  wait_peers(L.flags2, L, parity, rank, 4, blockIdx.x, tag, SIBWAIT ? (1u << sib) : 0xffffffffu);
  trace_at(L, e, 6);
  if (S2FP8) {
    const uint8_t* s8 = slot_at(L.data2, L, parity, sib);
    const float* ss = reinterpret_cast<const float*>(s8 + n);
    const int64_t a = qs(sib) + rlo, b = min(qe(sib), qs(sib) + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) {
      if ((2 * i - base_v) / vpr < count) {
        const int4 qv = ld_cv(reinterpret_cast<const int4*>(s8) + i);
        const float sc = __ldcv(ss + i / 8);
        float o[16] = {0};
        dequant16_add(qv, sc, o);
        store16(out + (2 * i - base_v) * 8, o);
      } else {
        ov[2 * i - base_v] = zero;
        ov[2 * i + 1 - base_v] = zero;
      }
    }
  } else {
    const int4* sred = reinterpret_cast<const int4*>(slot_at(L.data2, L, parity, sib));
    const int64_t a = 2 * (qs(sib) + rlo), b = 2 * min(qe(sib), qs(sib) + rhi);
    for (int64_t v = a + threadIdx.x; v < b; v += blockDim.x)
      ov[v - base_v] = ((v - base_v) / vpr < count) ? ld_cv(sred + v) : zero;
  }
  trace_at(L, e, 7);
  finish_epoch(L, epoch, e, nullptr);
}

// ---------------- DP-aware, SUM_LEN layout (prefill / extend forwards: eager, host row counts) ----------------
// dp_attention SUM_LEN: DP d's rows are global rows [s_d, s_d + n_d), s_0 = 0, s_1 = n0, with nothing between the
// two segments (n_d = the ceil-aligned global_num_tokens). The pair half h = rank & 1 of DP d covers global rows
// [lo(q), hi(q)) for q = rank: [s_d, s_d + H_d) and [s_d + H_d, s_d + n_d), H_d = ceil(n_d / 2) -- the gather's
// published half and the combine's owner quarter. Every block takes the same relative slice of every half, so a
// reader's block b reads exactly what the writer's block b published (per-block flags). Host slots are indexed by
// global row, as x.
struct SumSeg {
  int64_t n0, n1;
  __device__ __forceinline__ int64_t s(int d) const { return d ? n0 : 0; }
  __device__ __forceinline__ int64_t n(int d) const { return d ? n1 : n0; }
  __device__ __forceinline__ int64_t lo(int q) const { return s(q >> 1) + ((q & 1) ? (n(q >> 1) + 1) / 2 : 0); }
  __device__ __forceinline__ int64_t hi(int q) const {
    return s(q >> 1) + ((q & 1) ? n(q >> 1) : (n(q >> 1) + 1) / 2);
  }
  __device__ __forceinline__ int64_t qmax() const { return (max(n0, n1) + 1) / 2; }
};

// zero out[t0, t1) (int4 vectors), split over the grid
__device__ __forceinline__ void zero_tail(int4* ov, int64_t t0, int64_t t1) {
  if (t1 <= t0) return;
  const int64_t per = (t1 - t0 + gridDim.x - 1) / gridDim.x;
  const int64_t a = t0 + per * blockIdx.x, b = min(t1, a + per);
  for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) ov[i] = make_int4(0, 0, 0, 0);
}

// MoE dispatch gather (ARC_HC_SUMLEN). xloc = this DP's n_d rows (bitwise replicated within the pair after the
// attention all-reduce); out = the g_rows global rows: the own DP's rows copied from xloc, the other DP's two halves
// read from the other pair's slots, rows >= n0 + n1 zero. Each rank publishes its half and waits for all three peers
// (the two-parity invariant, as ag_dp_bf16). Per rank: writes ~S/4, reads ~S/2 (S = (n0 + n1) x hidden x 2 bytes)
// vs the zero-filled buffer's two-shot AR4: S / 1.5S, plus no fill / memcpy kernels. Bitwise equal to it (x + 0).
__global__ void ag_sum_bf16(const __nv_bfloat16* xloc, __nv_bfloat16* out, SumSeg G, int64_t g_rows, int64_t hidden,
                            Layout L, int rank, const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int d = rank >> 1, sib = rank ^ 1;
  const int64_t vpr = hidden / 8;
  const int64_t per = (G.qmax() * vpr + gridDim.x - 1) / gridDim.x;
  const int64_t rlo = per * blockIdx.x, rhi = rlo + per;
  const int4* xv = reinterpret_cast<const int4*>(xloc);
  int4* ov = reinterpret_cast<int4*>(out);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  const int64_t base = G.s(d) * vpr;  // xloc vector i is global vector base + i
  {
    const int64_t a = G.lo(rank) * vpr + rlo, b = min(G.hi(rank) * vpr, G.lo(rank) * vpr + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) {
      const int4 v = xv[i - base];
      st_wt(mine + i, v);
      ov[i] = v;
    }
  }
  {
    const int64_t a = G.lo(sib) * vpr + rlo, b = min(G.hi(sib) * vpr, G.lo(sib) * vpr + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) ov[i] = xv[i - base];
  }
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, 4, blockIdx.x, tag);
  for (int q = 0; q < 4; ++q) {
    if ((q >> 1) == d) continue;
    const int4* src = reinterpret_cast<const int4*>(slot_at(L.data, L, parity, q));
    const int64_t a = G.lo(q) * vpr + rlo, b = min(G.hi(q) * vpr, G.lo(q) * vpr + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) ov[i] = ld_cv(src + i);
  }
  zero_tail(ov, (G.n0 + G.n1) * vpr, g_rows * vpr);
}

// MoE combine (ARC_HC_SUMLEN=1). x = this rank's partial over the global rows; out = this DP's rows (out_rows >= n_d,
// rows >= n_d zero, as dp_scatter). rs_owner_bf16 with the SUM_LEN quarters: publish the three quarters the peers own,
// reduce the own quarter in rank order (fp32, so bitwise equal to ar_bf16_twoshot + dp_scatter), publish it, read the
// sibling's. Per rank: writes S, reads S vs the two-shot's S / 1.5S.
__global__ void rs_sum_owner_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, SumSeg G, int64_t out_rows,
                                  int64_t hidden, Layout L, int rank, const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int d = rank >> 1, sib = rank ^ 1;
  const int64_t vpr = hidden / 8;
  const int64_t per = (G.qmax() * vpr + gridDim.x - 1) / gridDim.x;
  const int64_t rlo = per * blockIdx.x, rhi = rlo + per;
  const int4* xv = reinterpret_cast<const int4*>(x);
  int4* mine = reinterpret_cast<int4*>(slot_at(L.data, L, parity, rank));
  for (int q = 0; q < 4; ++q) {
    if (q == rank) continue;
    const int64_t a = G.lo(q) * vpr + rlo, b = min(G.hi(q) * vpr, G.lo(q) * vpr + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) st_wt(mine + i, xv[i]);
  }
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, 4, blockIdx.x, tag);
  int4* red = reinterpret_cast<int4*>(slot_at(L.data2, L, parity, rank));
  int4* ov = reinterpret_cast<int4*>(out);
  const int64_t base = G.s(d) * vpr;  // global vector base + i is out vector i
  {
    const int64_t a = G.lo(rank) * vpr + rlo, b = min(G.hi(rank) * vpr, G.lo(rank) * vpr + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) {
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int4 v = (j == rank) ? xv[i] : ld_cv(reinterpret_cast<const int4*>(slot_at(L.data, L, parity, j)) + i);
        const __nv_bfloat16* bv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] += __bfloat162float(bv[k]);
      }
      int4 r;
      __nv_bfloat16* rb = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
      for (int k = 0; k < 8; ++k) rb[k] = __float2bfloat16(acc[k]);
      st_wt(red + i, r);
      ov[i - base] = r;
    }
  }
  publish(flag_at(L.flags2, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags2, L, parity, rank, 4, blockIdx.x, tag);
  {
    const int4* sred = reinterpret_cast<const int4*>(slot_at(L.data2, L, parity, sib));
    const int64_t a = G.lo(sib) * vpr + rlo, b = min(G.hi(sib) * vpr, G.lo(sib) * vpr + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) ov[i - base] = ld_cv(sred + i);
  }
  zero_tail(ov, G.n(d) * vpr, out_rows * vpr);
}

// MoE combine with an FP8 first stage (ARC_HC_SUMLEN=fp8): rs_owner_fp8 with the SUM_LEN quarters. The three
// quarters a rank publishes go out as e4m3 + one fp32 scale per 128 values (slot: [(n0 + n1) x hidden payload
// bytes][one float per 128]); the owner adds its own quarter exactly and the peers' dequantized partials in rank
// order in fp32; stage 2 stays bf16, so both ranks of a pair hold the same rows bitwise. hidden % 128 == 0.
__global__ void rs_sum_owner_fp8(const __nv_bfloat16* x, __nv_bfloat16* out, SumSeg G, int64_t out_rows,
                                 int64_t hidden, Layout L, int rank, const uint32_t* epoch) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  const int d = rank >> 1, sib = rank ^ 1;
  const int64_t ipr = hidden / 16;  // 16-element items per row; 8 items = one 128-group
  const int64_t n = (G.n0 + G.n1) * hidden;
  int64_t per = (G.qmax() * ipr + gridDim.x - 1) / gridDim.x;
  per = (per + 7) / 8 * 8;  // whole 128-groups per block (the 8 lanes of a group reduce its absmax together)
  const int64_t rlo = per * blockIdx.x, rhi = rlo + per;
  uint8_t* myq = slot_at(L.data, L, parity, rank);
  float* mys = reinterpret_cast<float*>(myq + n);
  __align__(16) __nv_bfloat16 zero16[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) zero16[k] = __float2bfloat16(0.f);
  for (int q = 0; q < 4; ++q) {
    if (q == rank) continue;
    const int64_t a = G.lo(q) * ipr + rlo, b = min(G.hi(q) * ipr, G.lo(q) * ipr + rhi);
    // block-uniform trip count: every lane reaches quant16's shuffles; groups are wholly in or out of [a, b)
    for (int64_t base = a; base < b; base += blockDim.x) {
      const int64_t i = base + threadIdx.x;
      int4 qv;
      float sc;
      quant16(i < b ? x + i * 16 : zero16, qv, sc);
      if (i < b) {
        st_wt(reinterpret_cast<int4*>(myq) + i, qv);
        if ((i & 7) == 0) __stwt(mys + i / 8, sc);
      }
    }
  }
  publish(flag_at(L.flags, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags, L, parity, rank, 4, blockIdx.x, tag);
  int4* red = reinterpret_cast<int4*>(slot_at(L.data2, L, parity, rank));  // bf16, indexed like x
  int4* ov = reinterpret_cast<int4*>(out);
  const int64_t vpr = hidden / 8;
  const int64_t base_v = G.s(d) * vpr;  // first 8-element vector of this DP
  {
    const int64_t a = G.lo(rank) * ipr + rlo, b = min(G.hi(rank) * ipr, G.lo(rank) * ipr + rhi);
    for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) {
      float acc[16] = {0};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (j == rank) {
          const int4* xv = reinterpret_cast<const int4*>(x) + 2 * i;
          int4 v0 = xv[0], v1 = xv[1];
          const __nv_bfloat16* b0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
          const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            acc[k] += __bfloat162float(b0[k]);
            acc[8 + k] += __bfloat162float(b1[k]);
          }
        } else {
          const uint8_t* qj = slot_at(L.data, L, parity, j);
          const int4 qv = ld_cv(reinterpret_cast<const int4*>(qj) + i);
          const float sc = __ldcv(reinterpret_cast<const float*>(qj + n) + i / 8);
          dequant16_add(qv, sc, acc);
        }
      }
      int4 r0, r1;
      __nv_bfloat16* p0 = reinterpret_cast<__nv_bfloat16*>(&r0);
      __nv_bfloat16* p1 = reinterpret_cast<__nv_bfloat16*>(&r1);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        p0[k] = __float2bfloat16(acc[k]);
        p1[k] = __float2bfloat16(acc[8 + k]);
      }
      st_wt(red + 2 * i, r0);
      st_wt(red + 2 * i + 1, r1);
      ov[2 * i - base_v] = r0;
      ov[2 * i + 1 - base_v] = r1;
    }
  }
  publish(flag_at(L.flags2, L, parity, rank, blockIdx.x), tag);
  wait_peers(L.flags2, L, parity, rank, 4, blockIdx.x, tag);
  {
    const int4* sred = reinterpret_cast<const int4*>(slot_at(L.data2, L, parity, sib));
    const int64_t a = 2 * (G.lo(sib) * ipr + rlo), b = 2 * min(G.hi(sib) * ipr, G.lo(sib) * ipr + rhi);
    for (int64_t v = a + threadIdx.x; v < b; v += blockDim.x) ov[v - base_v] = ld_cv(sred + v);
  }
  zero_tail(ov, G.n(d) * vpr, out_rows * vpr);
}

// ---------------- pipelined: writer / reader blocks with staged flags ----------------
// The kernels above run every block as write -> fence -> flag -> wait -> read, so an upstream link carries its two
// GPUs' writes (up) and then their reads (down), one direction at a time. Kaggle L4 10-12 (l4_hcbench): the deployed
// layer AR2 -> AG4 -> rs_owner_fp8 took 717 us against a half-duplex floor of 698, while a link moves 20.3 GB/s both
// ways at once against 12.4 / 13.2 one way.
// Pipelined: writer blocks [0, nb) write chunk c in `stages` steps and publish word s of their (parity, rank, c) flag
// slot after step s; reader blocks [nb, 2nb) take chunk c step by step as the peers publish it, so the reads of step s
// overlap the writes of step s + 1. Per-element arithmetic, accumulation order and payload bytes are those of
// fp8v_kernel / rs_owner_fp8, so the outputs are bitwise identical. Writers never wait; the readers wait on every peer,
// which keeps the two-parity slot invariant (a rank leaves a collective only after every peer has entered it).
// stages 1..8: the eight words of a 32-byte flag slot. __launch_bounds__(512): the roles share one kernel, and
// without the bound rs_owner_fp8_pipe needed more registers than 512 threads get (cudaErrorLaunchOutOfResources).

__device__ __forceinline__ void publish_word(uint32_t* f, uint32_t tag) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(f), "r"(tag) : "memory");
}

// wait until word `word` of chunk c's flag slot holds tag on every rank in peer_mask
__device__ __forceinline__ void wait_word(uint32_t* base, const Layout& L, int parity, int rank, unsigned peer_mask,
                                          int chunk, int word, uint32_t tag) {
  if (threadIdx.x < MAX_RANKS && ((peer_mask >> threadIdx.x) & 1u)) {
    uint32_t* f = flag_at(base, L, parity, threadIdx.x, chunk) + word;
    long long start = clock64();
    uint32_t v;
    while (true) {
      asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
      if (v == tag) break;
      if (HC_POLL_NS > 0) __nanosleep(HC_POLL_NS);
      if (clock64() - start > (long long)HC_TIMEOUT_CYCLES) {
        printf("hostcomm pipe: timeout rank %d waiting peer %d chunk %d word %d tag %u saw %u\n", rank,
               (int)threadIdx.x, chunk, word, tag, v);
        __trap();
      }
    }
  }
  __syncthreads();
}

// step length (items) when [lo, hi) is cut into `stages` steps: whole 128-groups (8 items), so a group is never split
__device__ __forceinline__ int64_t step_len(int64_t len, int stages) {
  return ((len + stages - 1) / stages + 7) / 8 * 8;
}

// MODE 0: one-shot all-reduce (= fp8v_kernel<0>, ar1_fp8v). MODE 2: gather that keeps the payload (= fp8v_kernel<2>,
// ag_fp8q). MODE 3: ag_fp8q under DP attention (4 ranks, pairs (0,1), (2,3)): x = this rank's full DP rows [2n]
// (bitwise replicated in the pair after the attention all-reduce); the rank publishes its own shard as MODE 2 but
// quantizes the sibling's shard locally (the same bytes the sibling publishes) and reads only the other pair's two
// shards. mismatch != nullptr (MODE 3): also read the sibling's published payload and count 32-bit words (payload and
// scales) that differ from the local quantization -- the replication check. check_ctr != nullptr: check only the
// calls whose device call counter c (advanced once per call) satisfies c % every == (c / every) % every: one call in
// `every`, at a position that moves by one every `every` calls, so with `every` = the gathers per forward the checked
// layer rotates through all layers (a Python counter fixes the position at capture). checked: += 1 per checked call.
template <int MODE>
__global__ void __launch_bounds__(512) fp8v_pipe(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t n, Layout L, int rank, int nranks,
                          const uint32_t* epoch, uint8_t* qout, float* sout, int nb, int stages,
                          unsigned long long* mismatch, unsigned long long* check_ctr, int check_every,
                          unsigned long long* checked) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  trace_at(L, e, 0);
  bool check = MODE == 3 && mismatch != nullptr;
  if (check && check_ctr != nullptr && check_every > 1) {
    const unsigned long long calls = *check_ctr;
    check = calls % check_every == (calls / check_every) % check_every;
  }
  const bool writer = (int)blockIdx.x < nb;
  const int c = writer ? (int)blockIdx.x : (int)blockIdx.x - nb;
  const int64_t items = n / 16;
  int64_t per = (items + nb - 1) / nb;
  per = (per + 7) / 8 * 8;
  const int64_t lo = min(items, per * c), hi = min(items, lo + per);
  const int64_t sz = step_len(hi - lo, stages);
  const int lane8 = threadIdx.x & 7;
  const int sib = rank ^ 1;
  const __nv_bfloat16* xown = (MODE == 3) ? x + (int64_t)(rank & 1) * n : x;
  const __nv_bfloat16* xsib = (MODE == 3) ? x + (int64_t)(sib & 1) * n : x;
  __align__(16) __nv_bfloat16 zero[16];
#pragma unroll
  for (int k = 0; k < 16; ++k) zero[k] = __float2bfloat16(0.f);
  if (writer) {
    uint8_t* myq = slot_at(L.data, L, parity, rank);
    float* mys = reinterpret_cast<float*>(myq + n);
    for (int s = 0; s < stages; ++s) {
      const int64_t a = min(hi, lo + s * sz), b = min(hi, a + sz);
      // block-uniform trip count: every lane reaches quant16's shuffles (invalid lanes quantize a zero item)
      for (int64_t base = a; base < b; base += blockDim.x) {
        const int64_t i = base + threadIdx.x;
        const bool valid = i < b;
        int4 q;
        float sc;
        quant16(valid ? xown + i * 16 : zero, q, sc);
        if (valid) {
          st_wt(reinterpret_cast<int4*>(myq) + i, q);
          if (lane8 == 0) __stwt(mys + i / 8, sc);
          if (MODE >= 2) {
            float acc[16] = {0};
            dequant16_add(q, sc, acc);
            store16(out + (int64_t)rank * n + i * 16, acc);
            reinterpret_cast<int4*>(qout + (int64_t)rank * n)[i] = q;
            if (lane8 == 0) sout[(int64_t)rank * (n / QBLOCK) + i / 8] = sc;
          }
        }
        if (MODE == 3) {
          int4 qs2;
          float sc2;
          quant16(valid ? xsib + i * 16 : zero, qs2, sc2);
          if (valid) {
            int4* qd = reinterpret_cast<int4*>(qout + (int64_t)sib * n) + i;
            *qd = qs2;
            if (lane8 == 0) sout[(int64_t)sib * (n / QBLOCK) + i / 8] = sc2;
            // Dequantize the stored copy, not qs2: ptxas 12.6 -O3 (sm_89) unpacked the low e4m3 pair of every
            // 32-bit word of qs2 from a half-merged register here, out[sib] wrong while qout[sib] was right
            // (pipe_collectives_test, 10-13; correct with -Xptxas -O0). A .cv reload is never forwarded.
            float acc[16] = {0};
            dequant16_add(ld_cv(qd), sc2, acc);
            store16(out + (int64_t)sib * n + i * 16, acc);
          }
        }
      }
      publish_word(flag_at(L.flags, L, parity, rank, c) + s, tag);
    }
    trace_at(L, e, 7);
    finish_epoch(L, epoch, e, check_ctr);
    return;
  }
  unsigned peers = 0;
  for (int j = 0; j < nranks; ++j)
    if (j != rank) peers |= 1u << j;
  unsigned long long bad = 0;
  if (check && checked != nullptr && c == 0 && threadIdx.x == 0) atomicAdd(checked, 1ull);
  for (int s = 0; s < stages; ++s) {
    wait_word(L.flags, L, parity, rank, peers, c, s, tag);
    if (s == 0) trace_at(L, e, 3);
    const int64_t a = min(hi, lo + s * sz), b = min(hi, a + sz);
    for (int64_t base = a; base < b; base += blockDim.x) {
      const int64_t i = base + threadIdx.x;
      const bool valid = i < b;
      if (MODE == 0) {
        // own contribution requantized from device memory: the bytes peers read from our slot
        int4 own;
        float own_sc;
        quant16(valid ? x + i * 16 : zero, own, own_sc);
        if (!valid) continue;
        float acc[16] = {0};
        for (int j = 0; j < nranks; ++j) {
          if (j == rank) {
            dequant16_add(own, own_sc, acc);
            continue;
          }
          const uint8_t* qj = slot_at(L.data, L, parity, j);
          int4 q = ld_cv(reinterpret_cast<const int4*>(qj) + i);
          float sc = __ldcv(reinterpret_cast<const float*>(qj + n) + i / 8);
          dequant16_add(q, sc, acc);
        }
        store16(out + i * 16, acc);
      } else {
        if (MODE == 3 && check) {
          int4 mq;
          float msc;
          quant16(valid ? xsib + i * 16 : zero, mq, msc);
          if (valid) {
            const uint8_t* qs = slot_at(L.data, L, parity, sib);
            int4 pq = ld_cv(reinterpret_cast<const int4*>(qs) + i);
            bad += (pq.x != mq.x) + (pq.y != mq.y) + (pq.z != mq.z) + (pq.w != mq.w);
            if (lane8 == 0) {
              const float ps = __ldcv(reinterpret_cast<const float*>(qs + n) + i / 8);
              bad += __float_as_uint(ps) != __float_as_uint(msc);
            }
          }
        }
        if (!valid) continue;
        for (int j = 0; j < nranks; ++j) {
          if (j == rank || (MODE == 3 && (j >> 1) == (rank >> 1))) continue;
          const uint8_t* qj = slot_at(L.data, L, parity, j);
          int4 q = ld_cv(reinterpret_cast<const int4*>(qj) + i);
          float sc = __ldcv(reinterpret_cast<const float*>(qj + n) + i / 8);
          float acc[16] = {0};
          dequant16_add(q, sc, acc);
          store16(out + (int64_t)j * n + i * 16, acc);
          if (MODE >= 2) {
            reinterpret_cast<int4*>(qout + (int64_t)j * n)[i] = q;
            if (lane8 == 0) sout[(int64_t)j * (n / QBLOCK) + i / 8] = sc;
          }
        }
      }
    }
  }
  if (MODE == 3 && check && bad) atomicAdd(mismatch, bad);
  trace_at(L, e, 7);
  finish_epoch(L, epoch, e, check_ctr);
}

// rs_owner_fp8 pipelined: role 0 blocks write the stage-1 partials (the three quarters the peers own, FP8) step by
// step; role 1 blocks reduce this rank's quarter step by step as the peers publish it and publish the reduced bf16 rows
// (stage 2) per step; role 2 blocks read the sibling's reduced quarter step by step. Same quarters, slices, arithmetic
// and bytes as rs_owner_fp8 (bitwise identical out). Grid 3 * nb. S2FP8: as rs_owner_fp8<.., true, ..> (role 2
// already waits for the sibling only).
template <typename IdxT, bool S2FP8>
__global__ void __launch_bounds__(512) rs_owner_fp8_pipe(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t g_rows, int64_t hidden,
                                  const IdxT* start_p, const IdxT* count_p, Layout L, int rank, const uint32_t* epoch,
                                  int nb, int stages) {
  const uint32_t e = *epoch;
  const int parity = e & 1;
  const uint32_t tag = e + 1;
  trace_at(L, e, 0);
  const int d = rank >> 1, sib = rank ^ 1;
  const int role = (int)blockIdx.x / nb, c = (int)blockIdx.x % nb;
  const int64_t ipr = hidden / 16;
  const int64_t n = g_rows * hidden;
  const int64_t Ld = g_rows / 2, Q0 = (Ld + 1) / 2;
  const int64_t start = (int64_t)*start_p, count = (int64_t)*count_p;
  if (start != d * Ld) {
    if (threadIdx.x == 0) printf("hostcomm rs_owner_fp8_pipe: rank %d start %lld != %lld (not MAX_LEN layout)\n", rank,
                                 (long long)start, (long long)(d * Ld));
    __trap();
  }
  auto qs = [&](int q) { return ((q >> 1) * Ld + (q & 1) * Q0) * ipr; };
  auto qe = [&](int q) { return ((q >> 1) * Ld + ((q & 1) ? Ld : Q0)) * ipr; };
  int64_t per = (Q0 * ipr + nb - 1) / nb;
  per = (per + 7) / 8 * 8;
  const int64_t rlo = per * c;
  const int64_t sz = step_len(per, stages);
  // step s of quarter q: items [qs(q) + rlo + s*sz, min(qe(q), qs(q) + min(rlo + per, rlo + (s+1)*sz)))
  auto sa = [&](int q, int s) { return qs(q) + rlo + s * sz; };
  auto sb = [&](int q, int s) { return min(qe(q), qs(q) + min(rlo + per, rlo + (s + 1) * sz)); };
  uint8_t* myq = slot_at(L.data, L, parity, rank);
  float* mys = reinterpret_cast<float*>(myq + n);
  const int64_t vpr = hidden / 8;
  const int64_t base_v = d * Ld * vpr;
  const int4 zero = make_int4(0, 0, 0, 0);
  if (role == 0) {
    __align__(16) __nv_bfloat16 zero16[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) zero16[k] = __float2bfloat16(0.f);
    for (int s = 0; s < stages; ++s) {
      for (int q = 0; q < 4; ++q) {
        if (q == rank) continue;
        const int64_t a = sa(q, s), b = sb(q, s);
        for (int64_t base = a; base < b; base += blockDim.x) {
          const int64_t i = base + threadIdx.x;
          int4 qv;
          float sc;
          quant16(i < b ? x + i * 16 : zero16, qv, sc);
          if (i < b) {
            st_wt(reinterpret_cast<int4*>(myq) + i, qv);
            if ((i & 7) == 0) __stwt(mys + i / 8, sc);
          }
        }
      }
      publish_word(flag_at(L.flags, L, parity, rank, c) + s, tag);
    }
    trace_at(L, e, 7);
    finish_epoch(L, epoch, e, nullptr);
    return;
  }
  int4* ov = reinterpret_cast<int4*>(out);
  if (role == 1) {
    int4* red = reinterpret_cast<int4*>(slot_at(L.data2, L, parity, rank));
    uint8_t* red8 = slot_at(L.data2, L, parity, rank);
    float* reds = reinterpret_cast<float*>(red8 + n);
    const unsigned peers = 0xfu & ~(1u << rank);
    for (int s = 0; s < stages; ++s) {
      wait_word(L.flags, L, parity, rank, peers, c, s, tag);
      if (s == 0) trace_at(L, e, 3);
      const int64_t a = sa(rank, s), b = sb(rank, s);
      // S2FP8: block-uniform trip count (quant16f's shuffles; a and b are whole 128-groups)
      for (int64_t ib = a; S2FP8 ? ib < b : ib + threadIdx.x < b; ib += blockDim.x) {
        const int64_t i = ib + threadIdx.x;
        const bool valid = i < b;
        float acc[16] = {0};
        if (valid) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            if (j == rank) {
              const int4* xv = reinterpret_cast<const int4*>(x) + 2 * i;
              int4 v0 = xv[0], v1 = xv[1];
              const __nv_bfloat16* b0 = reinterpret_cast<const __nv_bfloat16*>(&v0);
              const __nv_bfloat16* b1 = reinterpret_cast<const __nv_bfloat16*>(&v1);
#pragma unroll
              for (int k = 0; k < 8; ++k) {
                acc[k] += __bfloat162float(b0[k]);
                acc[8 + k] += __bfloat162float(b1[k]);
              }
            } else {
              const uint8_t* qj = slot_at(L.data, L, parity, j);
              const int4 qv = ld_cv(reinterpret_cast<const int4*>(qj) + i);
              const float sc = __ldcv(reinterpret_cast<const float*>(qj + n) + i / 8);
              dequant16_add(qv, sc, acc);
            }
          }
        }
        const bool live = (2 * i - base_v) / vpr < count;
        if (S2FP8) {
          int4 qv;
          float sc;
          quant16f(acc, qv, sc);
          if (valid) {
            st_wt(reinterpret_cast<int4*>(red8) + i, qv);
            if ((i & 7) == 0) __stwt(reds + i / 8, sc);
            float o[16] = {0};
            dequant16_add(qv, sc, o);  // own rows = the sibling's dequantization (see rs_owner_fp8)
            if (live) store16(out + (2 * i - base_v) * 8, o);
            else {
              ov[2 * i - base_v] = zero;
              ov[2 * i + 1 - base_v] = zero;
            }
          }
        } else {
          int4 r0, r1;
          __nv_bfloat16* p0 = reinterpret_cast<__nv_bfloat16*>(&r0);
          __nv_bfloat16* p1 = reinterpret_cast<__nv_bfloat16*>(&r1);
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            p0[k] = __float2bfloat16(acc[k]);
            p1[k] = __float2bfloat16(acc[8 + k]);
          }
          st_wt(red + 2 * i, r0);
          st_wt(red + 2 * i + 1, r1);
          ov[2 * i - base_v] = live ? r0 : zero;
          ov[2 * i + 1 - base_v] = live ? r1 : zero;
        }
      }
      publish_word(flag_at(L.flags2, L, parity, rank, c) + s, tag);
    }
    trace_at(L, e, 7);
    finish_epoch(L, epoch, e, nullptr);
    return;
  }
  const int4* sred = reinterpret_cast<const int4*>(slot_at(L.data2, L, parity, sib));
  const uint8_t* s8 = slot_at(L.data2, L, parity, sib);
  const float* ss = reinterpret_cast<const float*>(s8 + n);
  for (int s = 0; s < stages; ++s) {
    wait_word(L.flags2, L, parity, rank, 1u << sib, c, s, tag);
    if (s == 0) trace_at(L, e, 3);
    if (S2FP8) {
      const int64_t a = sa(sib, s), b = sb(sib, s);
      for (int64_t i = a + threadIdx.x; i < b; i += blockDim.x) {
        if ((2 * i - base_v) / vpr < count) {
          const int4 qv = ld_cv(reinterpret_cast<const int4*>(s8) + i);
          const float sc = __ldcv(ss + i / 8);
          float o[16] = {0};
          dequant16_add(qv, sc, o);
          store16(out + (2 * i - base_v) * 8, o);
        } else {
          ov[2 * i - base_v] = zero;
          ov[2 * i + 1 - base_v] = zero;
        }
      }
    } else {
      const int64_t a = 2 * sa(sib, s), b = 2 * sb(sib, s);
      for (int64_t v = a + threadIdx.x; v < b; v += blockDim.x)
        ov[v - base_v] = ((v - base_v) / vpr < count) ? ld_cv(sred + v) : zero;
    }
  }
  trace_at(L, e, 7);
  finish_epoch(L, epoch, e, nullptr);
}

__global__ void bump(uint32_t* epoch, unsigned long long* ctr) {
  *epoch += 1;
  if (ctr != nullptr) *ctr += 1;
}

// After a collective's kernel: advance the epoch (and the optional call counter) with the bump kernel, unless the
// kernel folds it (folds: it calls finish_epoch, and the region has a done counter).
static void end_call(const Layout& L, bool folds, uint32_t* epoch, unsigned long long* ctr, cudaStream_t stream) {
  if (!(folds && L.done != nullptr)) bump<<<1, 1, 0, stream>>>(epoch, ctr);
}

// Pair agreement for the distributed HiSparse spec guards (ARC_SPEC_AGREE_DEVICE in arc_hostcomm_hook.py): g is an
// int64 packet [2][n] all-gathered over an attention-TP pair; word n-1 of each half is that rank's own validity flag
// (0 = valid). A differing word or a set flag prints and fails a device assert: the context dies before anything
// queued after it (mamba commit, host backup) runs, and the next host call raises.
// g: [nranks][n] gathered packets; every rank's copy must equal rank 0's and the last word (validity flag) be 0
__global__ void pair_check_i64(const long long* g, int64_t n, int nranks, int tag) {
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    const long long a = g[i];
    for (int r = 1; r < nranks; ++r) {
      const long long b = g[(int64_t)r * n + i];
      if (a != b || (i == n - 1 && (a | b) != 0)) {
        printf("ARC_SPEC_AGREE_DEVICE fail tag=%d word=%lld/%lld rank0=%lld rank%d=%lld\n", tag, (long long)i,
               (long long)n, a, r, b);
        assert(false);
      }
    }
  }
}

extern "C" {

// kind: 0 ar_oneshot_bf16, 1 ar_twoshot_bf16, 2 ag_bf16, 3 ar_oneshot_fp8, 4 ag_fp8
int hc_launch(int kind, const void* x, void* out, int64_t n, void* flags, void* data, void* data2, void* flags2,
              int64_t slot_bytes, int max_blocks, int rank, int nranks, uint32_t* epoch, int blocks, int threads,
              cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, (uint8_t*)data2, (uint32_t*)flags2, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks > MAX_RANKS) return -1;
  const __nv_bfloat16* xb = (const __nv_bfloat16*)x;
  __nv_bfloat16* ob = (__nv_bfloat16*)out;
  switch (kind) {
    case 0: ar_bf16_oneshot<<<blocks, threads, 0, stream>>>(xb, ob, n, L, rank, nranks, epoch); break;
    case 1: ar_bf16_twoshot<<<blocks, threads, 0, stream>>>(xb, ob, n, L, rank, nranks, epoch); break;
    case 2: ag_bf16<<<blocks, threads, 0, stream>>>(xb, ob, n, L, rank, nranks, epoch); break;
    case 3: ar_fp8_oneshot<<<blocks, threads, 0, stream>>>(xb, ob, n, L, rank, nranks, epoch); break;
    case 4: ag_fp8<<<blocks, threads, 0, stream>>>(xb, ob, n, L, rank, nranks, epoch); break;
    case 5: fp8v_kernel<0><<<blocks, threads, 0, stream>>>(xb, ob, n, L, rank, nranks, epoch, nullptr, nullptr); break;
    case 6: fp8v_kernel<1><<<blocks, threads, 0, stream>>>(xb, ob, n, L, rank, nranks, epoch, nullptr, nullptr); break;
    default: return -2;
  }
  end_call(L, kind == 5 || kind == 6, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

// idx_bytes: 4 or 8 (dtype of start/count device scalars)
int hc_rs_rows(const void* x, void* out, int64_t g_rows, int64_t out_rows, int64_t hidden, const void* start_p,
               const void* count_p, int idx_bytes, void* flags, void* data, int64_t slot_bytes, int max_blocks,
               int rank, int nranks, uint32_t* epoch, int blocks, int threads, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, nullptr, nullptr, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks > MAX_RANKS || hidden % 8) return -1;
  const __nv_bfloat16* xb = (const __nv_bfloat16*)x;
  __nv_bfloat16* ob = (__nv_bfloat16*)out;
  if (idx_bytes == 8)
    rs_rows_bf16<int64_t><<<blocks, threads, 0, stream>>>(xb, ob, g_rows, out_rows, hidden, (const int64_t*)start_p,
                                                          (const int64_t*)count_p, L, rank, nranks, epoch);
  else if (idx_bytes == 4)
    rs_rows_bf16<int32_t><<<blocks, threads, 0, stream>>>(xb, ob, g_rows, out_rows, hidden, (const int32_t*)start_p,
                                                          (const int32_t*)count_p, L, rank, nranks, epoch);
  else
    return -3;
  end_call(L, false, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

int hc_ag_dp(const void* xfull, void* out, int64_t n, void* flags, void* data, int64_t slot_bytes, int max_blocks,
             int rank, int nranks, uint32_t* epoch, int blocks, int threads, void* mismatch, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, nullptr, nullptr, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks != 4 || n % 8 || n * 2 > slot_bytes) return -1;
  ag_dp_bf16<<<blocks, threads, 0, stream>>>((const __nv_bfloat16*)xfull, (__nv_bfloat16*)out, n, L, rank, epoch,
                                             (unsigned long long*)mismatch);
  end_call(L, false, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

int hc_rs_owner(const void* x, void* out, int64_t g_rows, int64_t hidden, const void* start_p, const void* count_p,
                int idx_bytes, void* flags, void* data, void* data2, void* flags2, int64_t slot_bytes, int max_blocks,
                int rank, int nranks, uint32_t* epoch, int blocks, int threads, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, (uint8_t*)data2, (uint32_t*)flags2, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks != 4 || hidden % 8 || g_rows % 2 || g_rows * hidden * 2 > slot_bytes) return -1;
  const __nv_bfloat16* xb = (const __nv_bfloat16*)x;
  __nv_bfloat16* ob = (__nv_bfloat16*)out;
  if (idx_bytes == 8)
    rs_owner_bf16<int64_t><<<blocks, threads, 0, stream>>>(xb, ob, g_rows, hidden, (const int64_t*)start_p,
                                                           (const int64_t*)count_p, L, rank, epoch);
  else if (idx_bytes == 4)
    rs_owner_bf16<int32_t><<<blocks, threads, 0, stream>>>(xb, ob, g_rows, hidden, (const int32_t*)start_p,
                                                           (const int32_t*)count_p, L, rank, epoch);
  else
    return -3;
  end_call(L, false, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

// mode: bit 0 = stage 2 in FP8 (S2FP8), bit 1 = stage 2 waits for the sibling only (SIBWAIT)
int hc_rs_owner_fp8(const void* x, void* out, int64_t g_rows, int64_t hidden, const void* start_p, const void* count_p,
                    int idx_bytes, void* flags, void* data, void* data2, void* flags2, int64_t slot_bytes,
                    int max_blocks, int rank, int nranks, uint32_t* epoch, int blocks, int threads, int mode,
                    cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, (uint8_t*)data2, (uint32_t*)flags2, slot_bytes, max_blocks};
  region_opts(L);
  // data holds the fp8 payload + scales (< S), data2 the bf16 reduced quarters indexed like x (S)
  if (blocks > max_blocks || nranks != 4 || hidden % QBLOCK || g_rows % 2 || g_rows * hidden * 2 > slot_bytes ||
      threads % 32)
    return -1;
  if (mode & ~3) return -4;
  const __nv_bfloat16* xb = (const __nv_bfloat16*)x;
  __nv_bfloat16* ob = (__nv_bfloat16*)out;
#define RS_OWNER_FP8(IDX, S2, SW)                                                                              \
  rs_owner_fp8<IDX, S2, SW><<<blocks, threads, 0, stream>>>(xb, ob, g_rows, hidden, (const IDX*)start_p,        \
                                                            (const IDX*)count_p, L, rank, epoch)
  if (idx_bytes != 8 && idx_bytes != 4) return -3;
  switch (mode + 4 * (idx_bytes == 8)) {
    case 0: RS_OWNER_FP8(int32_t, false, false); break;
    case 1: RS_OWNER_FP8(int32_t, true, false); break;
    case 2: RS_OWNER_FP8(int32_t, false, true); break;
    case 3: RS_OWNER_FP8(int32_t, true, true); break;
    case 4: RS_OWNER_FP8(int64_t, false, false); break;
    case 5: RS_OWNER_FP8(int64_t, true, false); break;
    case 6: RS_OWNER_FP8(int64_t, false, true); break;
    case 7: RS_OWNER_FP8(int64_t, true, true); break;
  }
#undef RS_OWNER_FP8
  end_call(L, true, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

// SUM_LEN MoE dispatch gather: xloc = this DP's n_d rows, out = g_rows global rows (rows >= n0 + n1 zeroed)
int hc_ag_sum(const void* xloc, void* out, int64_t n0, int64_t n1, int64_t g_rows, int64_t hidden, void* flags,
              void* data, int64_t slot_bytes, int max_blocks, int rank, int nranks, uint32_t* epoch, int blocks,
              int threads, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, nullptr, nullptr, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks != 4 || hidden % 8 || n0 < 0 || n1 < 0 || g_rows < n0 + n1 ||
      (n0 + n1) * hidden * 2 > slot_bytes)
    return -1;
  ag_sum_bf16<<<blocks, threads, 0, stream>>>((const __nv_bfloat16*)xloc, (__nv_bfloat16*)out, SumSeg{n0, n1}, g_rows,
                                              hidden, L, rank, epoch);
  end_call(L, false, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

// SUM_LEN MoE combine: x = partial over the global rows (>= n0 + n1), out = this DP's out_rows (>= n_d) rows
int hc_rs_sum_owner(const void* x, void* out, int64_t n0, int64_t n1, int64_t out_rows, int64_t hidden, int fp8,
                    void* flags, void* data, void* data2, void* flags2, int64_t slot_bytes, int max_blocks, int rank,
                    int nranks, uint32_t* epoch, int blocks, int threads, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, (uint8_t*)data2, (uint32_t*)flags2, slot_bytes, max_blocks};
  region_opts(L);
  const int64_t nd = (rank >> 1) ? n1 : n0;
  if (blocks > max_blocks || nranks != 4 || hidden % 8 || n0 < 0 || n1 < 0 || out_rows < nd ||
      (n0 + n1) * hidden * 2 > slot_bytes || (fp8 && (hidden % QBLOCK || threads % 32)))
    return -1;
  const __nv_bfloat16* xb = (const __nv_bfloat16*)x;
  __nv_bfloat16* ob = (__nv_bfloat16*)out;
  if (fp8)
    rs_sum_owner_fp8<<<blocks, threads, 0, stream>>>(xb, ob, SumSeg{n0, n1}, out_rows, hidden, L, rank, epoch);
  else
    rs_sum_owner_bf16<<<blocks, threads, 0, stream>>>(xb, ob, SumSeg{n0, n1}, out_rows, hidden, L, rank, epoch);
  end_call(L, false, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

// fp8v all-gather that also writes the gathered payload: qout [nranks*n] e4m3 bytes, sout [nranks*n/128] fp32 scales
int hc_ag_fp8q(const void* x, void* out, void* qout, void* sout, int64_t n, void* flags, void* data,
               int64_t slot_bytes, int max_blocks, int rank, int nranks, uint32_t* epoch, int blocks, int threads,
               cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, nullptr, nullptr, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks > MAX_RANKS || n % QBLOCK || n + (n / QBLOCK) * 4 > slot_bytes || threads % 32)
    return -1;
  fp8v_kernel<2><<<blocks, threads, 0, stream>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)out, n, L, rank, nranks,
                                                 epoch, (uint8_t*)qout, (float*)sout);
  end_call(L, true, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

// Pipelined variants (fp8v_pipe / rs_owner_fp8_pipe): blocks = chunks nb (grid 2 nb, rs_owner 3 nb), stages 1..8.
// kind: 5 ar1_fp8v (the only pipelined hc_launch kind)
int hc_launch_pipe(int kind, const void* x, void* out, int64_t n, void* flags, void* data, int64_t slot_bytes,
                   int max_blocks, int rank, int nranks, uint32_t* epoch, int blocks, int threads, int stages,
                   cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, nullptr, nullptr, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks > MAX_RANKS || stages < 1 || stages > 8 || n % QBLOCK ||
      n + (n / QBLOCK) * 4 > slot_bytes || threads % 32)
    return -1;
  if (kind != 5) return -2;
  fp8v_pipe<0><<<2 * blocks, threads, 0, stream>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)out, n, L, rank, nranks,
                                                   epoch, nullptr, nullptr, blocks, stages, nullptr, nullptr, 1,
                                                   nullptr);
  end_call(L, true, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

int hc_ag_fp8q_pipe(const void* x, void* out, void* qout, void* sout, int64_t n, void* flags, void* data,
                    int64_t slot_bytes, int max_blocks, int rank, int nranks, uint32_t* epoch, int blocks, int threads,
                    int stages, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, nullptr, nullptr, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks > MAX_RANKS || stages < 1 || stages > 8 || n % QBLOCK ||
      n + (n / QBLOCK) * 4 > slot_bytes || threads % 32)
    return -1;
  fp8v_pipe<2><<<2 * blocks, threads, 0, stream>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)out, n, L, rank, nranks,
                                                   epoch, (uint8_t*)qout, (float*)sout, blocks, stages, nullptr,
                                                   nullptr, 1, nullptr);
  end_call(L, true, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

// ag_fp8q under DP attention (4 ranks): xfull = this rank's DP rows [2n]; mismatch (uint64 device counter) or null;
// check_ctr (uint64 device call counter, advanced per call) or null = check every call; checked (uint64) or null
int hc_ag_dp_fp8q(const void* xfull, void* out, void* qout, void* sout, int64_t n, void* flags, void* data,
                  int64_t slot_bytes, int max_blocks, int rank, uint32_t* epoch, int blocks, int threads, int stages,
                  void* mismatch, void* check_ctr, int check_every, void* checked, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, nullptr, nullptr, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || stages < 1 || stages > 8 || n % QBLOCK || n + (n / QBLOCK) * 4 > slot_bytes ||
      threads % 32 || check_every < 1 || (check_ctr != nullptr && mismatch == nullptr))
    return -1;
  fp8v_pipe<3><<<2 * blocks, threads, 0, stream>>>((const __nv_bfloat16*)xfull, (__nv_bfloat16*)out, n, L, rank, 4,
                                                   epoch, (uint8_t*)qout, (float*)sout, blocks, stages,
                                                   (unsigned long long*)mismatch, (unsigned long long*)check_ctr,
                                                   check_every, (unsigned long long*)checked);
  end_call(L, true, epoch, (unsigned long long*)check_ctr, stream);
  return (int)cudaGetLastError();
}

// mode: bit 0 = stage 2 in FP8 (the pipelined stage 2 always waits for the sibling only; bit 1 is accepted, no-op)
int hc_rs_owner_fp8_pipe(const void* x, void* out, int64_t g_rows, int64_t hidden, const void* start_p,
                         const void* count_p, int idx_bytes, void* flags, void* data, void* data2, void* flags2,
                         int64_t slot_bytes, int max_blocks, int rank, int nranks, uint32_t* epoch, int blocks,
                         int threads, int stages, int mode, cudaStream_t stream) {
  Layout L{(uint32_t*)flags, (uint8_t*)data, (uint8_t*)data2, (uint32_t*)flags2, slot_bytes, max_blocks};
  region_opts(L);
  if (blocks > max_blocks || nranks != 4 || hidden % QBLOCK || g_rows % 2 || g_rows * hidden * 2 > slot_bytes ||
      threads % 32 || stages < 1 || stages > 8)
    return -1;
  if (mode & ~3) return -4;
  const __nv_bfloat16* xb = (const __nv_bfloat16*)x;
  __nv_bfloat16* ob = (__nv_bfloat16*)out;
#define RS_OWNER_FP8_PIPE(IDX, S2)                                                                             \
  rs_owner_fp8_pipe<IDX, S2><<<3 * blocks, threads, 0, stream>>>(xb, ob, g_rows, hidden, (const IDX*)start_p,  \
                                                                 (const IDX*)count_p, L, rank, epoch, blocks, stages)
  if (idx_bytes != 8 && idx_bytes != 4) return -3;
  switch ((mode & 1) + 2 * (idx_bytes == 8)) {
    case 0: RS_OWNER_FP8_PIPE(int32_t, false); break;
    case 1: RS_OWNER_FP8_PIPE(int32_t, true); break;
    case 2: RS_OWNER_FP8_PIPE(int64_t, false); break;
    case 3: RS_OWNER_FP8_PIPE(int64_t, true); break;
  }
#undef RS_OWNER_FP8_PIPE
  end_call(L, true, epoch, nullptr, stream);
  return (int)cudaGetLastError();
}

int hc_pair_check_i64(const void* g, int64_t n, int nranks, int tag, cudaStream_t stream) {
  if (n < 1 || nranks < 2 || nranks > MAX_RANKS) return -1;
  pair_check_i64<<<1, 256, 0, stream>>>((const long long*)g, n, nranks, tag);
  return (int)cudaGetLastError();
}

// Pair P2P (fp8v_kernel<0, true>): a zeroed cudaMalloc region of hc_p2p_bytes, IPC-shared with the sibling.
int hc_p2p_alloc(int64_t bytes, void** ptr) {
  cudaError_t err = cudaMalloc(ptr, bytes);
  if (err != cudaSuccess) return (int)err;
  return (int)cudaMemset(*ptr, 0, bytes);
}
int hc_p2p_free(void* ptr) { return (int)cudaFree(ptr); }
int hc_ipc_handle(void* ptr, void* handle_out) {  // handle_out: CUDA_IPC_HANDLE_SIZE (64) bytes
  return (int)cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(handle_out), ptr);
}
int hc_ipc_open(const void* handle, void** ptr) {
  cudaIpcMemHandle_t h;
  memcpy(&h, handle, sizeof(h));
  return (int)cudaIpcOpenMemHandle(ptr, h, cudaIpcMemLazyEnablePeerAccess);
}
int hc_ipc_close(void* ptr) { return (int)cudaIpcCloseMemHandle(ptr); }

// local / peer: [2 * MAX_RANKS slots of slot_bytes][2 * MAX_RANKS * max_blocks flags of 32 bytes] (hc_p2p_bytes)
int64_t hc_p2p_bytes(int64_t slot_bytes, int max_blocks) {
  return 2 * MAX_RANKS * slot_bytes + 2 * MAX_RANKS * (int64_t)max_blocks * 32;
}
int hc_ar_fp8v_p2p(const void* x, void* out, int64_t n, void* local, void* peer, int64_t slot_bytes, int max_blocks,
                   int rank, uint32_t* epoch, int blocks, int threads, cudaStream_t stream) {
  uint8_t* lb = (uint8_t*)local;
  uint8_t* pb = (uint8_t*)peer;
  const int64_t fo = 2 * MAX_RANKS * slot_bytes;  // flags after the slots
  Layout L{(uint32_t*)(lb + fo), lb, nullptr, nullptr, slot_bytes, max_blocks};
  if (blocks > max_blocks || rank < 0 || rank > 1 || n % QBLOCK || n + (n / QBLOCK) * 4 > slot_bytes || threads % 32)
    return -1;
  fp8v_kernel<0, true><<<blocks, threads, 0, stream>>>((const __nv_bfloat16*)x, (__nv_bfloat16*)out, n, L, rank, 2,
                                                       epoch, nullptr, nullptr, pb, (uint32_t*)(pb + fo));
  bump<<<1, 1, 0, stream>>>(epoch, nullptr);
  return (int)cudaGetLastError();
}

// Per-region options (Layout::done / trace), keyed by the region's flags pointer; nullptr clears an option.
int hc_region_opts(void* flags, void* done, void* trace) {
  for (int i = 0; i < g_region_opts_n; ++i)
    if (g_region_opts[i].flags == flags) {
      g_region_opts[i].done = (uint32_t*)done;
      g_region_opts[i].trace = (unsigned long long*)trace;
      return 0;
    }
  if (g_region_opts_n == (int)(sizeof(g_region_opts) / sizeof(g_region_opts[0]))) return -1;
  g_region_opts[g_region_opts_n++] = RegionOpts{flags, (uint32_t*)done, (unsigned long long*)trace};
  return 0;
}

// 1 when this library records HC_TRACE timestamps
int hc_trace_build() {
#ifdef HC_TRACE
  return 1;
#else
  return 0;
#endif
}

int hc_register(void* ptr, size_t bytes) {
  return (int)cudaHostRegister(ptr, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable);
}

int hc_unregister(void* ptr) { return (int)cudaHostUnregister(ptr); }

int hc_device_pointer(void* host, void** dev) { return (int)cudaHostGetDevicePointer(dev, host, 0); }
}
