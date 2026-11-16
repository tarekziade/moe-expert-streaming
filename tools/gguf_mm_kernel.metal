
// out[row0 + j, n] = sum_k X[row0 + j, k] * W[slot, n, k] for the tiles (slot, row0, rows <= 32) of rows sorted by
// expert. A threadgroup (4 SIMD groups, 128 threads) computes 64 output columns x 32 rows of one tile: per K step of
// 32 it dequantizes 64 x 32 weights and converts 32 x 32 inputs to half in threadgroup memory, then 8x8 simdgroup
// multiply-accumulates in float (ggml's kernel_mul_mm, with a tile list instead of ggml's batch dimensions).
// params: N (output columns), K (inputs), row_bytes (of W), column blocks (N / 64).
template <typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void gguf_mm_tiles(
        device const char * W [[buffer(0)]],
        device const float * X [[buffer(1)]],
        device const int * tiles [[buffer(2)]],
        device float * dst [[buffer(3)]],
        device const uint * params [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float shmem_f[64 * 32];
    threadgroup char * shmem = (threadgroup char *) shmem_f;
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;

    const uint N = params[0], K = params[1], row_bytes = params[2], column_blocks = params[3];
    const uint tile = group / column_blocks;
    const int r0 = (group % column_blocks) * NR0;
    const int slot = tiles[3 * tile + 0];
    const int r1 = tiles[3 * tile + 1];
    const short nr1 = tiles[3 * tile + 2];
    const short nr0 = (int(N) - r0 < NR0) ? (int(N) - r0) : NR0;

    const short lr0 = ((short) tiitg / NL0) < nr0 ? ((short) tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short) tiitg / NL1) < nr1 ? ((short) tiitg / NL1) : nr1 - 1;
    const short il0 = (tiitg % NL0);
    short il = il0;

    device const block_q * x = (device const block_q *)(W + ulong(slot) * N * row_bytes + ulong(row_bytes) * (r0 + lr0)) + il0 / nl;
    const short iy = 8 * (tiitg % NL1);
    device const float * y = X + ulong(r1 + lr1) * K + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < K; loop_k += NK) {
        half4x4 temp_a;
        dequantize_func(x, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }

        {
            const short sx = (tiitg % NL1);
            const short sy = (tiitg / NL1) / 8;
            const short ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) = (half2x4)(*((device const float2x4 *) y));
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x = (il < 2) ? x + (2 + nl - 1) / nl : x;
        y += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        FOR_UNROLL (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    if (nr0 == NR0 && nr1 == NR1) {
        device float * C = dst + (r0 + 32 * (sgitg & 1)) + ulong(r1 + 16 * (sgitg >> 1)) * N;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * N * (i / 4), N, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = shmem_f + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float * D = dst + r0 + ulong(r1 + j) * N;
                threadgroup float * Cs = shmem_f + j * NR0;
                for (int i = 0; i < nr0; i++) {
                    D[i] = Cs[i];
                }
            }
        }
    }
}

#define INSTANTIATE(NAME, BLOCK, NL) \
  template [[host_name("gguf_mm_tiles_" #NAME)]] kernel void gguf_mm_tiles<BLOCK, NL, dequantize_##NAME<half4x4>>( \
      device const char *, device const float *, device const int *, device float *, device const uint *, uint, ushort, ushort);
INSTANTIATE(q2_0, block_q2_0, 4)
INSTANTIATE(iq2_xxs, block_iq2_xxs, QK_NL)
INSTANTIATE(iq2_xs, block_iq2_xs, QK_NL)
INSTANTIATE(iq2_s, block_iq2_s, QK_NL)
INSTANTIATE(iq3_xxs, block_iq3_xxs, QK_NL)
INSTANTIATE(iq3_s, block_iq3_s, QK_NL)
INSTANTIATE(iq4_nl, block_iq4_nl, 2)

// ---- the same tile GEMM on Metal 4's matmul2d (MetalPerformancePrimitives; the M5's matrix units). The staging of
// dequantized weights into threadgroup memory for matmul2d follows Strata's metal/kernels/moe_mm.metal (MIT).
// A threadgroup (4 SIMD groups) computes TN = 64 output columns x TM = 32 rows of a tile, K in steps of TK = 64.
template <typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread half4x4 &)>
kernel void gguf_mm_mpp(
        device const char * W [[buffer(0)]],
        device const float * X [[buffer(1)]],
        device const int * tiles [[buffer(2)]],
        device float * dst [[buffer(3)]],
        device const uint * params [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int TM = 32, TN = 64, TK = 64, NTH = 128;
    threadgroup half xa[TM * TK];
    threadgroup half wd[TN * TK];
    const uint N = params[0], K = params[1], row_bytes = params[2], column_blocks = params[3];
    const uint tile = group / column_blocks;
    const int n0 = (group % column_blocks) * TN;
    const int slot = tiles[3 * tile + 0];
    const int r1 = tiles[3 * tile + 1];
    const int nr1 = tiles[3 * tile + 2];
    device const char * w0 = W + ulong(slot) * N * row_bytes + ulong(n0) * row_bytes;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TM, TN, TK, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(xa, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(wd, dextents<int32_t, 2>(TK, TN));
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += TK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int it = tid; it < TM * TK / 4; it += NTH) {
            const int m = (4 * it) / TK, k = (4 * it) % TK;
            const int row = r1 + min(m, nr1 - 1);
            const float4 v = *(device const float4 *)(X + ulong(row) * K + k0 + k);
            *(threadgroup half4 *)(xa + m * TK + k) = half4(m < nr1 ? v : float4(0.0f));
        }
        for (int it = tid; it < TN * (TK / 16); it += NTH) {
            const int r = it / (TK / 16), c = it % (TK / 16);
            const uint k = k0 + 16 * c;
            device const block_q * b = (device const block_q *)(w0 + ulong(r) * row_bytes) + k / (16 * nl);
            half4x4 v;
            dequantize_func(b, short((k / 16) % nl), v);
            threadgroup half * d = wd + r * TK + 16 * c;
            for (int i = 0; i < 4; ++i) *(threadgroup half4 *)(d + 4 * i) = v[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tA, tW, acc);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(dst, dextents<int32_t, 2>(N, r1 + nr1));
    acc.store(tY.slice(n0, r1));
}

#define INSTANTIATE_MPP(NAME, BLOCK, NL) \
  template [[host_name("gguf_mm_mpp_" #NAME)]] kernel void gguf_mm_mpp<BLOCK, NL, dequantize_##NAME<half4x4>>( \
      device const char *, device const float *, device const int *, device float *, device const uint *, uint, ushort);
INSTANTIATE_MPP(q2_0, block_q2_0, 4)
INSTANTIATE_MPP(iq2_xxs, block_iq2_xxs, QK_NL)
INSTANTIATE_MPP(iq2_xs, block_iq2_xs, QK_NL)
INSTANTIATE_MPP(iq2_s, block_iq2_s, QK_NL)
INSTANTIATE_MPP(iq3_xxs, block_iq3_xxs, QK_NL)
INSTANTIATE_MPP(iq3_s, block_iq3_s, QK_NL)
INSTANTIATE_MPP(iq4_nl, block_iq4_nl, 2)
