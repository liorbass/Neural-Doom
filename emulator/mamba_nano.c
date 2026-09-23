#include "mamba_nano.h"
#include "mamba_nano_weights.h"
#include "mamba_nano_arm_weights.h"
#include <string.h>
#include <math.h>

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

/* One trained Mamba-Nano weight set. The RV32I and ARM32 guests each get
 * their own predictor (trained on their own basic-block traces); the math
 * below is shared. */
typedef struct {
    const float *w_pc_proj, *b_pc_proj, *w_regs_proj, *b_regs_proj;
    const float *l_w_in[2], *l_conv_w[2], *l_conv_b[2], *l_w_xproj[2];
    const float *l_w_dtproj[2], *l_b_dtproj[2], *l_a_log[2], *l_d[2], *l_w_outproj[2];
    const float *ln_w, *ln_b, *w_head_branch, *b_head_branch, *w_head_pc, *b_head_pc;
    float a_neg[2][1024];
    int tables_ready;
} MambaNanoWeights;

static MambaNanoWeights s_w_rv32 = {
    MAMBA_W_PC_PROJ, MAMBA_B_PC_PROJ, MAMBA_W_REGS_PROJ, MAMBA_B_REGS_PROJ,
    { MAMBA_L0_W_IN, MAMBA_L1_W_IN },
    { MAMBA_L0_CONV_W, MAMBA_L1_CONV_W },
    { MAMBA_L0_CONV_B, MAMBA_L1_CONV_B },
    { MAMBA_L0_W_XPROJ, MAMBA_L1_W_XPROJ },
    { MAMBA_L0_W_DTPROJ, MAMBA_L1_W_DTPROJ },
    { MAMBA_L0_B_DTPROJ, MAMBA_L1_B_DTPROJ },
    { MAMBA_L0_A_LOG, MAMBA_L1_A_LOG },
    { MAMBA_L0_D, MAMBA_L1_D },
    { MAMBA_L0_W_OUTPROJ, MAMBA_L1_W_OUTPROJ },
    MAMBA_LN_W, MAMBA_LN_B, MAMBA_W_HEAD_BRANCH, MAMBA_B_HEAD_BRANCH,
    MAMBA_W_HEAD_PC, MAMBA_B_HEAD_PC,
    { { 0 } }, 0
};

static MambaNanoWeights s_w_arm = {
    MAMBA_ARM_W_PC_PROJ, MAMBA_ARM_B_PC_PROJ, MAMBA_ARM_W_REGS_PROJ, MAMBA_ARM_B_REGS_PROJ,
    { MAMBA_ARM_L0_W_IN, MAMBA_ARM_L1_W_IN },
    { MAMBA_ARM_L0_CONV_W, MAMBA_ARM_L1_CONV_W },
    { MAMBA_ARM_L0_CONV_B, MAMBA_ARM_L1_CONV_B },
    { MAMBA_ARM_L0_W_XPROJ, MAMBA_ARM_L1_W_XPROJ },
    { MAMBA_ARM_L0_W_DTPROJ, MAMBA_ARM_L1_W_DTPROJ },
    { MAMBA_ARM_L0_B_DTPROJ, MAMBA_ARM_L1_B_DTPROJ },
    { MAMBA_ARM_L0_A_LOG, MAMBA_ARM_L1_A_LOG },
    { MAMBA_ARM_L0_D, MAMBA_ARM_L1_D },
    { MAMBA_ARM_L0_W_OUTPROJ, MAMBA_ARM_L1_W_OUTPROJ },
    MAMBA_ARM_LN_W, MAMBA_ARM_LN_B, MAMBA_ARM_W_HEAD_BRANCH, MAMBA_ARM_B_HEAD_BRANCH,
    MAMBA_ARM_W_HEAD_PC, MAMBA_ARM_B_HEAD_PC,
    { { 0 } }, 0
};

static void weights_init_tables(MambaNanoWeights *w) {
    if (w->tables_ready) return;
    for (int i = 0; i < 1024; i++) {
        w->a_neg[0][i] = -expf(w->l_a_log[0][i]);
        w->a_neg[1][i] = -expf(w->l_a_log[1][i]);
    }
    w->tables_ready = 1;
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float softplus(float x) {
    return log1pf(expf(-fabsf(x))) + (x > 0.0f ? x : 0.0f);
}

static inline void matvec_simd(const float *x, const float *w, const float *bias, float *out, int in_dim, int out_dim) {
    if (bias) {
        memcpy(out, bias, out_dim * sizeof(float));
    } else {
        memset(out, 0, out_dim * sizeof(float));
    }

#if defined(__wasm_simd128__)
    for (int i = 0; i < in_dim; i++) {
        v128_t vx = wasm_f32x4_splat(x[i]);
        const float *w_row = &w[i * out_dim];
        for (int j = 0; j < out_dim; j += 4) {
            v128_t vw = wasm_v128_load(&w_row[j]);
            v128_t vacc = wasm_v128_load(&out[j]);
            vacc = wasm_f32x4_add(vacc, wasm_f32x4_mul(vx, vw));
            wasm_v128_store(&out[j], vacc);
        }
    }
#elif defined(__AVX2__)
    for (int i = 0; i < in_dim; i++) {
        __m256 vx = _mm256_set1_ps(x[i]);
        const float *w_row = &w[i * out_dim];
        for (int j = 0; j < out_dim; j += 8) {
            __m256 vw = _mm256_loadu_ps(&w_row[j]);
            __m256 vacc = _mm256_loadu_ps(&out[j]);
            vacc = _mm256_fmadd_ps(vx, vw, vacc);
            _mm256_storeu_ps(&out[j], vacc);
        }
    }
#else
    for (int i = 0; i < in_dim; i++) {
        float xi = x[i];
        const float *w_row = &w[i * out_dim];
        for (int j = 0; j < out_dim; j++) {
            out[j] += xi * w_row[j];
        }
    }
#endif
}

void mamba_nano_reset_state(MambaNanoState *state) {
    if (!state) return;
    memset(state->s0, 0, sizeof(state->s0));
    memset(state->s1, 0, sizeof(state->s1));
}

float mamba_nano_get_state_norm(const MambaNanoState *state) {
    if (!state) return 0.0f;
    float sum_sq = 0.0f;
    for (int i = 0; i < 64 * 16; i++) {
        sum_sq += state->s0[i] * state->s0[i];
    }
    return sqrtf(sum_sq);
}

static void mamba_nano_step_w(MambaNanoWeights *w,
                              MambaNanoState *state,
                              const float *pc_bits,
                              const float *regs_norm,
                              float *out_branch_logit,
                              float *out_pc_offset) {
    weights_init_tables(w);

    float h[64];

    /* 1. PC Projection: (32) -> (16) */
    matvec_simd(pc_bits, w->w_pc_proj, w->b_pc_proj, h, 32, 16);

    /* 2. Regs Projection: (32) -> (48) */
    matvec_simd(regs_norm, w->w_regs_proj, w->b_regs_proj, h + 16, 32, 48);

    /* 3. Mamba Layers (2 layers) */
    for (int l = 0; l < 2; l++) {
        float *s = (l == 0) ? state->s0 : state->s1;
        const float *w_in = w->l_w_in[l];
        const float *conv_w = w->l_conv_w[l];
        const float *conv_b = w->l_conv_b[l];
        const float *w_xproj = w->l_w_xproj[l];
        const float *w_dtproj = w->l_w_dtproj[l];
        const float *b_dtproj = w->l_b_dtproj[l];
        const float *d_param = w->l_d[l];
        const float *w_outproj = w->l_w_outproj[l];
        const float *a_neg_l = w->a_neg[l];

        /* In-projection: (64) -> (128) */
        float in_proj[128];
        matvec_simd(h, w_in, NULL, in_proj, 64, 128);
        const float *x_in = in_proj;
        const float *z = in_proj + 64;

        /* 1D Depthwise Conv (single tap at deployment, matches export) */
        float x_act[64];
#if defined(__wasm_simd128__)
        for (int j = 0; j < 64; j += 4) {
            v128_t vx = wasm_v128_load(&x_in[j]);
            v128_t vw = wasm_v128_load(&conv_w[j]);
            v128_t vb = wasm_v128_load(&conv_b[j]);
            v128_t vconv = wasm_f32x4_add(wasm_f32x4_mul(vx, vw), vb);
            float cval[4];
            wasm_v128_store(cval, vconv);
            x_act[j + 0] = silu(cval[0]);
            x_act[j + 1] = silu(cval[1]);
            x_act[j + 2] = silu(cval[2]);
            x_act[j + 3] = silu(cval[3]);
        }
#else
        for (int j = 0; j < 64; j++) {
            float conv_val = x_in[j] * conv_w[j] + conv_b[j];
            x_act[j] = silu(conv_val);
        }
#endif

        /* x_proj: (64) -> (40) [dt_raw: 8, B: 16, C: 16] */
        float ssm_params[40];
        matvec_simd(x_act, w_xproj, NULL, ssm_params, 64, 40);

        const float *dt_raw = &ssm_params[0];
        const float *b_ssm = &ssm_params[8];
        const float *c_ssm = &ssm_params[24];

        /* dt_proj: (8) -> (64) */
        float dt_raw_out[64];
        matvec_simd(dt_raw, w_dtproj, b_dtproj, dt_raw_out, 8, 64);
        float dt[64];
        for (int j = 0; j < 64; j++) {
            dt[j] = softplus(dt_raw_out[j]);
        }

        /* Discretization & State Update & Output */
        float y[64];
        for (int d = 0; d < 64; d++) {
            float dt_d = dt[d];
            float x_d = x_act[d];
            float y_acc = 0.0f;
            int base_idx = d * 16;

#if defined(__wasm_simd128__)
            v128_t v_yacc = wasm_f32x4_splat(0.0f);
            v128_t v_xd = wasm_f32x4_splat(x_d);
            v128_t v_dtd = wasm_f32x4_splat(dt_d);

            for (int n = 0; n < 16; n += 4) {
                int s_idx = base_idx + n;
                float da0 = expf(dt_d * a_neg_l[s_idx + 0]);
                float da1 = expf(dt_d * a_neg_l[s_idx + 1]);
                float da2 = expf(dt_d * a_neg_l[s_idx + 2]);
                float da3 = expf(dt_d * a_neg_l[s_idx + 3]);
                v128_t v_da = wasm_f32x4_make(da0, da1, da2, da3);

                v128_t v_b = wasm_v128_load(&b_ssm[n]);
                v128_t v_db = wasm_f32x4_mul(v_dtd, v_b);

                v128_t v_s = wasm_v128_load(&s[s_idx]);
                v128_t v_snew = wasm_f32x4_add(wasm_f32x4_mul(v_da, v_s), wasm_f32x4_mul(v_db, v_xd));
                wasm_v128_store(&s[s_idx], v_snew);

                v128_t v_c = wasm_v128_load(&c_ssm[n]);
                v_yacc = wasm_f32x4_add(v_yacc, wasm_f32x4_mul(v_snew, v_c));
            }
            float yacc_arr[4];
            wasm_v128_store(yacc_arr, v_yacc);
            y_acc = yacc_arr[0] + yacc_arr[1] + yacc_arr[2] + yacc_arr[3];
#elif defined(__AVX2__)
            __m256 v_yacc = _mm256_setzero_ps();
            __m256 v_xd = _mm256_set1_ps(x_d);
            __m256 v_dtd = _mm256_set1_ps(dt_d);

            for (int n = 0; n < 16; n += 8) {
                int s_idx = base_idx + n;
                float da[8];
                for (int k = 0; k < 8; k++) {
                    da[k] = expf(dt_d * a_neg_l[s_idx + k]);
                }
                __m256 v_da = _mm256_loadu_ps(da);
                __m256 v_b = _mm256_loadu_ps(&b_ssm[n]);
                __m256 v_db = _mm256_mul_ps(v_dtd, v_b);

                __m256 v_s = _mm256_loadu_ps(&s[s_idx]);
                __m256 v_snew = _mm256_fmadd_ps(v_da, v_s, _mm256_mul_ps(v_db, v_xd));
                _mm256_storeu_ps(&s[s_idx], v_snew);

                __m256 v_c = _mm256_loadu_ps(&c_ssm[n]);
                v_yacc = _mm256_fmadd_ps(v_snew, v_c, v_yacc);
            }
            __m128 v_low = _mm256_castps256_ps128(v_yacc);
            __m128 v_high = _mm256_extractf128_ps(v_yacc, 1);
            __m128 v_sum = _mm_add_ps(v_low, v_high);
            v_sum = _mm_hadd_ps(v_sum, v_sum);
            v_sum = _mm_hadd_ps(v_sum, v_sum);
            y_acc = _mm_cvtss_f32(v_sum);
#else
            for (int n = 0; n < 16; n++) {
                int s_idx = base_idx + n;
                float a_val = a_neg_l[s_idx];
                float da = expf(dt_d * a_val);
                float db = dt_d * b_ssm[n];

                float s_new = da * s[s_idx] + db * x_d;
                s[s_idx] = s_new;

                y_acc += s_new * c_ssm[n];
            }
#endif
            y[d] = y_acc + x_d * d_param[d];
        }

        /* Multiplicative Gating & Out Projection */
        float gated[64];
        for (int j = 0; j < 64; j++) {
            gated[j] = y[j] * silu(z[j]);
        }

        float h_res[64];
        matvec_simd(gated, w_outproj, NULL, h_res, 64, 64);
        for (int j = 0; j < 64; j++) {
            h[j] += h_res[j]; /* Residual connection */
        }
    }

    /* 4. Final LayerNorm: mean and variance over 64 dims */
    float sum_h = 0.0f;
    for (int j = 0; j < 64; j++) sum_h += h[j];
    float mean = sum_h / 64.0f;

    float sum_sq = 0.0f;
    for (int j = 0; j < 64; j++) {
        float diff = h[j] - mean;
        sum_sq += diff * diff;
    }
    float var = sum_sq / 64.0f;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);

    float h_ln[64];
#if defined(__wasm_simd128__)
    v128_t v_mean = wasm_f32x4_splat(mean);
    v128_t v_inv_std = wasm_f32x4_splat(inv_std);
    for (int j = 0; j < 64; j += 4) {
        v128_t v_h = wasm_v128_load(&h[j]);
        v128_t v_diff = wasm_f32x4_sub(v_h, v_mean);
        v128_t v_norm = wasm_f32x4_mul(v_diff, v_inv_std);
        v128_t v_w = wasm_v128_load(&w->ln_w[j]);
        v128_t v_b = wasm_v128_load(&w->ln_b[j]);
        v128_t v_res = wasm_f32x4_add(wasm_f32x4_mul(v_norm, v_w), v_b);
        wasm_v128_store(&h_ln[j], v_res);
    }
#elif defined(__AVX2__)
    __m256 v_mean = _mm256_set1_ps(mean);
    __m256 v_inv_std = _mm256_set1_ps(inv_std);
    for (int j = 0; j < 64; j += 8) {
        __m256 v_h = _mm256_loadu_ps(&h[j]);
        __m256 v_diff = _mm256_sub_ps(v_h, v_mean);
        __m256 v_norm = _mm256_mul_ps(v_diff, v_inv_std);
        __m256 v_w = _mm256_loadu_ps(&w->ln_w[j]);
        __m256 v_b = _mm256_loadu_ps(&w->ln_b[j]);
        __m256 v_res = _mm256_fmadd_ps(v_norm, v_w, v_b);
        _mm256_storeu_ps(&h_ln[j], v_res);
    }
#else
    for (int j = 0; j < 64; j++) {
        h_ln[j] = (h[j] - mean) * inv_std * w->ln_w[j] + w->ln_b[j];
    }
#endif

    /* 5. Heads */
    if (out_branch_logit) {
        float acc = w->b_head_branch[0];
#if defined(__wasm_simd128__)
        v128_t v_acc = wasm_f32x4_splat(0.0f);
        for (int j = 0; j < 64; j += 4) {
            v128_t v_h = wasm_v128_load(&h_ln[j]);
            v128_t v_w = wasm_v128_load(&w->w_head_branch[j]);
            v_acc = wasm_f32x4_add(v_acc, wasm_f32x4_mul(v_h, v_w));
        }
        float arr[4];
        wasm_v128_store(arr, v_acc);
        acc += arr[0] + arr[1] + arr[2] + arr[3];
#else
        for (int j = 0; j < 64; j++) {
            acc += h_ln[j] * w->w_head_branch[j];
        }
#endif
        *out_branch_logit = acc;
    }

    if (out_pc_offset) {
        float acc = w->b_head_pc[0];
#if defined(__wasm_simd128__)
        v128_t v_acc = wasm_f32x4_splat(0.0f);
        for (int j = 0; j < 64; j += 4) {
            v128_t v_h = wasm_v128_load(&h_ln[j]);
            v128_t v_w = wasm_v128_load(&w->w_head_pc[j]);
            v_acc = wasm_f32x4_add(v_acc, wasm_f32x4_mul(v_h, v_w));
        }
        float arr[4];
        wasm_v128_store(arr, v_acc);
        acc += arr[0] + arr[1] + arr[2] + arr[3];
#else
        for (int j = 0; j < 64; j++) {
            acc += h_ln[j] * w->w_head_pc[j];
        }
#endif
        *out_pc_offset = acc;
    }
}

void mamba_nano_step(MambaNanoState *state,
                     const float *pc_bits,
                     const float *regs_norm,
                     float *out_branch_logit,
                     float *out_pc_offset) {
    mamba_nano_step_w(&s_w_rv32, state, pc_bits, regs_norm, out_branch_logit, out_pc_offset);
}

void mamba_nano_step_arm(MambaNanoState *state,
                         const float *pc_bits,
                         const float *regs_norm,
                         float *out_branch_logit,
                         float *out_pc_offset) {
    mamba_nano_step_w(&s_w_arm, state, pc_bits, regs_norm, out_branch_logit, out_pc_offset);
}

void mamba_nano_step_from_pc_regs(MambaNanoState *state,
                                  uint32_t pc,
                                  const uint32_t *regs,
                                  float *out_branch_logit,
                                  float *out_pc_offset) {
    float pc_bits[32];
    for (int i = 0; i < 32; i++) {
        pc_bits[i] = (float)((pc >> i) & 1);
    }

    float regs_norm[32];
    for (int r = 0; r < 32; r++) {
        regs_norm[r] = log1pf((float)regs[r]) / 22.2f;
    }

    mamba_nano_step(state, pc_bits, regs_norm, out_branch_logit, out_pc_offset);
}

void mamba_nano_arm_step_from_pc_regs(MambaNanoState *state,
                                      uint32_t pc,
                                      const uint32_t *regs,
                                      float *out_branch_logit,
                                      float *out_pc_offset) {
    float pc_bits[32];
    for (int i = 0; i < 32; i++) {
        pc_bits[i] = (float)((pc >> i) & 1);
    }

    float regs_norm[32];
    for (int r = 0; r < 32; r++) {
        regs_norm[r] = log1pf((float)regs[r]) / 22.2f;
    }

    mamba_nano_step_arm(state, pc_bits, regs_norm, out_branch_logit, out_pc_offset);
}
