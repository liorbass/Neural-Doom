#ifndef MAMBA_NANO_H
#define MAMBA_NANO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float s0[64 * 16];
    float s1[64 * 16];
} MambaNanoState;

/* Reset recurrent state to zeros */
void mamba_nano_reset_state(MambaNanoState *state);

/* Single-step forward pass: takes raw pc_bits and regs_norm, updates state, computes logits */
void mamba_nano_step(MambaNanoState *state,
                     const float *pc_bits,
                     const float *regs_norm,
                     float *out_branch_logit,
                     float *out_pc_offset);

/* Forward pass directly from RV32I PC and register values */
void mamba_nano_step_from_pc_regs(MambaNanoState *state,
                                  uint32_t pc,
                                  const uint32_t *regs,
                                  float *out_branch_logit,
                                  float *out_pc_offset);
void mamba_nano_arm_step_from_pc_regs(MambaNanoState *state,
                                      uint32_t pc,
                                      const uint32_t *regs,
                                      float *out_branch_logit,
                                      float *out_pc_offset);

/* Calculate L2 norm of s0 state for telemetry */
float mamba_nano_get_state_norm(const MambaNanoState *state);

#ifdef __cplusplus
}
#endif

#endif /* MAMBA_NANO_H */
