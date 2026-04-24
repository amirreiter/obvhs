use glam::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::{
    cwbvh::{
        CwBvhNode,
        node::{EPSILON, extract_byte64},
    },
    ray::Ray,
};

impl CwBvhNode {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    #[inline(always)]
    pub fn intersect_ray_simd(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        let adj_ray_dir_inv = self.compute_extent() * ray.inv_direction;
        let adj_ray_origin = (Vec3A::from(self.p) - ray.origin) * ray.inv_direction;
        let mut hit_mask = 0u32;
        unsafe {
            let adj_ray_dir_inv_x = _mm_set1_ps(adj_ray_dir_inv.x);
            let adj_ray_dir_inv_y = _mm_set1_ps(adj_ray_dir_inv.y);
            let adj_ray_dir_inv_z = _mm_set1_ps(adj_ray_dir_inv.z);

            let adj_ray_orig_x = _mm_set1_ps(adj_ray_origin.x);
            let adj_ray_orig_y = _mm_set1_ps(adj_ray_origin.y);
            let adj_ray_orig_z = _mm_set1_ps(adj_ray_origin.z);

            let rdx = ray.direction.x < 0.0;
            let rdy = ray.direction.y < 0.0;
            let rdz = ray.direction.z < 0.0;

            let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

            #[inline(always)]
            fn get_q(v: &[u8; 8], i: usize) -> __m128 {
                // get_q is the most expensive part of intersect_simd
                // Tried version with _mm_cvtepu8_epi32 and _mm_cvtepi32_ps, it was a lot slower.
                // Tried transmuting v into a u64 and bit shifting, it was a lot slower.
                unsafe {
                    _mm_set_ps(
                        *v.get_unchecked(i * 4 + 3) as f32,
                        *v.get_unchecked(i * 4 + 2) as f32,
                        *v.get_unchecked(i * 4 + 1) as f32,
                        *v.get_unchecked(i * 4) as f32,
                    )
                }
            }

            // Intersect 4 aabbs at a time:
            for i in 0..2 {
                // It's possible to select hi/lo outside the loop with child_min_x, etc... but that seems quite a bit slower
                // using _mm_blendv_ps or similar instead of `if rdx`, etc... is slower

                // Interleaving x, y, z like this is slightly faster than loading all at once. Tried using _mm_prefetch without luck
                let q_lo_x = get_q(&self.child_min_x, i);
                let q_hi_x = get_q(&self.child_max_x, i);
                let x_min = if rdx { q_hi_x } else { q_lo_x };
                let x_max = if rdx { q_lo_x } else { q_hi_x };
                // Tried using _mm_fmadd_ps, it was a lot slower
                let tmin_x = _mm_add_ps(_mm_mul_ps(x_min, adj_ray_dir_inv_x), adj_ray_orig_x);
                let tmax_x = _mm_add_ps(_mm_mul_ps(x_max, adj_ray_dir_inv_x), adj_ray_orig_x);

                let q_lo_y = get_q(&self.child_min_y, i);
                let q_hi_y = get_q(&self.child_max_y, i);
                let y_min = if rdy { q_hi_y } else { q_lo_y };
                let y_max = if rdy { q_lo_y } else { q_hi_y };
                let tmin_y = _mm_add_ps(_mm_mul_ps(y_min, adj_ray_dir_inv_y), adj_ray_orig_y);
                let tmax_y = _mm_add_ps(_mm_mul_ps(y_max, adj_ray_dir_inv_y), adj_ray_orig_y);

                let q_lo_z = get_q(&self.child_min_z, i);
                let q_hi_z = get_q(&self.child_max_z, i);
                let z_min = if rdz { q_hi_z } else { q_lo_z };
                let z_max = if rdz { q_lo_z } else { q_hi_z };
                let tmin_z = _mm_add_ps(_mm_mul_ps(z_min, adj_ray_dir_inv_z), adj_ray_orig_z);
                let tmax_z = _mm_add_ps(_mm_mul_ps(z_max, adj_ray_dir_inv_z), adj_ray_orig_z);

                // Tried using _mm_fmadd_ps, it was a lot slower
                // Compute intersection
                let tmin = _mm_max_ps(tmin_x, _mm_max_ps(tmin_y, tmin_z));
                let tmax = _mm_min_ps(tmax_x, _mm_min_ps(tmax_y, tmax_z));
                let tmin = _mm_max_ps(tmin, _mm_set1_ps(EPSILON)); //ray.tmin?
                let tmax = _mm_min_ps(tmax, _mm_set1_ps(ray.tmax));

                let intersected = _mm_cmple_ps(tmin, tmax);
                let mask = _mm_movemask_ps(intersected);

                for j in 0..4 {
                    let offset = i * 4 + j;
                    if (mask & (1 << j)) != 0 {
                        let child_bits = extract_byte64(child_bits8, offset);
                        let bit_index = extract_byte64(bit_index8, offset);
                        hit_mask |= child_bits << bit_index;
                    }
                }
            }
        }
        hit_mask
    }

    #[inline(always)]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    pub fn intersect_ray_simd(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        let adj_ray_dir_inv = self.compute_extent() * ray.inv_direction;
        let adj_ray_origin = (Vec3A::from(self.p) - ray.origin) * ray.inv_direction;
        let mut hit_mask = 0u32;

        unsafe {
            let inv_x = vdupq_n_f32(adj_ray_dir_inv.x);
            let inv_y = vdupq_n_f32(adj_ray_dir_inv.y);
            let inv_z = vdupq_n_f32(adj_ray_dir_inv.z);
            let orig_x = vdupq_n_f32(adj_ray_origin.x);
            let orig_y = vdupq_n_f32(adj_ray_origin.y);
            let orig_z = vdupq_n_f32(adj_ray_origin.z);

            let rdx = ray.direction.x < 0.0;
            let rdy = ray.direction.y < 0.0;
            let rdz = ray.direction.z < 0.0;

            let tmin_clamp = vdupq_n_f32(EPSILON);
            let tmax_clamp = vdupq_n_f32(ray.tmax);

            let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);

            // Let LLVM auto-vectorize u8->f32 widening — it produces better
            // code than explicit vmovl_u8 -> vmovl_u16 -> vcvtq_f32_u32 chains.
            #[inline(always)]
            unsafe fn get_q(v: &[u8; 8], i: usize) -> float32x4_t {
                unsafe {
                    let base = i * 4;
                    let f = [
                        *v.get_unchecked(base) as f32,
                        *v.get_unchecked(base + 1) as f32,
                        *v.get_unchecked(base + 2) as f32,
                        *v.get_unchecked(base + 3) as f32,
                    ];
                    vld1q_f32(f.as_ptr())
                }
            }

            // vshlq_u32 (1-cycle) instead of vmulq_u32 (3-cycle)
            #[inline(always)]
            unsafe fn movemask(cmp: uint32x4_t) -> u32 {
                unsafe {
                    let bits = vshrq_n_u32(cmp, 31);
                    let shifted = vshlq_u32(bits, vld1q_s32([0i32, 1, 2, 3].as_ptr()));
                    vaddvq_u32(shifted)
                }
            }

            for i in 0..2 {
                // Load near/far per axis — scalar branch is free, mask is lane-uniform
                let q_lo_x = get_q(&self.child_min_x, i);
                let q_hi_x = get_q(&self.child_max_x, i);
                let (x_near, x_far) = if rdx {
                    (q_hi_x, q_lo_x)
                } else {
                    (q_lo_x, q_hi_x)
                };

                let q_lo_y = get_q(&self.child_min_y, i);
                let q_hi_y = get_q(&self.child_max_y, i);
                let (y_near, y_far) = if rdy {
                    (q_hi_y, q_lo_y)
                } else {
                    (q_lo_y, q_hi_y)
                };

                let q_lo_z = get_q(&self.child_min_z, i);
                let q_hi_z = get_q(&self.child_max_z, i);
                let (z_near, z_far) = if rdz {
                    (q_hi_z, q_lo_z)
                } else {
                    (q_lo_z, q_hi_z)
                };

                let tmin_x = vfmaq_f32(orig_x, x_near, inv_x);
                let tmax_x = vfmaq_f32(orig_x, x_far, inv_x);
                let tmin_y = vfmaq_f32(orig_y, y_near, inv_y);
                let tmax_y = vfmaq_f32(orig_y, y_far, inv_y);
                let tmin_z = vfmaq_f32(orig_z, z_near, inv_z);
                let tmax_z = vfmaq_f32(orig_z, z_far, inv_z);

                // Plain max/min — 2-cycle latency vs 3-cycle for vmaxnmq/vminnmq.
                // Safe here: quantized CWBVH bounds are never NaN.
                let tmin = vmaxq_f32(tmin_x, vmaxq_f32(tmin_y, tmin_z));
                let tmax = vminq_f32(tmax_x, vminq_f32(tmax_y, tmax_z));
                let tmin = vmaxq_f32(tmin, tmin_clamp);
                let tmax = vminq_f32(tmax, tmax_clamp);

                let hit = vcleq_f32(tmin, tmax);
                let mask = movemask(hit);

                for j in 0..4 {
                    if mask & (1 << j) != 0 {
                        let offset = i * 4 + j;
                        let cb = extract_byte64(child_bits8, offset) as u32;
                        let bi = extract_byte64(bit_index8, offset) as u32;
                        hit_mask |= cb << bi;
                    }
                }
            }
        }

        hit_mask
    }
}
