use glam::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::{
    cwbvh::{CwBvhNode, node::EPSILON},
    ray::Ray,
};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse2"
))]
use crate::cwbvh::node::extract_byte64;

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

    #[cfg(target_arch = "aarch64")]
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[inline(always)]
    pub fn intersect_ray_simd(&self, ray: &Ray, oct_inv4: u32) -> u32 {
        unsafe {
            // Step 1: Decode exponent scaling shared by all 8 children
            // CWBVH: actual_bound = p + q_raw * scale, where scale = f32::from_bits(e << 23)
            let scale_x = f32::from_bits((self.e[0] as u32) << 23);
            let scale_y = f32::from_bits((self.e[1] as u32) << 23);
            let scale_z = f32::from_bits((self.e[2] as u32) << 23);

            let adj_ray_dir_inv = Vec3A::new(
                scale_x * ray.inv_direction.x,
                scale_y * ray.inv_direction.y,
                scale_z * ray.inv_direction.z,
            );
            let adj_ray_origin = (Vec3A::from(self.p) - ray.origin) * ray.inv_direction;

            // Broadcast to SIMD registers
            let inv_x = vdupq_n_f32(adj_ray_dir_inv.x);
            let inv_y = vdupq_n_f32(adj_ray_dir_inv.y);
            let inv_z = vdupq_n_f32(adj_ray_dir_inv.z);
            let orig_x = vdupq_n_f32(adj_ray_origin.x);
            let orig_y = vdupq_n_f32(adj_ray_origin.y);
            let orig_z = vdupq_n_f32(adj_ray_origin.z);

            // Direction sign masks for vbsl: 0x0 or 0xFFFFFFFF per lane
            let neg_x = vdupq_n_u32(if ray.direction.x < 0.0 { !0u32 } else { 0 });
            let neg_y = vdupq_n_u32(if ray.direction.y < 0.0 { !0u32 } else { 0 });
            let neg_z = vdupq_n_u32(if ray.direction.z < 0.0 { !0u32 } else { 0 });

            let (child_bits8, bit_index8) = self.get_child_and_index_bits(oct_inv4);
            let mut hit_mask = 0u32;

            // Step 2: Load all 8 child bounds at once, then split into two batches

            // X axis: load 8 bytes -> widen to u16x8 -> split -> widen to u32x4 -> convert to f32x4
            let min_x_bytes = vld1_u8(self.child_min_x.as_ptr());
            let max_x_bytes = vld1_u8(self.child_max_x.as_ptr());
            let min_x_u16 = vmovl_u8(min_x_bytes);
            let max_x_u16 = vmovl_u8(max_x_bytes);
            let min_x_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(min_x_u16)));
            let min_x_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(min_x_u16)));
            let max_x_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(max_x_u16)));
            let max_x_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(max_x_u16)));

            // Y axis (same pattern)
            let min_y_bytes = vld1_u8(self.child_min_y.as_ptr());
            let max_y_bytes = vld1_u8(self.child_max_y.as_ptr());
            let min_y_u16 = vmovl_u8(min_y_bytes);
            let max_y_u16 = vmovl_u8(max_y_bytes);
            let min_y_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(min_y_u16)));
            let min_y_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(min_y_u16)));
            let max_y_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(max_y_u16)));
            let max_y_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(max_y_u16)));

            // Z axis (same pattern)
            let min_z_bytes = vld1_u8(self.child_min_z.as_ptr());
            let max_z_bytes = vld1_u8(self.child_max_z.as_ptr());
            let min_z_u16 = vmovl_u8(min_z_bytes);
            let max_z_u16 = vmovl_u8(max_z_bytes);
            let min_z_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(min_z_u16)));
            let min_z_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(min_z_u16)));
            let max_z_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(max_z_u16)));
            let max_z_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(max_z_u16)));

            // Step 3: Process batch 0 (children 0-3)
            process_batch_neon(
                min_x_lo,
                max_x_lo,
                min_y_lo,
                max_y_lo,
                min_z_lo,
                max_z_lo,
                neg_x,
                neg_y,
                neg_z,
                inv_x,
                inv_y,
                inv_z,
                orig_x,
                orig_y,
                orig_z,
                EPSILON,
                ray.tmax,
                std::mem::transmute(&child_bits8),
                std::mem::transmute(&bit_index8),
                0,
                &mut hit_mask,
            );

            // Step 4: Process batch 1 (children 4-7)
            process_batch_neon(
                min_x_hi,
                max_x_hi,
                min_y_hi,
                max_y_hi,
                min_z_hi,
                max_z_hi,
                neg_x,
                neg_y,
                neg_z,
                inv_x,
                inv_y,
                inv_z,
                orig_x,
                orig_y,
                orig_z,
                EPSILON,
                ray.tmax,
                std::mem::transmute(&child_bits8),
                std::mem::transmute(&bit_index8),
                4,
                &mut hit_mask,
            );

            hit_mask
        }
    }
}

/// Process 4 children in NEON — fully unrolled, no loops
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn process_batch_neon(
    x_min: float32x4_t,
    x_max: float32x4_t,
    y_min: float32x4_t,
    y_max: float32x4_t,
    z_min: float32x4_t,
    z_max: float32x4_t,
    neg_x: uint32x4_t,
    neg_y: uint32x4_t,
    neg_z: uint32x4_t,
    inv_x: float32x4_t,
    inv_y: float32x4_t,
    inv_z: float32x4_t,
    orig_x: float32x4_t,
    orig_y: float32x4_t,
    orig_z: float32x4_t,
    ray_tmin: f32,
    ray_tmax: f32, // ← explicit parameters
    child_bits: &[u8; 8],
    bit_index: &[u8; 8],
    batch_offset: usize,
    hit_mask: &mut u32,
) {
    unsafe {
        // ARM-native predicate selection: vbsl is single-cycle, no branches
        let x_min_sel = vbslq_f32(neg_x, x_max, x_min);
        let x_max_sel = vbslq_f32(neg_x, x_min, x_max);
        let y_min_sel = vbslq_f32(neg_y, y_max, y_min);
        let y_max_sel = vbslq_f32(neg_y, y_min, y_max);
        let z_min_sel = vbslq_f32(neg_z, z_max, z_min);
        let z_max_sel = vbslq_f32(neg_z, z_min, z_max);

        // FMA slab test: t = origin + bound * inv_dir (vfmaq = a + b*c)
        let tmin_x = vfmaq_f32(orig_x, x_min_sel, inv_x);
        let tmax_x = vfmaq_f32(orig_x, x_max_sel, inv_x);
        let tmin_y = vfmaq_f32(orig_y, y_min_sel, inv_y);
        let tmax_y = vfmaq_f32(orig_y, y_max_sel, inv_y);
        let tmin_z = vfmaq_f32(orig_z, z_min_sel, inv_z);
        let tmax_z = vfmaq_f32(orig_z, z_max_sel, inv_z);

        // NaN-safe cross-axis reduction (important for axis-aligned/degenerate rays)
        let tmin = vmaxnmq_f32(tmin_x, vmaxnmq_f32(tmin_y, tmin_z));
        let tmax = vminnmq_f32(tmax_x, vminnmq_f32(tmax_y, tmax_z));

        // Clamp to ray range (also NaN-safe)
        let tmin = vmaxnmq_f32(tmin, vdupq_n_f32(ray_tmin));
        let tmax = vminnmq_f32(tmax, vdupq_n_f32(ray_tmax));

        // ARM-native bitmask extraction: shift -> weight -> horizontal add
        let hit = vcleq_f32(tmin, tmax); // uint32x4_t: 0x0 or 0xFFFFFFFF per lane
        let hit_mask_simd = movemask_u32x4(hit); // 4-bit mask: bits 0-3

        // Unrolled scalar tail for metadata extraction
        if hit_mask_simd & 0x1 != 0 {
            let cb = child_bits[batch_offset + 0] as u32;
            let bi = bit_index[batch_offset + 0] as u32;
            *hit_mask |= cb << bi;
        }
        if hit_mask_simd & 0x2 != 0 {
            let cb = child_bits[batch_offset + 1] as u32;
            let bi = bit_index[batch_offset + 1] as u32;
            *hit_mask |= cb << bi;
        }
        if hit_mask_simd & 0x4 != 0 {
            let cb = child_bits[batch_offset + 2] as u32;
            let bi = bit_index[batch_offset + 2] as u32;
            *hit_mask |= cb << bi;
        }
        if hit_mask_simd & 0x8 != 0 {
            let cb = child_bits[batch_offset + 3] as u32;
            let bi = bit_index[batch_offset + 3] as u32;
            *hit_mask |= cb << bi;
        }
    }
}

/// ARM-native: extract 4-bit mask from uint32x4_t comparison result
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
unsafe fn movemask_u32x4(cmp: uint32x4_t) -> u32 {
    static BIT_POSITIONS: [u32; 4] = [1, 2, 4, 8];

    unsafe {
        let bits = vshrq_n_u32(cmp, 31);
        let weighted = vmulq_u32(bits, vld1q_u32(BIT_POSITIONS.as_ptr()));
        vaddvq_u32(weighted)
    }
}
