// g1_vecadd.cl — G1 kernel: the same vector add as POLARIS/02_vecadd_gpu.mojo,
// written in OpenCL C for now. G1 proves the Mojo HOST side (Level Zero);
// G2 replaces this file with a kernel written in Mojo.
//
// Compiled to SPIR-V on aurora-uan-0007 by g1_build_kernel.sh.

__kernel void vecadd(__global const float *a,
                     __global const float *b,
                     __global float *c,
                     const int n) {
    int gid = (int)get_global_id(0);
    if (gid < n) {                // last group overhangs n — guard it
        c[gid] = a[gid] + b[gid];
    }
}
