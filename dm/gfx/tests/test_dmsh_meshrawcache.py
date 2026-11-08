#!/usr/bin/env python3
"""A mesh shader's raw global loads, branch-free on one pool-table entry with an exact rerun (dmsh_meshstage.py _raw_cache): the structure of the generated stage (the fast body and its loads, the
exact body taking its last entries, the wrapper that runs one or both), the refusals (a stage that writes, or has no raw load, stays as it was), and on the GPU a mesh shader that reads three
pools, a null pointer and an address in no pool through the lookup's helpers: its picture is the same to the bit with the exact body alone, with the entry its loads hit already in the scratch
buffer, with none there, with another pool's, and with a word that names no entry. Skipped without the Metal compiler or a GPU."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
import dmsh_abi as abi
import dmsh_meshstage as ms

HEAD = """#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;
template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    thread T& operator [] (size_t pos) thread { return elements[pos]; }
    threadgroup T& operator [] (size_t pos) threadgroup { return elements[pos]; }
    device T& operator [] (size_t pos) device { return elements[pos]; }
};
struct _27 { uint _m0[1]; };
struct _10 { uint _m0[16]; };
struct spvDescriptorSetBuffer0 { constant _10* cp_c3 [[id(0)]]; const device _27* cp_s3 [[id(1)]]; };
"""
# The two unresolved raw loads as the translator writes them (GlobalToStorage.TryGenerateRawStorageOp): the entry this helper chose last (its variable, a thread uint), else the hint of the
# address's 2 MiB chunk, else a search of the table; the entry's delta moves the address; an address in no pool reads 0
LOOKUP = """    ulong a = as_type<ulong>(uint2(uint(%(lo)s), uint(%(hi)s)));
    uint e = %(h)s & 8191u;
    ulong s = as_type<ulong>(uint2(cp_s3._m0[e * 8u], cp_s3._m0[e * 8u + 1u])), n = as_type<ulong>(uint2(cp_s3._m0[e * 8u + 2u], cp_s3._m0[e * 8u + 3u]));
    bool holds = a >= s && a < n;
    if (!holds)
    {
        uint chunk = (uint(%(hi)s) << 11) | uint(int(uint(%(lo)s) >> uint(21)));
        uint hw = cp_s3._m0[(chunk & 8191u) * 8u + 6u];
        uint idx = hw & 8191u;
        e = (idx != 0u && (hw >> 13) == chunk && uint(%(hi)s) < 256u) ? idx : 0u;
        if (e == 0u)
        {
            uint step = 4096u;
            do
            {
                uint c = e + step;
                ulong cs = as_type<ulong>(uint2(cp_s3._m0[c * 8u], cp_s3._m0[c * 8u + 1u]));
                e = cs <= a ? c : e;
                step >>= 1;
            } while (step != 0u);
        }
        s = as_type<ulong>(uint2(cp_s3._m0[e * 8u], cp_s3._m0[e * 8u + 1u])); n = as_type<ulong>(uint2(cp_s3._m0[e * 8u + 2u], cp_s3._m0[e * 8u + 3u]));
        holds = a >= s && a < n;
        %(h)s = e;
    }
"""
HELPERS = ('static inline __attribute__((always_inline))\nint _73(thread const int& _a, thread const int& _b, const device _27& cp_s3, thread uint& _42)\n{\n' + LOOKUP % dict(lo='_a', hi='_b', h='_42') +
           '    if (!holds) return 0;\n    ulong d = as_type<ulong>(uint2(cp_s3._m0[e * 8u + 4u], cp_s3._m0[e * 8u + 5u]));\n    uint v = *(const device uint*)(a + d);\n    return as_type<int>(v);\n}\n\n'
           'static inline __attribute__((always_inline))\nint _74(thread const int& _a, thread const int& _b, const device _27& cp_s3, thread uint& _44, thread spvUnsafeArray<uint, 3>& _49)\n{\n' +
           LOOKUP % dict(lo='_a', hi='_b', h='_44') +
           '    if (!holds) { _49[0] = 0u; _49[1] = 0u; _49[2] = 0u; return 0; }\n    ulong d = as_type<ulong>(uint2(cp_s3._m0[e * 8u + 4u], cp_s3._m0[e * 8u + 5u]));\n'
           '    uint4 v = *(const device uint4*)(a + d);\n    _49[0] = v.y; _49[1] = v.z; _49[2] = v.w;\n    return as_type<int>(v.x);\n}\n')
# Vertex v of the meshlet in cell g (5 x 5 vertices, 32 triangles): position from the cell, a1 from a 16-byte load through a pool, a2 from a word. Which pool a lane reads is its number plus the cell's
# modulo 6: pool A, B or C (one 8 MiB with hints), a null pointer, an address in no pool, and a pool's last word. The pointers sit in the uniform buffer: A, B, C, none (lo, hi each)
BODY = """    threadgroup spvUnsafeArray<uint, %(words)d> _60;
    uint _42 = 0u;
    uint _44 = 0u;
    uint lane = gl_SubgroupInvocationID;
    spvUnsafeArray<uint, 3> _49;
    uint g = _60[0];
    uint k = (g + lane) %% 6u;
    constant _10& c = *spvDescriptorSet0.cp_c3;
    uint base = k == 1u ? 2u : k == 2u ? 4u : k == 4u ? 6u : 0u;
    uint off = k == 5u ? c._m0[11] - 16u : 16u * ((lane * 7u + g) %% 1000u);
    int lo = int(c._m0[base] + off), hi = int(c._m0[base + 1u]) + int(c._m0[base] + off < off);
    if (k == 3u) { lo = 0; hi = 0; }
    int w0 = _74(lo, hi, (*spvDescriptorSet0.cp_s3), _44, _49);
    float x0 = as_type<float>(w0), x1 = as_type<float>(_49[0]), x2 = as_type<float>(_49[1]), x3 = as_type<float>(_49[2]);
    int lo2 = lo + 4;
    int hi2 = hi + int(uint(lo2) < uint(lo));
    int w4 = _73(lo2, hi2, (*spvDescriptorSet0.cp_s3), _42);
    float x4 = as_type<float>(w4);
    if (lane < 25u)
    {
        uint vx = lane %% 5u, vy = lane / 5u;
        uint cx = g %% 16u, cy = (g / 16u) %% 16u;
        float fx = (float(vx) * 0.2f + float(cx)) * 0.125f - 1.0f, fy = (float(vy) * 0.2f + float(cy)) * 0.125f - 1.0f;
        uint b = %(out)du + lane;
        _60[b + 0u] = as_type<uint>(fx);
        _60[b + 32u] = as_type<uint>(fy);
        _60[b + 64u] = as_type<uint>(0.5f);
        _60[b + 96u] = as_type<uint>(1.0f);
        _60[b + 128u] = as_type<uint>(x0 + x1 * 0.5f);
        _60[b + 160u] = as_type<uint>(x2 + x3 * 0.5f);
        _60[b + 192u] = as_type<uint>(x4);
    }
    if (lane == 0u) _60[%(prim)du] = 32u << 24;
    threadgroup uchar* by = (threadgroup uchar*)&_60[0];
    if (lane < 32u)
    {
        uint q = lane >> 1, v00 = (q >> 2) * 5u + (q & 3u);
        by[%(idx)du + lane * 3u] = uchar((lane & 1u) ? v00 + 1u : v00);
        by[%(idx)du + lane * 3u + 1u] = uchar((lane & 1u) ? v00 + 6u : v00 + 1u);
        by[%(idx)du + lane * 3u + 2u] = uchar(v00 + 5u);
    }
"""
FRAG = "struct F { float4 position [[position]]; float4 a1 [[user(locn1)]]; float a2 [[user(locn2)]]; };\nfragment float4 dm_frag(F f [[stage_in]]) { return float4(fract(f.a1.xy * 3.0f), fract(f.a1.z + f.a2 * 5.0f), 1.0f); }\n"


def control(stage=7, payload=128, max_v=256, max_p=64):
    c = bytearray(0x80); c[0x56] = stage; c[0x58:0x5c] = (32).to_bytes(2, 'little') + payload.to_bytes(2, 'little'); c[0x5f] = 3
    c[0x60:0x64] = max_v.to_bytes(2, 'little') + max_p.to_bytes(2, 'little')
    return bytes(c)


def sph():
    s = bytearray(0x80); s[0x35] = 0xf0   # the position; generic 1: x and y (the high nibble of byte 0x36), generic 2: x (the low nibble of byte 0x37)
    s[0x36], s[0x37] = 0x30, 0x01
    return bytes(s)


def fixture(raw=True):
    """(the generated mesh stage source, its info)"""
    i = ms.info(control(), sph())
    lay, size = ms.memory(i)
    out, prim = lay[2] >> 2, lay[3]
    kernel = ('kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])\n{\n' +
              BODY % dict(words=size // 4, out=out, prim=prim >> 2, idx=prim + 4) + '}\n')
    src = HEAD + HELPERS + '\n' + kernel
    keep = ms._raw_cache
    try:
        if not raw: ms._raw_cache = lambda *a: None
        return ms.to_stage(src, i), i
    finally: ms._raw_cache = keep


class Structure(unittest.TestCase):
    def test_fast_and_exact_bodies(self):
        src, i = fixture()
        self.assertIn('int _74_f(thread const int& _a, thread const int& _b, const device _27& cp_s3, thread ulong3& dm_c, thread bool& dm_miss, thread spvUnsafeArray<uint, 3>& _49)', src)
        self.assertIn('int _73_f(thread const int& _a, thread const int& _b, const device _27& cp_s3, thread ulong3& dm_c, thread bool& dm_miss)', src)
        fast, exact = src.split('void dm_stage_body_f(')[1].split('void dm_stage_body(')
        self.assertIn('int w0 = _74_f(lo, hi, (*spvDescriptorSet0.cp_s3), dm_c, dm_miss, _49);', fast)
        self.assertIn('_73_f(lo2, hi2, ', fast); self.assertNotIn(' _74(', fast); self.assertNotIn(' _73(', fast)
        self.assertIn('thread uint& _42, thread uint& _44)', exact)   # the exact body gives its last entries back: the wrapper's next group starts from one
        self.assertNotIn('uint _42 = 0u;', exact); self.assertIn('int w0 = _74(lo, hi, (*spvDescriptorSet0.cp_s3), _44, _49);', exact)

    def test_wrapper_runs_the_fast_body_then_the_exact_one_where_a_lane_missed(self):
        src, i = fixture()
        entry = src.split('[[mesh]] void main0(')[1].split('[[mesh]] void main0_z(')[0]
        self.assertIn(f'device uint* dm_slot = (device uint*)dm_scr + ', entry)
        self.assertIn('uint dm_s = simd_broadcast_first(*dm_slot), dm_bad = dm_s >> 13 & 15u;', entry)
        self.assertRegex(entry, r'ulong3 dm_c = dm_entry\(\*spvDescriptorSet0\.cp_s3, dm_s\);\n    if \(dm_try\) dm_stage_body_f\(spvDescriptorSet0, gl_SubgroupInvocationID, _60, dm_c, dm_miss\);\n    if \(!simd_any\(dm_miss\)\)')
        after = entry.split('if (!simd_any(dm_miss))')[1].split('uint np =')[0]
        self.assertIn('dm_stage_body(spvDescriptorSet0, gl_SubgroupInvocationID, _60, dm_h0, dm_h1);', after)
        self.assertIn('((device uint4*)&_60[0])[i] = uint4(0u);', after)   # the stage memory zero again, and its inputs filled again, as at the start
        self.assertIn('_60[0] = dm_first[0] + dm_tg;', after)
        self.assertIn('*dm_slot = (dm_n != 0u ? dm_n : dm_s & 8191u) | min(dm_bad + 1u, 9u) << 13;', after)   # the entry it ended on, and one more miss counted
        slot = int(re.search(r'dm_scr \+ (\d+);', entry)[1]); self.assertTrue(1 <= slot < abi.MESH_SCRATCH_HEAD // 4)   # not the ticket's word 0

    def test_table_entries_are_masked(self):   # the scratch word may hold anything: every value names one of the table's 8,192 entries
        self.assertIn('uint w = (e & 8191u) * 8u;', fixture()[0])

    def test_a_stage_that_writes_stays_as_it_was(self):
        for bad in ('atomic_fetch_add_explicit((device atomic_uint*)dm_scr, 1u, memory_order_relaxed);', 'cp_c3_w;'):
            src = (HEAD + HELPERS).replace('const device _27* cp_s3 [[id(1)]];', 'device _27* cp_s3 [[id(1)]];') if bad == 'cp_c3_w;' else HEAD + HELPERS
            i = ms.info(control(), sph()); size = ms.memory(i)[1]
            kernel = ('kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])\n{\n' +
                      BODY % dict(words=size // 4, out=0, prim=0, idx=0) + ('    ' + bad + '\n' if bad.startswith('atomic') else '') + '}\n')
            out = ms.to_stage(src.replace('device _27& cp_s3', 'device _27& cp_s3') + '\n' + kernel, i)
            self.assertNotIn('dm_stage_body_f', out, bad)
            self.assertIn('dm_stage_body(spvDescriptorSet0, gl_SubgroupInvocationID, _60);', out)

    def test_no_raw_load_no_change(self):
        i = ms.info(control(), sph()); size = ms.memory(i)[1]
        kernel = ('kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])\n{\n'
                  f'    threadgroup spvUnsafeArray<uint, {size // 4}> _60;\n    _60[0] = 1u;\n}}\n')
        out = ms.to_stage(HEAD + kernel, i)
        self.assertNotIn('dm_miss', out); self.assertIn('dm_stage_body(spvDescriptorSet0, gl_SubgroupInvocationID, _60);', out)

    def test_only_a_device_slice_takes_it(self):   # the entry's word lives in the scratch buffer: a stage array in threadgroup memory has none
        i = ms.info(control(max_v=96), sph()); lay, size = ms.memory(i); self.assertLess(size, ms.SPILL_MIN)
        kernel = ('kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]])\n{\n' +
                  BODY % dict(words=size // 4, out=lay[2] >> 2, prim=lay[3] >> 2, idx=lay[3] + 4) + '}\n')
        self.assertNotIn('dm_miss', ms.to_stage(HEAD + HELPERS + kernel, i))


def have_metal():
    return shutil.which('xcrun') and subprocess.run(['xcrun', '-sdk', 'macosx', '-f', 'metal'], capture_output=True).returncode == 0 and shutil.which('clang')


@unittest.skipUnless(have_metal(), 'no Metal compiler')
class Compiles(unittest.TestCase):
    def test_both_platforms(self):   # the iOS app takes the same stage: the SIMD-group functions and ulong3 compile there too
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, 's.metal'); f.write_text(fixture()[0])
            for sdk, target in (('macosx', 'air64-apple-macos15.0'), ('iphoneos', 'air64-apple-ios18.0')):
                r = subprocess.run(['xcrun', '-sdk', sdk, 'metal', '-target', target, '-c', str(f), '-o', str(f.with_suffix('.air'))], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, f'{sdk}: {r.stderr[-1500:]}')


RUNNER = r'''
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <string.h>
#include <stdio.h>
#define W 256
#define H 256
static id<MTLDevice> dev; static id<MTLCommandQueue> q; static NSString *dir;
static id<MTLLibrary> lib(NSString *n) { NSError *e = nil; id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[dir stringByAppendingFormat:@"/%@.metallib", n]] error:&e]; if (!l) { printf("no %s: %s\n", n.UTF8String, e.localizedDescription.UTF8String); exit(1); } return l; }
static id<MTLBuffer> buf(size_t n) { id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; memset(b.contents, 0, n); return b; }
static id<MTLTexture> tex(MTLPixelFormat f) { MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:W height:H mipmapped:NO]; d.usage = MTLTextureUsageRenderTarget; d.storageMode = MTLStorageModeShared; return [dev newTextureWithDescriptor:d]; }
typedef struct { uint64_t lo, hi, delta; } Pool;
int main(int argc, char **argv) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice(); if (!dev) { printf("no Metal device\n"); return 2; } q = [dev newCommandQueue];
        NSDictionary *s = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]] options:0 error:nil]; dir = s[@"dir"];
        int slot = [s[@"slot"] intValue], groups = [s[@"groups"] intValue]; NSString *entry = s[@"entry"];
        // three pools of the guest's address space: A and B small, C 8 MiB on 2 MiB boundaries (its chunks have hints); each holds floats of a pattern of its address
        uint64_t guest[3] = {0x100000, 0x200000, 0x40000000}, size[3] = {0x10000, 0x10000, 0x800000};
        id<MTLBuffer> pool[3]; Pool p[3];
        for (int k = 0; k < 3; k++) { pool[k] = buf(size[k]); float *f = pool[k].contents; for (size_t i = 0; i < size[k] / 4; i++) f[i] = (float)((i * 37 + k * 11) % 251) / 251.0f;
            p[k] = (Pool){guest[k], guest[k] + size[k], pool[k].gpuAddress - guest[k]}; }
        id<MTLBuffer> table = buf(8192 * 32); uint32_t *t = table.contents; for (int e = 1; e < 8192; e++) { t[e * 8] = 0xffffffff; t[e * 8 + 1] = 0xffffffff; t[e * 8 + 2] = 0xffffffff; t[e * 8 + 3] = 0xffffffff; }
        for (int k = 0; k < 3; k++) { uint64_t *e = (uint64_t *)(t + (k + 1) * 8); e[0] = p[k].lo; e[1] = p[k].hi; e[2] = p[k].delta; }
        for (uint64_t c = (guest[2] + 0x1fffff) >> 21; c < (guest[2] + size[2]) >> 21; c++) t[(c & 8191) * 8 + 6] = (uint32_t)(c << 13 | 3);
        id<MTLBuffer> ub = buf(256); uint32_t *u = ub.contents;
        u[0] = (uint32_t)guest[0]; u[1] = guest[0] >> 32; u[2] = (uint32_t)guest[1]; u[3] = guest[1] >> 32; u[4] = (uint32_t)guest[2]; u[5] = guest[2] >> 32; u[6] = 0x10000000; u[7] = 0x20;   // u[6..7]: an address in no pool
        u[8] = 0; u[9] = 0; u[11] = 0x10000;   // the last word of a 64 KiB pool: offset u[11] - 16 from pool A (kind 5 reads pool A's end)
        id<MTLBuffer> arg = buf(16); ((uint64_t *)arg.contents)[0] = ub.gpuAddress; ((uint64_t *)arg.contents)[1] = table.gpuAddress;
        id<MTLBuffer> first = buf(16);
        id<MTLBuffer> scratch = buf(ABI_HEAD + (size_t)ABI_SLICES * ABI_STRIDE);
        id<MTLTexture> c = tex(MTLPixelFormatRGBA8Unorm), d = tex(MTLPixelFormatDepth32Float);
        MTLMeshRenderPipelineDescriptor *pd = [MTLMeshRenderPipelineDescriptor new];
        pd.meshFunction = [lib(@"mesh") newFunctionWithName:entry]; pd.fragmentFunction = [lib(@"frag") newFunctionWithName:@"dm_frag"]; pd.maxTotalThreadsPerMeshThreadgroup = 32;
        pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm; pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        NSError *err = nil; id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithMeshDescriptor:pd options:MTLPipelineOptionNone reflection:nil error:&err];
        if (!ps) { printf("pipeline: %s\n", err.localizedDescription.UTF8String); return 1; }
        MTLDepthStencilDescriptor *dd = [MTLDepthStencilDescriptor new]; dd.depthCompareFunction = MTLCompareFunctionAlways; id<MTLDepthStencilState> ds = [dev newDepthStencilStateWithDescriptor:dd];
        uint32_t *sw = scratch.contents; if (slot > 0) for (int i = 1; i < 16; i++) sw[i] = [s[@"word"] unsignedIntValue];   // every shader's word: whichever the stage hashes to
        id<MTLCommandBuffer> cb = [q commandBuffer];
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor new]; rp.colorAttachments[0].texture = c; rp.colorAttachments[0].loadAction = MTLLoadActionClear; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.depthAttachment.texture = d; rp.depthAttachment.loadAction = MTLLoadActionClear; rp.depthAttachment.storeAction = MTLStoreActionStore; rp.depthAttachment.clearDepth = 1.0;
        id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp]; [e setRenderPipelineState:ps]; [e setDepthStencilState:ds];
        [e useResource:ub usage:MTLResourceUsageRead stages:MTLRenderStageMesh]; [e useResource:table usage:MTLResourceUsageRead stages:MTLRenderStageMesh];
        for (int k = 0; k < 3; k++) [e useResource:pool[k] usage:MTLResourceUsageRead stages:MTLRenderStageMesh];
        [e setMeshBuffer:arg offset:0 atIndex:0]; [e setMeshBuffer:first offset:0 atIndex:ABI_FIRST]; [e setMeshBuffer:scratch offset:0 atIndex:ABI_SCRATCH];
        [e drawMeshThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1) threadsPerMeshThreadgroup:MTLSizeMake(32, 1, 1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) { printf("failed: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        static uint32_t px[W * H]; [c getBytes:px bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0];
        FILE *f = fopen([[dir stringByAppendingString:@"/out.bin"] UTF8String], "wb"); fwrite(px, 4, W * H, f); fclose(f);
        int cover = 0; for (int i = 0; i < W * H; i++) cover += px[i] >> 24 != 0;
        printf("covered %d pixels; scratch words:", cover); for (int i = 0; i < 16; i++) printf(" %u", sw[i]); printf("\n");
        return 0;
    }
}
'''


def compile_lib(d, name, src):
    (d / f'{name}.metal').write_text(src)
    subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-c', '-O1', '-fpreserve-invariance', '-fbracket-depth=1024', str(d / f'{name}.metal'), '-o', str(d / f'{name}.air')], check=True, capture_output=True)
    subprocess.run(['xcrun', '-sdk', 'macosx', 'metallib', str(d / f'{name}.air'), '-o', str(d / f'{name}.metallib')], check=True, capture_output=True)


@unittest.skipUnless(have_metal(), 'no Metal compiler')
class OnTheGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.root = Path(cls.tmp.name)
        runner = RUNNER.replace('ABI_HEAD', str(abi.MESH_SCRATCH_HEAD)).replace('ABI_SLICES', str(abi.MESH_SCRATCH_SLICES)).replace('ABI_STRIDE', str(abi.MESH_SCRATCH_STRIDE)) \
            .replace('ABI_FIRST', str(abi.MESH_FIRST_SLOT)).replace('ABI_SCRATCH', str(abi.MESH_SCRATCH_SLOT))
        (cls.root / 'run.m').write_text(runner)
        subprocess.run(['clang', '-fobjc-arc', '-O1', '-Wno-nontrivial-memaccess', str(cls.root / 'run.m'), '-framework', 'Metal', '-framework', 'Foundation', '-o', str(cls.root / 'run')], check=True, capture_output=True)
        cls.dirs = {}
        for name, raw in (('exact', False), ('fast', True), ('cut', True)):   # cut: the same stage with its rerun cut out
            d = cls.root / name; d.mkdir(); cls.dirs[name] = d
            src, _ = fixture(raw)
            if name == 'cut': src = src.replace('if (!simd_any(dm_miss))', 'if (true)'); assert 'if (true)' in src
            compile_lib(d, 'mesh', src)
            compile_lib(d, 'frag', HEAD.split('struct _27')[0] + FRAG)
        cls.slot = int(re.search(r'dm_scr \+ (\d+);', fixture()[0])[1])

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def run_case(self, which, word=0, entry='main0', groups=120):
        d = self.dirs[which]
        (d / 'spec.json').write_text(json.dumps(dict(dir=str(d), slot=self.slot if which != 'exact' else 0, word=word, groups=groups, entry=entry)))
        r = subprocess.run([str(self.root / 'run'), str(d / 'spec.json')], capture_output=True, text=True)
        if r.returncode == 2: self.skipTest(r.stdout.strip())
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return (d / 'out.bin').read_bytes(), r.stdout

    def test_the_same_picture_whatever_the_scratch_word_holds(self):
        for entry in ('main0', 'main0_z'):
            want, out = self.run_case('exact', entry=entry)
            self.assertRegex(out, r'covered [1-9]\d* pixels')   # the loads reached data: nothing is drawn for a mesh that read zeros alone
            for word, why in ((0, 'no entry yet'), (1, 'pool A'), (2, 'pool B'), (3, 'the big pool, whose chunks have hints'), (0xdeadbeef, 'a word that names no entry'), (0xffffffff, 'every bit')):
                with self.subTest(entry=entry, word=word, why=why):
                    got, out = self.run_case('fast', word, entry)
                    self.assertEqual(got, want, f'{why}: {out}')

    def test_the_picture_needs_the_rerun(self):   # the equality above is not vacuous: a stage that never reran draws another picture, since its loads leave the pool the word names
        want, _ = self.run_case('exact')
        for word in (1, 3): self.assertNotEqual(self.run_case('cut', word)[0], want)

    def test_the_exact_body_leaves_the_entry_its_loads_ended_on(self):   # the next group's start: a real entry (1-3), whatever it held before
        for word in (0, 0xdeadbeef):
            _, out = self.run_case('fast', word)
            words = [int(x) for x in out.split('scratch words:')[1].split()]
            self.assertIn(words[self.slot] & 8191, (1, 2, 3), out)

    def test_a_shader_that_keeps_missing_runs_the_exact_body_alone(self):   # this one reads three pools: from 8 counted misses its groups skip the fast body (one in 64 tries it)
        want, _ = self.run_case('exact')
        got, out = self.run_case('fast', 3 | 9 << 13)
        self.assertEqual(got, want, out)
        word = [int(x) for x in out.split('scratch words:')[1].split()][self.slot]
        self.assertEqual(word & 8191, 3, out); self.assertGreaterEqual(word >> 13 & 15, 8, out)

    def test_a_long_run_of_misses_gives_up(self):   # the count is racy (groups in flight read the same one), so it takes many groups: 30,000 of this one reach the limit from a clean word
        _, out = self.run_case('fast', 0, groups=30000)
        word = [int(x) for x in out.split('scratch words:')[1].split()][self.slot]
        self.assertGreaterEqual(word >> 13 & 15, 8, out)


if __name__ == '__main__':
    unittest.main()
