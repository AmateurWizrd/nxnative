"""Float atomic adds (dmsh_atomic.py float_adds): SPIRV-Cross's compare-and-swap loop becomes one native atomic_float add, and on this Mac's GPU the rewritten kernels sum what the loop does
(exactly for small integers, within rounding for others), for one word, a divergent pair of words, a word per lane and one lane per SIMD-group."""
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import dmsh_atomic  # noqa: E402

CELL = '(*spvDescriptorSet0.cp_s0)._m0[%s]'


def loop(addr, val, n, ind='    ', pre1='', pre2='', hoist=False, load=None):
    """The loop as SPIRV-Cross writes the translator's EmitAtomicAddF32 (names are n*10+k); pre1 / pre2: statements of the loop body before the load and after the sum"""
    o, s, e, c, f, g = (f'_{n}{k}' for k in range(6))
    lines = [f'{ind}bool {g};', f'{ind}do', f'{ind}{{', *(f'{ind}    {x}' for x in pre1), f'{ind}    float {o} = as_type<float>({load or addr});', f'{ind}    float {s} = {o} + {val};', *(f'{ind}    {x}' for x in pre2),
             f'{ind}    uint {e} = as_type<uint>({o});', f'{ind}    uint {c};', f'{ind}    do', f'{ind}    {{', f'{ind}        {c} = {e};',
             f'{ind}    }} while (!atomic_compare_exchange_weak_explicit((device atomic_uint*)&{addr}, &{c}, as_type<uint>({s}), memory_order_relaxed, memory_order_relaxed) && {c} == {e});',
             f'{ind}    {"" if hoist else "uint "}{f} = {c};', f'{ind}    {g} = int({f}) != as_type<int>({o});', f'{ind}}} while ({g});']
    return ''.join(x + '\n' for x in lines)


HEAD = '#include <metal_stdlib>\nusing namespace metal;\nstruct _B { uint _m0[8]; };\nstruct _S { device _B* cp_s0; };\n'
KERNEL = ('kernel void k(device _B* acc [[buffer(0)]], device const float* v [[buffer(1)]], uint id [[thread_position_in_grid]])\n{\n'
          '    _S spvDescriptorSet0 = {acc};\n    float _10 = v[id];\n%s}\n')


def shader(body):
    return HEAD + KERNEL % body


class Rewrite(unittest.TestCase):
    def test_no_loop_no_change(self):
        src = shader('    float _12 = _10 + 1.0;\n')
        self.assertEqual(dmsh_atomic.float_adds(src), src)

    def test_loop_is_one_native_add(self):
        out = dmsh_atomic.float_adds(shader(loop(CELL % 32, '_10', 20)))
        self.assertNotIn('atomic_compare_exchange', out)
        self.assertIn('    float _200 = atomic_fetch_add_explicit((device atomic_float*)&(*spvDescriptorSet0.cp_s0)._m0[32], _10, memory_order_relaxed);\n', out)
        self.assertNotIn('as_type<uint>(_200)', out)   # the loop's result is read by nobody

    def test_address_worked_out_in_the_loop_is_kept(self):
        out = dmsh_atomic.float_adds(shader(loop(CELL % 'int(_53)', '_10', 20, ind='        ', pre1=['uint _51 = id & 3u;'], pre2=['uint _53 = id & 3u;'], load=CELL % 'int(_51)')))
        self.assertNotIn('atomic_compare_exchange', out)
        self.assertIn('        uint _51 = id & 3u;\n        uint _53 = id & 3u;\n        float _200 = atomic_fetch_add_explicit((device atomic_float*)&(*spvDescriptorSet0.cp_s0)._m0[int(_53)], _10, memory_order_relaxed);\n', out)

    def test_result_that_may_be_read_is_kept(self):
        out = dmsh_atomic.float_adds(shader('    uint _204;\n' + loop(CELL % 0, '_10', 20, hoist=True) + '    acc->_m0[1] = _204;\n'))
        self.assertNotIn('atomic_compare_exchange', out)
        self.assertIn('    _204 = as_type<uint>(_200);\n', out)

    def test_old_value_read_elsewhere_keeps_the_loop(self):
        src = shader(loop(CELL % 0, '_10', 20) + '    float _60 = _200;\n')
        self.assertEqual(dmsh_atomic.float_adds(src), src)

    def test_other_loops_stay(self):
        src = shader('    uint _30;\n    do\n    {\n        _30 = 0u;\n    } while (!atomic_compare_exchange_weak_explicit((device atomic_uint*)&' + CELL % 0 + ', &_30, 5u, memory_order_relaxed, memory_order_relaxed) && _30 == 0u);\n')
        self.assertEqual(dmsh_atomic.float_adds(src), src)

    def test_every_loop_of_a_shader(self):
        out = dmsh_atomic.float_adds(shader(''.join(loop(CELL % k, '_10', 20 + k) for k in range(4))))
        self.assertEqual(out.count('atomic_fetch_add_explicit'), 4)
        self.assertNotIn('atomic_compare_exchange', out)


RUNNER = '''#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
int main(int ac, char **av) { @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice(); NSError *e = nil;
    id<MTLLibrary> l = [d newLibraryWithURL:[NSURL fileURLWithPath:@(av[1])] error:&e]; if (!l) { puts(e.description.UTF8String); return 2; }
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k"] error:&e]; if (!p) { puts(e.description.UTF8String); return 2; }
    unsigned n = atoi(av[2]), mode = atoi(av[3]), reps = ac > 4 ? atoi(av[4]) : 1, tg = ac > 5 ? atoi(av[5]) : 128;
    id<MTLBuffer> v = [d newBufferWithLength:n * 4 options:MTLResourceStorageModeShared], o = [d newBufferWithLength:32 options:MTLResourceStorageModeShared];
    float *f = v.contents; for (unsigned i = 0; i < n; i++) f[i] = mode ? (float)((i * 2654435761u >> 8 & 0xffff) / 65536.0) : (float)(i % 7);
    id<MTLCommandQueue> q = [d newCommandQueue]; double best = 1e9;
    for (unsigned r = 0; r < reps; r++) {
        memset(o.contents, 0, 32);
        id<MTLCommandBuffer> c = [q commandBuffer]; id<MTLComputeCommandEncoder> ce = [c computeCommandEncoder]; [ce setComputePipelineState:p]; [ce setBuffer:o offset:0 atIndex:0]; [ce setBuffer:v offset:0 atIndex:1];
        [ce dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)]; [ce endEncoding]; [c commit]; [c waitUntilCompleted];
        double t = (c.GPUEndTime - c.GPUStartTime) * 1e6; if (t < best) best = t;
    }
    const float *a = o.contents; for (int i = 0; i < 8; i++) printf("%.9g ", a[i]); printf("%.2f\\n", best); return 0;
} }
'''

BODIES = {
    'uniform': loop(CELL % 0, '_10', 20),
    'divergent': '    bool _90 = id % 3u != 0u;\n    if (_90)\n    {\n' + loop(CELL % 0, '_10', 20, '        ') + '    }\n    else\n    {\n    float _91 = _10 * 2.0;\n'
                 + loop(CELL % 1, '_91', 21, '        ') + '    }\n',
    'lane': loop(CELL % 'int(_53)', '_10', 20, pre1=['uint _51 = id & 3u;'], pre2=['uint _53 = id & 3u;'], load=CELL % 'int(_51)'),
    'leader': '    if (!simd_is_first()) return;\n' + ''.join(loop(CELL % k, '_10 * %d.0' % (k + 1), 20 + k) for k in range(4)),
}


def expect(name, n, mode):
    v = [float(i % 7) if not mode else float((i * 2654435761 % 2 ** 32 >> 8 & 0xffff) / 65536.0) for i in range(n)]
    out = [0.0] * 8
    if name == 'uniform': out[0] = sum(v)
    elif name == 'divergent':
        out[0] = sum(x for i, x in enumerate(v) if i % 3); out[1] = 2 * sum(x for i, x in enumerate(v) if not i % 3)
    elif name == 'lane':
        for i, x in enumerate(v): out[i & 3] += x
    return out


def build(w, name, body, rewrite):
    """kernel source -> metallib in w; rewrite: False (the loop) or True (float_adds)"""
    src = shader(body)
    if rewrite: src = dmsh_atomic.float_adds(src)
    f = w / f'{name}_{int(rewrite)}.metal'
    f.write_text(src)
    subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-c', '-O1', str(f), '-o', str(f.with_suffix('.air'))], check=True, capture_output=True)
    subprocess.run(['xcrun', '-sdk', 'macosx', 'metallib', str(f.with_suffix('.air')), '-o', str(f.with_suffix('.metallib'))], check=True)
    return f.with_suffix('.metallib')


@unittest.skipUnless(shutil.which('xcrun'), 'needs Metal')
class OnGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.w = Path(cls.dir.name)
        (cls.w / 'r.m').write_text(RUNNER)
        subprocess.run(['clang', '-fobjc-arc', '-framework', 'Metal', '-framework', 'Foundation', str(cls.w / 'r.m'), '-o', str(cls.w / 'r')], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def run_kernel(self, name, rewrite, n, mode, reps=1, tg=128):
        lib = build(self.w, name, BODIES[name], rewrite)
        r = subprocess.run([str(self.w / 'r'), str(lib), str(n), str(mode), str(reps), str(tg)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        *vals, t = r.stdout.split()
        return [float(x) for x in vals], float(t)

    def test_sums_equal_the_loop(self):
        n = 70000   # not a multiple of the SIMD-group or the threadgroup: the last one runs partly
        for name in ('uniform', 'divergent', 'lane'):
            for mode in (0, 1):
                want = expect(name, n, mode)
                for rewrite in (False, True):
                    got, _ = self.run_kernel(name, rewrite, n, mode)
                    for g, w in zip(got, want):
                        self.assertAlmostEqual(g, w, delta=max(abs(w) * 1e-3, 1e-6) if mode else 0, msg=f'{name} rewrite={rewrite} mode {mode}: {got} want {want}')   # small integers: any order is exact

    def test_leader_adds(self):
        n, tg = 8192, 128
        first = [float((32 * s) % 7) for s in range(n // 32)]   # the first lane of each SIMD-group is lane 0
        for rewrite in (False, True):
            got, _ = self.run_kernel('leader', rewrite, n, 0, tg=tg)
            for k in range(4):
                self.assertEqual(got[k], (k + 1) * sum(first), f'rewrite={rewrite} word {k}')


if __name__ == '__main__':
    unittest.main()
