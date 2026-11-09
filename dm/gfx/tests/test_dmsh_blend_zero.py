"""A non-finite rgb at alpha 0 returned as 0 by main0 (dmsh_build.py blend_zero), as NVIDIA's blender takes a zero factor times anything: every return of main0 guarded, the shared body
and the integer entry points left as they are, integer outputs never touched, a second pass a no-op, and the result compiled by Metal's compiler when it is there."""
import ast
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import dmsh_side   # noqa: E402

tree = ast.parse((TOOLS / 'dmsh_build.py').read_text())
selected = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == 'blend_zero')
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'BLEND_ZERO' for t in n.targets))]
scope = {'re': re, 'dmsh_side': dmsh_side}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(TOOLS / 'dmsh_build.py'), 'exec'), scope)
blend_zero = scope['blend_zero']

HEAD = '#include <metal_stdlib>\n#include <simd/simd.h>\n\nusing namespace metal;\n\n'
# a light shader's shape as spirv-cross writes it: two colour outputs, an early return that writes zeros, a weighted average divided by its weight sum
PLAIN = HEAD + '''struct main0_out
{
    float4 m_79 [[color(0)]];
    float4 m_81 [[color(1)]];
};

struct main0_in
{
    float4 m_83 [[user(locn0)]];
};

fragment main0_out main0(main0_in in [[stage_in]], float4 gl_FragCoord [[position]])
{
    main0_out out = {};
    float _631 = in.m_83.x;
    float _633 = in.m_83.y;
    if (_633 < 0.0)
    {
        out.m_79 = float4(0.0);
        out.m_81 = float4(0.0);
        return out;
    }
    float _727 = 1.0 / _633;
    float _731 = _727 * _631;
    out.m_79 = float4(_731, _731, _731, in.m_83.z);
    out.m_81 = float4(_731, 0.0, 0.0, in.m_83.z);
    return out;
}
'''


def shared_form(msl):   # main0's body as dm_body_main0, main0 calling it, and an integer entry point calling it too (what fs_uint gives a shader that writes integer bits)
    s = dmsh_side.shared_body(msl)
    e = dmsh_side.entry(s, 'main0')
    struct = 'struct main0_out_u\n{\n    uint4 m_79 [[color(0)]];\n    uint4 m_81 [[color(1)]];\n};\n\n'
    return s + '\n' + struct + dmsh_side.calling(e, 'main0_u', 'main0_out_u', 'return main0_out_u{as_type<uint4>(out.m_79), as_type<uint4>(out.m_81)};')


def body_of(msl, name):
    e = dmsh_side.entry(msl, name)
    return msl[e['head']:e['end']]


class BlendZero(unittest.TestCase):
    def test_every_return_of_main0(self):
        out, n = blend_zero(PLAIN)
        self.assertEqual(n, 2)
        b = body_of(out, 'main0')
        self.assertEqual(b.count('out.m_79 = dm_az(out.m_79); out.m_81 = dm_az(out.m_81);'), 2)
        self.assertEqual(len(re.findall(r'\breturn\b', b)), 2)   # the shape other stages check (one return, the last line) is kept
        self.assertTrue(re.search(r'\n    return out;\n\}$', b))
        self.assertEqual(out.count('float4 dm_az(float4 c)'), 1)
        self.assertLess(out.index('dm_az(float4 c)'), out.index('fragment main0_out main0('))

    def test_shared_body_and_integer_entry_untouched(self):
        src = shared_form(PLAIN)
        out, n = blend_zero(src)
        self.assertEqual(n, 1)   # main0 is the calling entry point: one return
        self.assertIn('dm_az(out.m_79)', body_of(out, 'main0'))
        self.assertNotIn('dm_az', body_of(out, 'main0_u'))   # bits for an integer target stay bits
        b = dmsh_side.body(out)
        self.assertNotIn('dm_az(out', out[b[0]:b[2]])

    def test_integer_outputs_only(self):
        src = PLAIN.replace('float4 m_79 [[color(0)]];', 'uint4 m_79 [[color(0)]];').replace('float4 m_81 [[color(1)]];', 'uint4 m_81 [[color(1)]];')
        self.assertEqual(blend_zero(src), (src, 0))

    def test_no_main0_out(self):
        src = HEAD + 'kernel void main0(uint i [[thread_position_in_grid]]) {}\n'
        self.assertEqual(blend_zero(src), (src, 0))

    def test_second_pass_is_a_no_op(self):
        once, _ = blend_zero(PLAIN)
        self.assertEqual(blend_zero(once), (once, 0))

    def test_return_not_on_its_own_line_refused(self):
        src = PLAIN.replace('        return out;\n    }', '        { return out; }\n    }')
        self.assertEqual(blend_zero(src), (src, 0))   # all or nothing: an unguarded return would let the NaN through on one path

    def test_guard_semantics(self):
        # the helper's rule, evaluated on the CPU the same way: alpha exactly 0 and a non-finite channel -> 0; everything else unchanged
        def az(c):
            r, g, b, a = c
            fix = lambda v: 0.0 if a == 0.0 and (v != v or v in (float('inf'), float('-inf'))) else v
            return (fix(r), fix(g), fix(b), a)
        nan, inf = float('nan'), float('inf')
        self.assertEqual(az((nan, inf, 0.5, 0.0)), (0.0, 0.0, 0.5, 0.0))
        r = az((nan, 1.0, 2.0, 1.0)); self.assertTrue(r[0] != r[0]); self.assertEqual(r[1:], (1.0, 2.0, 1.0))   # alpha 1: the blender keeps a NaN too
        self.assertEqual(az((0.25, -3.0, 7.0, 0.0)), (0.25, -3.0, 7.0, 0.0))   # finite at alpha 0: unchanged
        self.assertIn('c.w == 0.0f', scope['BLEND_ZERO'])
        self.assertIn('isnan(c.xyz) || isinf(c.xyz)', scope['BLEND_ZERO'])

    @unittest.skipUnless(shutil.which('xcrun'), 'no Metal compiler')
    def test_compiles(self):
        for name, src in (('plain', PLAIN), ('shared', shared_form(PLAIN))):
            out, n = blend_zero(src)
            self.assertGreater(n, 0)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / f'{name}.metal'; p.write_text(out)
                r = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-c', '-O1', '-fpreserve-invariance', '-fwrapv', str(p), '-o', str(Path(d) / f'{name}.air')],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr[:500])


if __name__ == '__main__':
    unittest.main()
