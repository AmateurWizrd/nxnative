"""Border colours Metal has no sampler state for (dmsh_build.py border_emu): the rewrite of a shader's samples, the reflection of the twins and the block, the readable-tile variant
of a rewritten shader, the compile on both platforms, and with DM_TEST_BORDER (the t_border executable) the samples on the GPU. border_twin: a pixel shader's compile unit with entry points
without the correction and their twins with it. border_split: a compute kernel's body twice on the block's mask, the same samples bit for bit with the mask clear and set.
With --fixture <dir>, writes the libraries of t_border's draw test (the plain entry points and the twins as the build cuts them, and the shader as it was)"""
import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import dmsh_abi as abi   # noqa: E402
import dmsh_fetch   # noqa: E402
import dmsh_geometry   # noqa: E402
import dmsh_side   # noqa: E402
import pso_offline   # noqa: E402

tree = ast.parse((TOOLS / 'dmsh_build.py').read_text())
names = {'border_emu', 'border_split', 'border_twin', 'reflect', 'decode_tex', 'decode_ubo_tex', 'decode_ubo_img', 'const_sampler', 'c2_array'}
selected = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(e, 'id', None) in ('BORDER_SAMPLE', 'BORDER_HELPERS', 'BORDER_SPLIT_SAMPLE', '_BS_BODY', '_BS_CALL', 'SSBO', 'TEX_BASE') for t in n.targets for e in (t.elts if isinstance(t, ast.Tuple) else [t])))]
scope = {'re': re, 'abi': abi, 'dmsh_geometry': dmsh_geometry, 'dmsh_side': dmsh_side}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(TOOLS / 'dmsh_build.py'), 'exec'), scope)
border_emu, border_split, border_twin, reflect = scope['border_emu'], scope['border_split'], scope['border_twin'], scope['reflect']


HEAD = '#include <metal_stdlib>\n#include <simd/simd.h>\n\nusing namespace metal;\n\n'
# Tears of the Kingdom's ff9def5b3c418851 as spirv-cross writes it, cut down: a uniform segment, the bend map and its sampler
GRASS = HEAD + '''struct _16
{
    float4 _m0[4096];
};

struct spvDescriptorSetBuffer0
{
    constant _16* fp_c1 [[id(0)]];
    texture2d<float> fp_t_tcb_8 [[id(1)]];
    sampler fp_t_tcb_8Smplr [[id(2)]];
    array<texture2d<float>, 4> fp_t2d_cb14_0 [[id(3)]];
    array<sampler, 4> fp_t2d_cb14_0Smplr [[id(7)]];
};

struct main0_out
{
    float4 m_32 [[color(0)]];
};

struct main0_in
{
    float4 m_29 [[user(locn0), center_no_perspective]];
};

fragment main0_out main0(main0_in in [[stage_in]], constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]])
{
    main0_out out = {};
    float _39 = in.m_29.x;
    float _41 = in.m_29.y;
    float2 _182 = float2(_39, _41);
    float2 _45 = spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, select(_182, float2(0.0), isnan(_182))).xy;
    float4 _50 = spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, _182, level(0.0)).xy, bias(0.5));
    float4 _51 = spvDescriptorSet0.fp_t2d_cb14_0[1].sample(spvDescriptorSet0.fp_t2d_cb14_0Smplr[1], _182);
    out.m_32 = float4(_45, _50.x + _51.y, spvDescriptorSet0.fp_c1->_m0[0].x);
    return out;
}
'''
# what t_border runs: one texture and its sampler, sampled at explicit coordinates
KERNEL = HEAD + '''struct spvDescriptorSetBuffer0
{
    texture2d<float> cp_t_tcb_8 [[id(0)]];
    sampler cp_t_tcb_8Smplr [[id(1)]];
};

kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], device float4* out [[buffer(1)]], constant float2* uv [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    out[i] = spvDescriptorSet0.cp_t_tcb_8.sample(spvDescriptorSet0.cp_t_tcb_8Smplr, uv[i], level(0.0));
}
'''
# the same as the body of a pixel shader, as dmsh_side.shared_body and unit make it, with the kernel t_border runs calling it
KERNEL_BODY = HEAD + '''struct spvDescriptorSetBuffer0
{
    texture2d<float> cp_t_tcb_8 [[id(0)]];
    sampler cp_t_tcb_8Smplr [[id(1)]];
};

struct main0_out
{
    float4 m_0;
};

static main0_out dm_body_main0(float2 uv, constant spvDescriptorSetBuffer0& spvDescriptorSet0)
{
    main0_out out = {};
    out.m_0 = spvDescriptorSet0.cp_t_tcb_8.sample(spvDescriptorSet0.cp_t_tcb_8Smplr, uv, level(0.0));
    return out;
}

kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], device float4* out [[buffer(1)]], constant float2* uv [[buffer(2)]], uint i [[thread_position_in_grid]])
{
    out[i] = dm_body_main0(uv[i], spvDescriptorSet0).m_0;
}
'''

# an integer texture beside a float one, each with its sampler, and an image with an attribute after its id (SPIRV-Cross's interlock) as the last member
INTEGER = HEAD + '''struct spvDescriptorSetBuffer0
{
    texture2d<uint> fp_t_tcb_8 [[id(0)]];
    sampler fp_t_tcb_8Smplr [[id(1)]];
    texture2d<float> fp_t_tcb_9 [[id(2)]];
    sampler fp_t_tcb_9Smplr [[id(3)]];
    texture2d<float, access::read_write> fp_i_0 [[id(4), raster_order_group(0)]];
};

struct main0_out
{
    float4 m_32 [[color(0)]];
};

fragment main0_out main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]])
{
    main0_out out = {};
    uint4 _10 = spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, float2(0.5), level(0.0));
    float4 _11 = spvDescriptorSet0.fp_t_tcb_9.sample(spvDescriptorSet0.fp_t_tcb_9Smplr, float2(0.5));
    out.m_32 = float4(_10) + _11;
    return out;
}
'''


def members(msl):
    body = re.search(r'struct spvDescriptorSetBuffer0\s*\{(.*?)\};', msl, re.S).group(1)
    return [(m.group(1), m.group(2), int(m.group(3))) for m in re.finditer(r'^\s*(.+?)\s+(\w+)\s*\[\[id\((\d+)\)\]\];', body, re.M)]


class Rewrite(unittest.TestCase):
    def test_members_follow_every_entry(self):
        got = members(border_emu(GRASS))
        self.assertEqual(got[-2:], [('sampler', 'fp_t_tcb_8Smplr' + abi.BORDER_TWIN, 11), ('constant dm_border_t*', abi.BORDER_MEMBER, 12)],
                         'the twin and the block come after the sampler array (ids 7-10)')

    def test_samples_are_wrapped(self):
        out = border_emu(GRASS)
        self.assertIn('dm_bs(spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, select(_182, float2(0.0), isnan(_182))), spvDescriptorSet0.dm_border, 0u, '
                      'spvDescriptorSet0.fp_t_tcb_8, spvDescriptorSet0.fp_t_tcb_8Smplr_w, select(_182, float2(0.0), isnan(_182))).xy', out)
        self.assertEqual(out.count('dm_bs('), 5, 'the helper, and four samples through the scalar sampler: one, one inside another (copied into its twin sample), the outer one')
        self.assertIn('spvDescriptorSet0.fp_t2d_cb14_0[1].sample(spvDescriptorSet0.fp_t2d_cb14_0Smplr[1], _182)', out, 'a sampler array keeps its samples')
        inner = 'dm_bs(spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, _182, level(0.0)), spvDescriptorSet0.dm_border, 0u, spvDescriptorSet0.fp_t_tcb_8, spvDescriptorSet0.fp_t_tcb_8Smplr_w, _182, level(0.0))'
        self.assertEqual(out.count(inner), 2, 'the inner sample, rewritten, in both of the outer one\'s samples')

    def test_integer_textures_keep_their_samples(self):
        out = border_emu(INTEGER)
        self.assertIn('uint4 _10 = spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, float2(0.5), level(0.0));', out, 'an integer sample is not wrapped')
        self.assertNotIn('fp_t_tcb_8Smplr' + abi.BORDER_TWIN, out, 'nor its sampler twinned')
        self.assertIn('dm_bs(spvDescriptorSet0.fp_t_tcb_9.sample(', out)

    def test_members_follow_an_attributed_member(self):
        body = re.search(r'struct spvDescriptorSetBuffer0\s*\{(.*?)\};', border_emu(INTEGER), re.S).group(1)
        self.assertIn('sampler fp_t_tcb_9Smplr' + abi.BORDER_TWIN + ' [[id(5)]];', body, 'after the image at id 4, which has raster_order_group')
        self.assertIn(abi.BORDER_MEMBER + ' [[id(6)]];', body)

    def test_unchanged_without_samples(self):
        src = HEAD + 'struct spvDescriptorSetBuffer0\n{\n    texture2d<float> fp_t_tcb_8 [[id(0)]];\n};\n\nfragment void main0() {}\n'
        self.assertEqual(border_emu(src), src)

    def test_reflection(self):
        b = {e['msl_name']: e for e in reflect('k', 'ps', border_emu(GRASS), {'sbufs': []})}
        base, twin = b['fp_t_tcb_8Smplr'], b['fp_t_tcb_8Smplr' + abi.BORDER_TWIN]
        self.assertEqual((twin['kind'], twin.get('reg'), twin['border_bit'], twin['arg_id']), (base['kind'], base.get('reg'), 0, 11))
        self.assertNotIn('border_bit', base)
        self.assertEqual((b[abi.BORDER_MEMBER]['kind'], b[abi.BORDER_MEMBER]['arg_id']), (abi.KIND_BORDER, 12))

    def test_readable_tile_variant_takes_the_wrapped_sample(self):
        src = tile_source()
        refl = reflect('k', 'ps', src, {'sbufs': []})
        fv, meta = dmsh_fetch.fold(src, refl)
        self.assertIsNotNone(fv, meta)
        self.assertEqual(len(meta['calls']), 1)
        self.assertEqual(meta['mixed_regs'], [])
        self.assertRegex(fv, r'\(fx0 == 0 \? fb0 : [^;]*fx0 == 7 \? fb7 : dm_bs\(spvDescriptorSet0\.fp_t_tcb_8\.sample\(','a tile read replaces the corrected sample')

    @unittest.skipUnless(shutil.which('xcrun'), 'requires Metal compilers')
    def test_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, src in (('grass', GRASS), ('kernel', KERNEL), ('integer', INTEGER)):
                msl = root / (name + '.metal'); msl.write_text(border_emu(src))
                for sdk in ('macosx', 'iphoneos'):
                    with self.subTest(shader=name, sdk=sdk):
                        r = subprocess.run(['xcrun', '-sdk', sdk, 'metal', '-c', '-fmodules-cache-path=' + str(root / sdk), str(msl), '-o', str(root / f'{name}_{sdk}.air')], capture_output=True, text=True)
                        self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipUnless(os.environ.get('DM_TEST_BORDER'), 'set DM_TEST_BORDER to the native t_border executable')
    def test_gpu(self):
        out = border_emu(KERNEL)
        self.assertEqual([n for _, n, _ in members(out)], ['cp_t_tcb_8', 'cp_t_tcb_8Smplr', 'cp_t_tcb_8Smplr' + abi.BORDER_TWIN, abi.BORDER_MEMBER], 't_border fills these four')
        with tempfile.TemporaryDirectory(prefix='border_') as directory:
            path = Path(directory) / 'kernel.metal'; path.write_text(out)
            r = subprocess.run([os.environ['DM_TEST_BORDER'], str(path)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


def tile_source():   # GRASS with one sample left and a plain interpolated input: what the readable-tile variant takes
    return border_emu(re.sub(r'    float4 _50 = .*\n', '    float4 _50 = float4(0.0);\n', GRASS.replace(', center_no_perspective', '')))


# the compile sources ps_libraries makes (dmsh_side.unit, then dmsh_fetch.fold of it) of GRASS and of the one-sample shader
UNIT = dmsh_side.unit(border_emu(GRASS))[0]
TILE_UNIT = dmsh_side.unit(tile_source())[0]
FOLDED = dmsh_fetch.fold(TILE_UNIT, reflect('k', 'ps', tile_source(), {'sbufs': []}))[0]


class Split(unittest.TestCase):   # a compute kernel's body twice (a pixel shader's entry points have border_twin)
    def test_a_kernel_calls_its_body_with_its_parameters(self):
        s = border_split(border_emu(KERNEL))
        self.assertIn('template<bool DM_BS> static inline __attribute__((always_inline)) void dm_body_main0_bs(constant spvDescriptorSetBuffer0& spvDescriptorSet0, device float4* out, constant float2* uv, uint i)\n{', s)
        self.assertIn('kernel void main0(constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]], device float4* out [[buffer(1)]], constant float2* uv [[buffer(2)]], uint i [[thread_position_in_grid]])\n'
                      '{\n    if (spvDescriptorSet0.dm_border->mask != 0u) dm_body_main0_bs<true>(spvDescriptorSet0, out, uv, i);\n    else dm_body_main0_bs<false>(spvDescriptorSet0, out, uv, i);\n}', s)
        self.assertEqual((s.count('dm_bs<DM_BS>('), s.count('dm_bs(')), (1, 1))
        self.assertIn('if (S && (b->mask >> i & 1u)) r = dm_bmix(r, t.sample(w, a...), b->k[i]);', s)

    def test_unchanged(self):
        k = border_emu(KERNEL)
        helper = 'static float4 dm_h(constant spvDescriptorSetBuffer0& spvDescriptorSet0)\n{\n    return dm_bs(spvDescriptorSet0.cp_t_tcb_8.sample(spvDescriptorSet0.cp_t_tcb_8Smplr, float2(0.5)), ' \
                 'spvDescriptorSet0.dm_border, 0u, spvDescriptorSet0.cp_t_tcb_8, spvDescriptorSet0.cp_t_tcb_8Smplr_w, float2(0.5));\n}\n'
        for why, v in (('a dm_bs call in a helper function', k.replace('kernel void main0(', helper + 'kernel void main0(')),
                       ('no correction', KERNEL),
                       ('a threadgroup variable (its declaration would have to stay in the entry point)', k.replace('{\n    out[i]', '{\n    threadgroup float tg[4];\n    out[i]')),
                       ('a pixel shader (border_twin)', UNIT)):
            with self.subTest(why):
                self.assertEqual(border_split(v), v)

    @unittest.skipUnless(shutil.which('xcrun'), 'requires Metal compilers')
    def test_compiles(self):
        with tempfile.TemporaryDirectory() as directory:
            msl = Path(directory) / 'kernel.metal'; msl.write_text(border_split(border_emu(KERNEL)))
            for sdk in ('macosx', 'iphoneos'):
                with self.subTest(sdk=sdk):
                    r = subprocess.run(['xcrun', '-sdk', sdk, 'metal', '-c', '-fmodules-cache-path=' + str(Path(directory) / sdk), str(msl), '-o', str(Path(directory) / f'kernel_{sdk}.air')], capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipUnless(os.environ.get('DM_TEST_BORDER'), 'set DM_TEST_BORDER to the native t_border executable')
    def test_gpu_split_samples_as_the_unsplit(self):   # both instantiations, the block's mask clear and set, bit for bit as border_emu alone gives them
        out = border_split(border_emu(KERNEL))
        self.assertIn('dm_body_main0_bs<false>(', out)
        self.assertEqual([n for _, n, _ in members(out)], ['cp_t_tcb_8', 'cp_t_tcb_8Smplr', 'cp_t_tcb_8Smplr' + abi.BORDER_TWIN, abi.BORDER_MEMBER], 't_border fills these four')
        with tempfile.TemporaryDirectory(prefix='border_') as directory:
            split, plain = Path(directory) / 'split.metal', Path(directory) / 'plain.metal'; split.write_text(out); plain.write_text(border_emu(KERNEL))
            r = subprocess.run([os.environ['DM_TEST_BORDER'], str(split), str(plain)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


def halves(v):   # border_twin(v) as (the entry points without the correction, with the prelude: the text before the first twin; the twins' functions; their names)
    text, twins = border_twin(v)
    i = text.rindex('\n', 0, text.index('dm_body_main0' + abi.BORDER_FN + '(')) + 1
    return text[:i], text[i:], twins


class Twin(unittest.TestCase):
    UNITS = (('unit', UNIT, ('main0_bs', 'main0_a_bs', 'main0_l_bs')), ('tile unit', TILE_UNIT, ('main0_bs', 'main0_a_bs', 'main0_l_bs')),
             ('readable-tile version', FOLDED, ('main0_bs', 'main0_a_bs', 'main0_l_bs', 'main0_f_bs')))

    def test_every_entry_point_has_a_twin(self):
        for name, v, twins in self.UNITS:
            with self.subTest(name):
                text, got = border_twin(v)
                self.assertEqual(got, twins)
                names = re.findall(r'^(?:\[\[[^\]]*\]\]\s*)?fragment \w+ (\w+)\(', text, re.M)
                self.assertEqual(names, [n.removesuffix(abi.BORDER_FN) for n in twins] + list(twins), 'the entry points, then their twins')

    def test_the_entry_points_have_no_correction(self):
        for name, v, _ in self.UNITS:
            with self.subTest(name):
                plain, twin, _ = halves(v)
                self.assertEqual(plain.count('dm_bs('), 1, 'only the helper\'s definition (the twins use it)')
                self.assertNotIn('spvDescriptorSet0.' + abi.BORDER_MEMBER, plain)
                self.assertNotRegex(plain, r'spvDescriptorSet0\.\w+' + abi.BORDER_TWIN + r'\b', 'no second sample through a twin sampler')
                self.assertEqual(twin.count('dm_bs('), v.count('dm_bs(') - 1, 'every sample of the unit is wrapped in the twins')

    def test_a_twin_is_the_function_as_it_was(self):
        for name, v, twins in self.UNITS:
            with self.subTest(name):
                _, twin, _ = halves(v)
                functions = [f for f in re.split(r'(?m)^(?=static |(?:\[\[[^\]]*\]\]\s*)?fragment )', twin) if f]
                self.assertEqual(len(functions), len(twins) + 1, 'the body and the entry points')
                for f in functions: self.assertIn(re.sub(r'(?<!dm)_bs\b', '', f).rstrip('\n'), v)

    def test_samples_without_the_wrapper(self):
        plain, twin, _ = halves(UNIT)
        self.assertIn('float2 _45 = spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, select(_182, float2(0.0), isnan(_182))).xy;', plain)
        self.assertIn('float4 _50 = spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, _182, level(0.0)).xy, bias(0.5));', plain,
                      'a sample inside another one\'s arguments too')
        self.assertIn('spvDescriptorSet0.fp_t2d_cb14_0[1].sample(spvDescriptorSet0.fp_t2d_cb14_0Smplr[1], _182)', plain)
        inner = 'dm_bs(spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, _182, level(0.0)), spvDescriptorSet0.dm_border, 0u, spvDescriptorSet0.fp_t_tcb_8, spvDescriptorSet0.fp_t_tcb_8Smplr_w, _182, level(0.0))'
        self.assertEqual(twin.count(inner), 2, 'the twins keep the correction, the inner sample in both samples of the outer one')
        self.assertIn('return dm_body_main0_bs(in, spvDescriptorSet0);', twin)
        self.assertIn('return dm_body_main0(in, spvDescriptorSet0);', plain)

    def test_readable_tile_version(self):
        plain, twin, _ = halves(FOLDED)
        self.assertRegex(plain, r'\(fx0 == 0 \? fb0 : [^;]*fx0 == 7 \? fb7 : spvDescriptorSet0\.fp_t_tcb_8\.sample\(', 'a tile read replaces the sample')
        self.assertRegex(twin, r'\(fx0 == 0 \? fb0 : [^;]*fx0 == 7 \? fb7 : dm_bs\(spvDescriptorSet0\.fp_t_tcb_8\.sample\(', 'or the corrected sample')
        self.assertIn('main0_out r = dm_body_main0_bs(in, spvDescriptorSet0, FX_0, _fb0, _fb1, _fb2, _fb3, _fb4, _fb5, _fb6, _fb7);', twin, 'main0_f_bs passes the body its function constants')
        text = plain + twin
        self.assertEqual(text.count('constant int FX_0 [[function_constant(0)]];'), 1, 'the constants are declared once')
        self.assertEqual(text.count('fragment main0_out_f main0_f('), 1); self.assertEqual(text.count('fragment main0_out_f main0_f_bs('), 1)

    def test_an_entry_point_with_its_body_inside(self):   # no dm_body_main0: the entry point is the function that holds the samples
        text, twins = border_twin(border_emu(GRASS))
        self.assertEqual(twins, ('main0_bs',))
        i = text.index('fragment main0_out main0_bs(')
        self.assertEqual(text[:i].count('dm_bs('), 1); self.assertEqual(text[i:].count('dm_bs('), 4)

    def test_unchanged(self):
        helper = 'static float4 dm_h(constant spvDescriptorSetBuffer0& spvDescriptorSet0)\n{\n    return dm_bs(spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, float2(0.5)), ' \
                 'spvDescriptorSet0.dm_border, 0u, spvDescriptorSet0.fp_t_tcb_8, spvDescriptorSet0.fp_t_tcb_8Smplr_w, float2(0.5));\n}\n'
        for why, v in (('a dm_bs call in a helper function', UNIT.replace('static main0_out dm_body_main0(', helper + 'static main0_out dm_body_main0(')),
                       ('no correction', dmsh_side.unit(GRASS)[0]),
                       ('a kernel', border_emu(KERNEL))):
            with self.subTest(why):
                self.assertEqual(border_twin(v), (v, ()))

    @unittest.skipUnless(shutil.which('xcrun'), 'requires Metal compilers')
    def test_compiles_and_cuts(self):   # for both platforms; the library cut as ps_libraries cuts it holds the twins and only them in <key>_bs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, v, twins in self.UNITS + (('the entry point alone', border_emu(GRASS), ('main0_bs',)),):
                text, got = border_twin(v)
                for sdk in ('macosx', 'iphoneos'):
                    with self.subTest(shader=name, sdk=sdk):
                        try: lib = compile_lib(text, root, f'{name.replace(" ", "_")}_{sdk}', sdk)
                        except subprocess.CalledProcessError as e: self.fail(e.stderr)
                        side = {n for n in ('main0_a', 'main0_l') if n + abi.BORDER_FN in twins}
                        parts = dmsh_side.cut(lib.read_bytes(), *[g for g in (side, {'main0_f'} if 'main0_f_bs' in twins else set(), set(got)) if g])
                        for i, part in enumerate(parts):
                            (root / f'p{i}.metallib').write_bytes(part)
                        names = [sorted(pso_offline.functions(root / f'p{i}.metallib')) for i in range(len(parts))]
                        self.assertEqual(names[-1], sorted(twins), 'the twins')
                        self.assertEqual(names[0], ['main0'])

    @unittest.skipUnless(os.environ.get('DM_TEST_BORDER') and shutil.which('xcrun'), 'set DM_TEST_BORDER to the native t_border executable')
    def test_draws(self):   # t_border: a draw whose samplers need no correction takes the plain entry point, any other the twin, and the pixels are the shader's as they were
        with tempfile.TemporaryDirectory(prefix='border_') as directory:
            fixture(directory)
            r = subprocess.run([os.environ['DM_TEST_BORDER'], directory], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


# t_border's draw test: a pixel shader sampling a 4x4 texture over a full-screen triangle whose texture coordinates run from -0.125 to 1.125 (a ring of border around it), and its vertex shader
PIXEL = HEAD + '''struct spvDescriptorSetBuffer0
{
    texture2d<float> fp_t_tcb_8 [[id(0)]];
    sampler fp_t_tcb_8Smplr [[id(1)]];
};

struct main0_out
{
    float4 m_0 [[color(0)]];
};

struct main0_in
{
    float2 m_1 [[user(locn0)]];
};

fragment main0_out main0(main0_in in [[stage_in]], constant spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]])
{
    main0_out out = {};
    out.m_0 = spvDescriptorSet0.fp_t_tcb_8.sample(spvDescriptorSet0.fp_t_tcb_8Smplr, in.m_1, level(0.0));
    return out;
}
'''
VERTEX = '''
struct vertex_out { float4 position [[position]]; float2 uv [[user(locn0)]]; };

vertex vertex_out vertex_main(uint i [[vertex_id]])
{
    float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    vertex_out o; o.position = float4(p[i], 0, 1); o.uv = p[i] * 0.625 + 0.5;
    return o;
}
'''


def compile_lib(src, root, name, sdk='macosx'):   # → the library path; CalledProcessError with the compiler's message
    msl = root / (name + '.metal'); msl.write_text(src); air = root / (name + '.' + sdk + '.air'); lib = root / (name + '.metallib')
    for cmd in (['xcrun', '-sdk', sdk, 'metal', '-c', '-fmodules-cache-path=' + str(root / 'mc'), str(msl), '-o', str(air)], ['xcrun', '-sdk', sdk, 'metallib', str(air), '-o', str(lib)]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode: raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return lib


def fixture(out):
    """t_border's libraries, with the key 0123456789abcdef: <key>_b (main0 without the correction, and vertex_main), <key>_bx (the alpha test entry point), <key>_bs (the twins of both), cut
    from one compile as ps_libraries cuts them; and old.metallib, the unit as it was (main0 and main0_a with the correction in their samples)"""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    unit, info = dmsh_side.unit(border_emu(PIXEL))
    text, twins = border_twin(unit)
    parts = dmsh_side.cut(compile_lib(text + VERTEX, out, 'new').read_bytes(), set(info['alpha']), set(twins))
    for suffix, part in zip(('_b', '_bx', abi.BORDER_FN), parts): (out / f'0123456789abcdef{suffix}.metallib').write_bytes(part)
    (out / 'old.metallib').write_bytes(compile_lib(unit + VERTEX, out, 'old').read_bytes())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fixture': fixture(sys.argv[2])
    else: unittest.main()
