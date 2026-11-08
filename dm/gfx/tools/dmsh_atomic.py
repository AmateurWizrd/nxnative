"""Image atomics run natively on the Metal texture (MSL 3.1 texture atomics).

SPIRV-Cross below MSL 3.1 emulates an image atomic with a raw device pointer
beside the texture (<image>_atomic), indexed by the bound texture's width: a
block-linear image is a deswizzled Metal copy whose pool memory has another
layout, so those atomics went to a scratch buffer that nothing read, and a 3D
image did not compile at all (an int3 subscript). Native texture atomics act on
the texture every other access of the image uses, for buffer, 2D, 2D-array and
3D images. The translator still guards each coordinate (IGN bounds) before the
atomic.

Only shaders with image atomics (OpImageTexelPointer) are written at MSL 3.1:
every other shader keeps its MSL 2.3 source byte for byte, so its library comes
from the MSL-keyed compile cache. The emulation must never come back silently:
check_msl fails the shader when SPIRV-Cross writes the pointer form anyway.
"""
import re
import struct

MSL_VERSION = '20300'
MSL_VERSION_ATOMICS = '30100'
OP_IMAGE_TEXEL_POINTER = 60


def uses_image_atomics(spv):
    """Whether the SPIR-V module (path) has an image texel pointer, the operand of every image atomic"""
    with open(spv, 'rb') as f:
        data = f.read()
    if len(data) < 20 or data[:4] != b'\x03\x02\x23\x07':
        raise ValueError('not a SPIR-V module: ' + spv)
    if b'\x3c\x00\x06\x00' not in data:   # the instruction's first word (always 6 words long, opcode 60), nowhere in the module: most shaders, without walking their instructions in Python
        return False
    words = struct.unpack('<%dI' % (len(data) // 4), data[:len(data) // 4 * 4])
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xffff
        if op == OP_IMAGE_TEXEL_POINTER:
            return True
        if n == 0:
            raise ValueError('malformed SPIR-V instruction in ' + spv)
        i += n
    return False


def cross_args(xc, spv):
    """spirv-cross arguments for this module: xc with its --msl-version raised to 3.1 when the module has image atomics"""
    if not uses_image_atomics(spv):
        return list(xc)
    args = list(xc)
    args[args.index('--msl-version') + 1] = MSL_VERSION_ATOMICS
    return args


_EMULATED = re.compile(r'\bspvImage2DAtomicCoord\b|device\s+atomic_u?int\s*\*\s*\w+_atomic\b')


def check_msl(source):
    """Fail closed on SPIRV-Cross's pointer emulation of image atomics; return the source unchanged"""
    if 'tomic' in source and _EMULATED.search(source):   # (both alternatives hold the word; the pattern alone tries every character of the source)
        raise ValueError('image atomics emulated through a raw pointer; they must be native texture atomics (MSL 3.1)')
    return source


# SPIRV-Cross's form of the translator's float add (InstEmitMemory.cs EmitAtomicAddF32): a compare-and-swap loop on the word, the outer loop retrying until the word was the one read. Several
# lanes (and every SIMD-group of the dispatch) on one word retry each other; Ocarina of Time's scene analysis (AglShader's probe and shadow sums) adds to a few words from every lane or SIMD-group.
_FADD = re.compile(
    r'(?P<i>[ \t]*)do\n(?P=i)\{\n'
    r'(?P<pre1>(?:(?P=i)    [^{}\n]*;\n)*?)'   # the address, worked out in the loop (an indexed word); the load and the compare-and-swap each work it out
    r'(?P=i)    float (?P<old>_\d+) = as_type<float>\((?P<load>[^\n]+)\);\n'
    r'(?P=i)    float (?P<sum>_\d+) = (?P=old) \+ (?P<val>[^;\n]+);\n'
    r'(?P<pre2>(?:(?P=i)    [^{}\n]*;\n)*?)'
    r'(?P=i)    uint (?P<exp>_\d+) = as_type<uint>\((?P=old)\);\n'
    r'(?P=i)    uint (?P<cur>_\d+);\n'
    r'(?P=i)    do\n(?P=i)    \{\n(?P=i)        (?P=cur) = (?P=exp);\n'
    r'(?P=i)    \} while \(!atomic_compare_exchange_weak_explicit\(\(device atomic_uint\*\)&(?P<addr>[^\n]+), &(?P=cur), as_type<uint>\((?P=sum)\), memory_order_relaxed, memory_order_relaxed\) && (?P=cur) == (?P=exp)\);\n'
    r'(?P=i)    (?P<fdecl>uint )?(?P<found>_\d+) = (?P=cur);\n'
    r'(?P=i)    (?P<flag>_\d+) = int\((?P=found)\) != as_type<int>\((?P=old)\);\n'
    r'(?P=i)\} while \((?P=flag)\);\n')


def float_adds(source):
    """Each float add loop as one native atomic_float add (Metal 3, Apple GPUs).

    The loop costs a read, a compare-and-swap and a retry for every add that lost the word to another; Metal adds a float in one instruction. The order of the additions was never fixed (the
    atomics of different threads and SIMD-groups run in any order), so the sums differ in rounding only, as two runs of the loop differ.
    Not a SIMD-group sum and one add per group, which the lanes' same-word adds look made for: on this GPU the native add of every lane to one word takes as long (the hardware merges them).
    A loop of another shape, or one whose names are read elsewhere, stays what it was.
    """
    if 'atomic_compare_exchange_weak_explicit' not in source:
        return source

    def one(m):
        i = m['i']
        used = lambda n: len(re.findall(r'\b%s\b' % n, source)) > len(re.findall(r'\b%s\b' % n, m[0]))   # also outside the loop
        if any(used(m[n]) for n in ('old', 'sum', 'exp', 'cur')): return m[0]   # a name the rewrite removes is read elsewhere
        pre = re.sub('(?m)^' + i + '    ', i, m['pre1'] + m['pre2'])   # kept, outside the loop now: statements of single values
        if re.search(r'\b(%s)\b' % '|'.join(m[n] for n in ('old', 'sum', 'exp', 'cur', 'found', 'flag')), pre): return m[0]
        found = f"{i}{m['fdecl'] or ''}{m['found']} = as_type<uint>({m['old']});\n" if m['fdecl'] is None or used(m['found']) else ''   # the loop's result, if read (or declared before the loop: it may be)
        return pre + f"{i}float {m['old']} = atomic_fetch_add_explicit((device atomic_float*)&{m['addr']}, {m['val']}, memory_order_relaxed);\n" + found
    return _FADD.sub(one, source)
