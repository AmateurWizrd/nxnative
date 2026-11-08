// Border colours Metal has no sampler state for (nvn_border.h, dmsh_build.py border_emu): nvn_border's choice of Metal colours and factors, the production sampler registry's twin
// (mtl_reg_samp, samp_by_id), and, given the path of a shader border_emu rewrote (test_dmsh_border.py writes it), the GPU: samples through the sampler, its twin and the block come
// out as Maxwell's do, bilinear between the texels and the border colour, and with the block's mask clear as Metal's nearest colour. A second path is a shader that must give the same
// samples bit for bit (the first one after border_split, a kernel's body twice). Given a directory (test_dmsh_border.py --fixture) it draws the border twins (border_twin): a pixel
// shader's pipeline for a draw that needs no correction has none in it, a draw that does takes the twin (the alpha test's entry point has its twin too), the pixels are those of the shader as
// it was, and the twin's pipeline is named in pso_user.tsv
#include "../nvn/nvn_mtl.m"
uint64_t g_t0_ns;
int dm_guest_as_contains(uintptr_t address) { (void)address; return 0; }
void *front_metal_layer(void) { return NULL; }
module_t *module_of(const void *p) { (void)p; return NULL; }
module_t *g_mod; int g_nmod;
void svc_thread_dump(void) {}
uint64_t now_ns(void) { return bp_now(); }
static int checks, fails;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; fprintf(stderr, "FAIL: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { W = 4, H = 4, NUV = 9 };
static const float UV[NUV][2] = { {-0.1f, 0.5f}, {0.5f, 1.05f}, {1.2f, 1.2f}, {0.06f, 0.94f}, {0.5f, 0.5f}, {0.f, 0.f}, {1.f, 0.3f}, {0.97f, -0.02f}, {0.3f, 0.6f} };
static float texel(int x, int y, int c) { const float v[4] = { (float)x / W, (float)y / H, (float)(x + y) / 8, 1 - (float)x / 8 }; return v[c]; }
static void reference(const float B[4], float u, float v, float out[4]) {   // Maxwell's linear filter: texel centres at (i + 0.5) / size, texels outside read the border colour
    float x = u * W - 0.5f, y = v * H - 0.5f; int x0 = (int)floorf(x), y0 = (int)floorf(y); float fx = x - x0, fy = y - y0;
    for (int c = 0; c < 4; c++) {
        float t[2][2];
        for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) { int xi = x0 + i, yj = y0 + j; t[j][i] = xi < 0 || yj < 0 || xi >= W || yj >= H ? B[c] : texel(xi, yj, c); }
        out[c] = (t[0][0] * (1 - fx) + t[0][1] * fx) * (1 - fy) + (t[1][0] * (1 - fx) + t[1][1] * fx) * fy;
    }
}

static void choice(const float B[4], uint8_t wrap, int reduction, int want_twin, int want_bn) {
    const uint8_t w[3] = { wrap, wrap, 0 }; int bn = -1, bw = -1; float k[4] = {0};
    int twin = nvn_border(B, w, reduction, &bn, &bw, k);
    CHECK(twin == want_twin, "border %g,%g,%g,%g wrap %d reduction %d: twin %d, expected %d", B[0], B[1], B[2], B[3], wrap, reduction, twin, want_twin);
    if (want_bn >= 0) CHECK(bn == want_bn, "border %g,%g,%g,%g: Metal colour %d, expected %d", B[0], B[1], B[2], B[3], bn, want_bn);
    if (!twin) return;
    CHECK(bw != bn && bw >= 0 && bw < 3, "border %g,%g,%g,%g: twin colour %d beside %d", B[0], B[1], B[2], B[3], bw, bn);
    for (int c = 0; c < 4; c++) {   // the correction gives B back where only the border is read
        float got = nvn_mtl_border[bn][c] + (nvn_mtl_border[bw][c] - nvn_mtl_border[bn][c]) * k[c];
        CHECK(fabsf(got - B[c]) < 1e-6f, "border %g,%g,%g,%g channel %d: corrected %g", B[0], B[1], B[2], B[3], c, got);
    }
}

static id<MTLTexture> g_tex;
static void gpu_case(id<MTLComputePipelineState> p, id<MTLComputePipelineState> same, int sid, const float B[4], int mask_on) {
    samp_par sp = {0}; sp.magf = 1; sp.minf = 1; sp.ws = sp.wt = sp.wr = 5; sp.lod_max = 0; memcpy(sp.border, B, 16);
    mtl_reg_samp(sid, &sp);
    os_unfair_lock_lock(&g_rmu);
    id<MTLSamplerState> s = samp_by_id(sid); int twin = g_sview_w[sid] != NULL; uint64_t rid = g_srid[sid], rid_w = twin ? g_srid_w[sid] : rid; float k[4]; memcpy(k, twin ? g_sbk[sid] : (float[4]){0}, 16);
    os_unfair_lock_unlock(&g_rmu);
    CHECK(s != nil, "sampler %d created", sid);
    int bn, bw; float kk[4]; const uint8_t w[3] = { 5, 5, 5 }; int want = nvn_border(B, w, 0, &bn, &bw, kk);
    CHECK(twin == want, "sampler %d: twin %d, nvn_border says %d", sid, twin, want);
    id<MTLBuffer> block = [g_dev newBufferWithLength:DMSH_BORDER_K + 16 options:MTLResourceStorageModeShared];
    uint32_t mask = twin && mask_on; memcpy(block.contents, &mask, 4); memcpy((uint8_t *)block.contents + DMSH_BORDER_K, k, 16);
    id<MTLBuffer> args = [g_dev newBufferWithLength:32 options:MTLResourceStorageModeShared];   // the members as border_emu lays them out: texture, sampler, its twin, the block
    uint64_t a[4] = { g_tex.gpuResourceID._impl, rid, rid_w, block.gpuAddress }; memcpy(args.contents, a, 32);
    id<MTLBuffer> uv = [g_dev newBufferWithBytes:UV length:sizeof UV options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = nil, out2 = nil;
    for (int n = 0; n < (same ? 2 : 1); n++) {
        id<MTLBuffer> o = [g_dev newBufferWithLength:16 * NUV options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = q_cb(); id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:n ? same : p]; [e setBuffer:args offset:0 atIndex:0]; [e setBuffer:o offset:0 atIndex:1]; [e setBuffer:uv offset:0 atIndex:2];
        [e useResource:g_tex usage:MTLResourceUsageRead]; [e useResource:block usage:MTLResourceUsageRead];
        [e dispatchThreads:MTLSizeMake(NUV, 1, 1) threadsPerThreadgroup:MTLSizeMake(NUV, 1, 1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        CHECK(cb.status == MTLCommandBufferStatusCompleted, "sampler %d: GPU work completed", sid);
        if (n) out2 = o; else out = o;
    }
    const float *got = out.contents; const float *Bref = mask ? B : nvn_mtl_border[bn];   // the mask clear: Metal's nearest colour
    for (int i = 0; i < NUV; i++) { float ref[4]; reference(Bref, UV[i][0], UV[i][1], ref);
        for (int c = 0; c < 4; c++) CHECK(fabsf(got[4 * i + c] - ref[c]) < 6e-3f, "border %g,%g,%g,%g mask %u at (%g, %g) channel %d: %g, expected %g", B[0], B[1], B[2], B[3], mask, UV[i][0], UV[i][1], c, got[4 * i + c], ref[c]); }
    if (same) CHECK(!memcmp(out.contents, out2.contents, 16 * NUV), "sampler %d mask %u: the second shader's samples are not the first's", sid, mask);
}
static id<MTLComputePipelineState> pipeline(const char *path) {
    NSError *error = nil; NSString *src = [NSString stringWithContentsOfFile:@(path) encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> lib = src ? [g_dev newLibraryWithSource:src options:nil error:&error] : nil;
    id<MTLComputePipelineState> p = lib ? [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"main0"] error:&error] : nil;
    CHECK(p != nil, "the shader %s compiles: %s", path, error.localizedDescription.UTF8String);
    return p;
}

// The twins through a draw (test_dmsh_border.py --fixture): a pixel shader as the build cuts its libraries (<key>_b without the correction, <key>_bx the alpha test, <key>_bs the twins of both),
// drawn through pso_get and encode_args with a sampler whose border color Metal has (the plain entry point, no correction in its pipeline) and one it lacks (the twin), against the shader as it was
// (old.metallib: the corrections in its samples) drawn the same way
enum { DW = 8 };
static id<MTLTexture> tw_tex(tex_t *t, uint64_t serial, int w, int h) {
    *t = (tex_t){.magic = MAGIC('T', 'E', 'X', 0), .serial = serial};
    t->d = (nvn_texdesc){.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA8, .width = w, .height = h, .depth = 1, .levels = 1, .samples = 1};
    MTLTextureDescriptor *d = tex_desc(t, 0); d.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [g_dev newTextureWithDescriptor:d];
    res_add(texture); mtl_tex_init(t); serial_put(serial, texture);
    return texture;
}
static NvProg *draw_prog(id<MTLLibrary> lib, int twins) {   // twins: the reflection's border_twin
    NvProg *pr = [NvProg new]; NvShader *vs = [NvShader new], *fs = [NvShader new]; pr->sh[0] = vs; pr->sh[1] = fs;
    vs->stage = 0; strcpy(vs->key, "0123456789abcdef_b"); vs->fn = fn4_note([lib newFunctionWithName:@"vertex_main"], lib, @"vertex_main", nil); vs->rf.pos_attr = -1;
    fs->stage = 1; strcpy(fs->key, "0123456789abcdef_b"); fs->fn = fn4_note([lib newFunctionWithName:@"main0"], lib, @"main0", nil); fs->has_bs = twins; fs->has_alpha = twins;
    fs->rf.nb = 4; fs->rf.abi = 0; fs->rf.border = 4; fs->rf.color_known = 1; fs->rf.color_mask = 1;
    fs->rf.b[0] = (nvn_rbind){.kind = NVN_RB_TEX, .reg = 0, .aid = 0, .dtt = MTLTextureType2D};
    fs->rf.b[1] = (nvn_rbind){.kind = NVN_RB_SAMPT, .reg = 0, .aid = 1};
    fs->rf.b[2] = (nvn_rbind){.kind = NVN_RB_SAMPT, .reg = 0, .aid = 2, .bw = 1};   // the twin of the sampler, bit 0 of the block
    fs->rf.b[3] = (nvn_rbind){.kind = NVN_RB_BORDER, .aid = 3};
    fs->ae = [fs->fn newArgumentEncoderWithBufferIndex:0]; fs->arg_len = fs->ae.encodedLength;
    CHECK(fs->arg_len == 32, "the argument buffer of the entry point that never reads the twin members has all four (shader_load's own check): %lu bytes", (unsigned long)fs->arg_len);
    return pr;
}
// One draw of the program over the target with the handle (texture 700, sampler id << 20); the pipeline, its key, and the target's pixels
static id<MTLRenderPipelineState> tw_draw(NvProg *pr, tex_t *target, id<MTLTexture> rt, uint32_t handle, uint8_t px[256 * DW], pso_key *key) {
    state_reset(); memset(&R, 0, sizeof R); mtl_sub_begin();
    R.has_rt = 1; R.rt.n = 1; R.rt.color[0] = target; S.tex[1][0] = handle; S.chmask.mask = 0xffffffffu;
    R.clear_c[0] = 1; R.cc[0] = MTLClearColorMake(0, 0, 0, 0);
    CHECK(rp_begin() != nil, "the render pass opens");
    pso_key_fill(key, pr->sh[0], pr->sh[1], g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, 0);
    id<MTLRenderPipelineState> ps = pso_get(pr, nil);
    CHECK(ps != nil, "the pipeline builds");
    if (ps) {
        enc_r_pso(g_enc, ps); enc_r_viewport(g_enc, (MTLViewport){0, 0, DW, DW, 0, 1}); args_dirty_all();
        CHECK(encode_args((id<MTLCommandEncoder>)g_enc, pr->sh[1], 1, 0), "fragment arguments encoded");
        enc_r_draw(g_enc, MTLPrimitiveTypeTriangle, 0, 3);
    }
    rp_end();
    id<MTLBuffer> b = [g_dev newBufferWithLength:256 * DW options:MTLResourceStorageModeShared]; res_add(b);
    enc_b_copy_texture_buffer(blit(), rt, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(DW, DW, 1), b, 0, 256, 256 * DW);
    CHECK(mtl_wait(mtl_sub_commit(), 5000000000ull), "GPU submission completed");
    CHECK(!mtl_last_commit_error(), "GPU submission has no error");
    memcpy(px, b.contents, 256 * DW);
    return ps;
}
static int same_pixels(const uint8_t *a, const uint8_t *b) { for (int y = 0; y < DW; y++) if (memcmp(a + 256 * y, b + 256 * y, 4 * DW)) return 0; return 1; }
static NSString *fragment_name(NvProg *pr, const pso_key *k) { return pso_desc(k, pr->sh[0], pr->sh[1], nil, 0).fragmentFunction.name; }
static void draw_cases(const char *dir) {
    NSString *d = @(dir); g_msl_dir = d;   // bs_fn loads <key>_bs.metallib, at_fn <key>_bx.metallib from here
    NSString *user = [d stringByAppendingPathComponent:@"pso_user.tsv"]; setenv("DM_PSO_USER", user.fileSystemRepresentation, 1);
    NSError *e = nil;
    id<MTLLibrary> nl = lib_from_file([d stringByAppendingPathComponent:@"0123456789abcdef_b.metallib"], &e), ol = lib_from_file([d stringByAppendingPathComponent:@"old.metallib"], &e);
    CHECK(nl && ol, "the fixture's libraries load: %s", e.localizedDescription.UTF8String);
    if (!nl || !ol) return;
    CHECK([nl newFunctionWithName:@"main0_bs"] == nil, "the base library holds no twin");
    tex_t src, tgt; id<MTLTexture> st = tw_tex(&src, 99900, 4, 4), rt = tw_tex(&tgt, 99901, DW, DW);
    uint8_t px[4][4][4]; for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) memcpy(px[y][x], (uint8_t[4]){(uint8_t)(40 + 50 * x), (uint8_t)(40 + 50 * y), (uint8_t)(30 * (x + y)), 255}, 4);
    [st replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:px bytesPerRow:16];
    mtl_reg_tex(700, &src, NULL);
    samp_par own = {0}, lacks = {0}; own.magf = own.minf = lacks.magf = lacks.minf = 1; own.ws = own.wt = own.wr = lacks.ws = lacks.wt = lacks.wr = 5;   // linear, clamp to border
    memcpy(lacks.border, (float[4]){0.5f, 0.25f, 0.75f, 1}, 16); memcpy(own.border, (float[4]){0, 0, 0, 1}, 16);   // opaque black is Metal's own
    mtl_reg_samp(31, &own); mtl_reg_samp(32, &lacks);
    os_unfair_lock_lock(&g_rmu); samp_by_id(31); samp_by_id(32); int twin31 = g_sview_w[31] != NULL, twin32 = g_sview_w[32] != NULL; os_unfair_lock_unlock(&g_rmu);   // the twin is made with the sampler
    CHECK(!twin31 && twin32, "sampler 31 (Metal's color) has no twin, sampler 32 has one");
    NvProg *np = draw_prog(nl, 1), *op = draw_prog(ol, 0);
    uint8_t a[256 * DW], b[256 * DW], c[256 * DW]; pso_key kn, ko;
    // no border color to correct: the plain entry point
    id<MTLRenderPipelineState> plain = tw_draw(np, &tgt, rt, 700 | 31u << 20, a, &kn); tw_draw(op, &tgt, rt, 700 | 31u << 20, b, &ko);
    CHECK(kn.bs == 0 && ko.bs == 0, "no twin is needed: key bs %u, old %u", kn.bs, ko.bs);
    CHECK([fragment_name(np, &kn) isEqualToString:@"main0"], "the plain entry point is the fragment function: %s", fragment_name(np, &kn).UTF8String);
    CHECK(same_pixels(a, b), "mask clear: the plain entry point draws what the shader drew");
    // a sampler whose border color Metal lacks: the twin, and what the shader drew
    id<MTLRenderPipelineState> twin = tw_draw(np, &tgt, rt, 700 | 32u << 20, a, &kn); tw_draw(op, &tgt, rt, 700 | 32u << 20, b, &ko);
    CHECK(kn.bs == 1 && ko.bs == 0, "a twin is needed: key bs %u, old %u", kn.bs, ko.bs);
    CHECK([fragment_name(np, &kn) isEqualToString:@"main0_bs"], "the twin is the fragment function: %s", fragment_name(np, &kn).UTF8String);
    CHECK(twin && plain && twin != plain, "another pipeline for the twin");
    CHECK(same_pixels(a, b), "mask set: the twin draws what the shader drew");
    np->sh[1]->has_bs = 0; tw_draw(np, &tgt, rt, 700 | 32u << 20, c, &kn); np->sh[1]->has_bs = 1;   // the plain entry point with the mask set draws no correction
    CHECK(kn.bs == 0 && !same_pixels(c, b), "the plain entry point leaves the border as Metal has it, so the twin's pixels differ from it");
    CHECK(tw_draw(np, &tgt, rt, 700 | 31u << 20, c, &kn) == plain, "the plain pipeline is the one made before");
    // the variants compose: the alpha test's entry point has its twin, and a draw whose key says plain takes the plain one
    uint32_t cf[8] = {MTLPixelFormatRGBA8Unorm};
    S.tex[1][0] = 700 | 32u << 20; S.color.alpha_func = NVN_ALPHA_FUNC_GREATER; pso_key_fill(&kn, np->sh[0], np->sh[1], cf, 0, 0, 1, 0);
    CHECK(kn.atest == 1 && kn.bs == 1 && [fragment_name(np, &kn) isEqualToString:@"main0_a_bs"], "alpha test with a twin: %s", fragment_name(np, &kn).UTF8String);
    S.tex[1][0] = 700 | 31u << 20; pso_key_fill(&kn, np->sh[0], np->sh[1], cf, 0, 0, 1, 0);
    CHECK(kn.atest == 1 && kn.bs == 0 && [fragment_name(np, &kn) isEqualToString:@"main0_a"], "alpha test without: %s", fragment_name(np, &kn).UTF8String);
    S.color.alpha_func = NVN_ALPHA_FUNC_ALWAYS;
    // the manifest: the twin's pipeline is named by a row that parses with bs set, and the plain one's with it clear
    NSString *rows = [NSString stringWithContentsOfFile:user encoding:NSUTF8StringEncoding error:nil]; int with = 0, without = 0;
    for (NSString *line in [rows componentsSeparatedByString:@"\n"]) {
        NSArray *col = [line componentsSeparatedByString:@"\t"]; pso_skey sk;
        if (col.count == 2 && [col[0] isEqualToString:@"R"] && !pso_skey_parse([col[1] UTF8String], &sk)) { if (sk.k.bs) with++; else without++; }
    }
    CHECK(with == 1 && without >= 1, "pso_user.tsv has the twin's pipeline once (%d) and the plain ones (%d)", with, without);
    printf("border twins: draws with the mask clear and set\n");
}

int main(int argc, char **argv) { @autoreleasepool {
    choice((float[4]){0.5f, 0.5f, 0.5f, 0.5f}, 5, 0, 1, 0);   // Tears of the Kingdom's grass bend map
    choice((float[4]){0.25f, 0.5f, 0.75f, 1}, 5, 0, 1, 1);     // opaque: kept with opaque black, corrected with opaque white
    choice((float[4]){0.8f, 0.9f, 0.7f, 0.6f}, 4, 0, 1, 2);
    choice((float[4]){1, 0, 0, 1}, 5, 0, 1, -1);
    choice((float[4]){0.2f, 0, 0, 1}, 5, 0, 1, 1);             // opaque black and opaque white differ in red only where it matters
    choice((float[4]){0, 0, 0, 0}, 5, 0, 0, 0); choice((float[4]){0, 0, 0, 1}, 5, 0, 0, 1); choice((float[4]){1, 1, 1, 1}, 5, 0, 0, 2);   // Metal's own
    choice((float[4]){0.5f, 0.5f, 0.5f, 0.5f}, 0, 0, 0, 0);   // the border is never read
    choice((float[4]){0.5f, 0.5f, 0.5f, 0.5f}, 5, 1, 0, 0);   // a min reduction is not linear in it
    mtl_init();
    struct stat sb;
    if (argc > 1 && !stat(argv[1], &sb) && S_ISDIR(sb.st_mode)) draw_cases(argv[1]);   // the fixture of the twins' draw test
    else if (argc > 1) {   // the rewritten shader
        id<MTLComputePipelineState> p = pipeline(argv[1]), same = argc > 2 ? pipeline(argv[2]) : nil;
        if (p && (argc < 3 || same)) {
            MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:W height:H mipmapped:NO];
            d.storageMode = MTLStorageModeShared; g_tex = [g_dev newTextureWithDescriptor:d];
            float px[H][W][4]; for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) for (int c = 0; c < 4; c++) px[y][x][c] = texel(x, y, c);
            [g_tex replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:px bytesPerRow:16 * W];
            if (![g_dev supports32BitFloatFiltering]) { fprintf(stderr, "t_border: this GPU does not filter RGBA32Float, GPU cases skipped\n"); }
            else {
                gpu_case(p, same, 11, (float[4]){0.5f, 0.5f, 0.5f, 0.5f}, 1);
                gpu_case(p, same, 11, (float[4]){0.5f, 0.5f, 0.5f, 0.5f}, 0);
                gpu_case(p, same, 12, (float[4]){0.25f, 0.5f, 0.75f, 1}, 1);
                gpu_case(p, same, 12, (float[4]){0.25f, 0.5f, 0.75f, 1}, 0);
                gpu_case(p, same, 13, (float[4]){0.8f, 0.9f, 0.7f, 0.6f}, 1);
                gpu_case(p, same, 14, (float[4]){0, 0, 0, 1}, 1);   // Metal's own: no twin, the plain sample
            }
        }
    }
    printf("t_border: %d checks, %d failed\n", checks, fails);
    return fails != 0;
} }
