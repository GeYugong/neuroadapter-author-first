"""Compile the pinned author's K1 generation prefix without a GT argument.

This module does not load models, start training, or generate benchmark outputs.
The eventual runner must separately verify full600 training completion and GPUs.
"""
import ast
import copy
import hashlib
from pathlib import Path

SOURCE_SHA256 = 'cdf5f71c5a0c8d30b3dc3deb16e43f0b019f7abbda524bff4825c2c840021a99'
B = Path(__file__).resolve().parent


def compile_native(torch, source=None):
    """Return (GT-free K1 function, unmodified generation-prefix reference, audit)."""
    source = source or B / 'upstream/mindbridge/src/utils.py'
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA256, 'Pinned author source changed'
    tree = ast.parse(raw)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reconstruction')
    decode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'decode_latents')
    stop = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'best_picks' for t in n.targets))
    reference = copy.deepcopy(fn)
    reference.name = 'native_generation_prefix'
    reference.body = reference.body[:stop] + [ast.Return(value=ast.Name(id='brain_recons', ctx=ast.Load()))]

    class FixedVanilla(ast.NodeTransformer):
        def __init__(self):
            self.counts = dict(image_slice=0, memory_branch=0, clip_device=0, lowlevel_branch=0)

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'image' for t in node.targets):
                assert ast.unparse(node) == 'image = image[:n_samples_save]'
                self.counts['image_slice'] += 1
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == 'mem_efficient':
                self.counts['memory_branch'] += 1
                result = []
                for statement in node.orelse:  # original recon.py always passes False
                    changed = self.visit(statement)
                    if isinstance(changed, list):
                        result.extend(changed)
                    elif changed is not None:
                        result.append(changed)
                return result
            if ast.unparse(node.test) == 'img_lowlevel is not None':
                self.counts['lowlevel_branch'] += 1
                return [self.visit(n) for n in node.orelse]  # original Vanilla has no AE
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if ast.unparse(node) == 'clip_extractor.to(device)':
                self.counts['clip_device'] += 1
                return None
            return self.generic_visit(node)

    adapter = copy.deepcopy(reference)
    adapter.name = 'reconstruct_k1'
    adapter.args = ast.parse('def f(voxel, voxel2clip, unet, vae, noise_scheduler, *, device): pass').body[0].args
    transform = FixedVanilla()
    adapter = transform.visit(adapter)
    assert transform.counts == dict(image_slice=1, memory_branch=4, clip_device=1, lowlevel_branch=1), transform.counts
    fixed = ast.parse('''
n_samples_save = 1
recons_per_sample = 1
num_inference_steps = 20
guidance_scale = 5.0
seed = 42
verbose = False
assert voxel2clip is not None and unet is not None and vae is not None
assert voxel.shape == (1, 8192)
''').body
    adapter.body = fixed + adapter.body
    forbidden = {'image', 'clip_extractor', 'img_lowlevel', 'plotting', 'best_picks', 'mem_efficient'}
    names = {n.id for n in ast.walk(adapter) if isinstance(n, ast.Name)}
    assert not names.intersection(forbidden), names.intersection(forbidden)
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(decode), reference, adapter], type_ignores=[]))
    namespace = {'torch': torch}
    exec(compile(module, str(source), 'exec'), namespace)
    audit = dict(source_sha256=SOURCE_SHA256, changes=transform.counts,
                 adapter_ast_sha256=hashlib.sha256(ast.dump(adapter).encode()).hexdigest(),
                 protocol=dict(steps=20, cfg=5, seed=42, K=1, lowlevel=False, memory_efficient=False),
                 gt_argument=False, selector_and_plotting_removed=True)
    return namespace['reconstruct_k1'], namespace['native_generation_prefix'], audit
