"""Redirect official model asset locations only; preserve original model methods."""
from pathlib import Path
import functools

B = Path(__file__).resolve().parent


def configure():
    import clip
    from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
    local = B / 'assets/clip-vit-large-patch14'
    assert (local / 'pytorch_model.bin').is_file()
    for cls in (CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer):
        original = cls.from_pretrained

        def redirected(_cls, name, *args, _original=original, **kwargs):
            if str(name) == 'openai/clip-vit-large-patch14':
                name = str(local)
                kwargs['local_files_only'] = True
            return _original(name, *args, **kwargs)

        cls.from_pretrained = classmethod(redirected)
    original_load = clip.load

    @functools.wraps(original_load)
    def load(*args, **kwargs):
        kwargs.setdefault('download_root', str(B / 'assets/mindbridge/clip'))
        return original_load(*args, **kwargs)

    clip.load = load
