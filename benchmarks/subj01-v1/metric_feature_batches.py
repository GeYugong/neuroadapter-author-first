"""Memory adapter only: original metric functions retain the full negative pool."""
import torch

def install(module, batch_size):
    def wrapped_model(model):
        def forward(images):
            outputs=[]
            for part in images.split(batch_size):
                result=model(part)
                outputs.append({k:v.detach().cpu() for k,v in result.items()} if isinstance(result,dict) else result.detach().cpu())
            if isinstance(outputs[0],dict):
                return {k:torch.cat([x[k] for x in outputs],dim=0) for k in outputs[0]}
            return torch.cat(outputs,dim=0)
        return forward
    for name in ('compute_two_way_identification','compute_feature_similarity'):
        original=getattr(module,name)
        def wrapper(first,second,model,preprocess,*args,_original=original,**kwargs):
            return _original(first,second,wrapped_model(model),preprocess,*args,**kwargs)
        setattr(module,name,wrapper)
