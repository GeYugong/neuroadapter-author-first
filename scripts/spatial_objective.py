"""Fixed auxiliary spatial objective. No parameter updates or RNG draws here."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
LAMBDA=0.05
TMAX=250
SIZE=64

def auxiliary(noise_pred,noisy_latents,timesteps,scheduler,vae,images):
    assert scheduler.config.prediction_type=='epsilon'
    assert not any(p.requires_grad for p in vae.parameters())
    alpha=scheduler.alphas_cumprod.to(device=noise_pred.device,dtype=torch.float32)[timesteps].reshape(-1,1,1,1)
    clean=(noisy_latents.float()-(1-alpha).sqrt()*noise_pred.float())/alpha.sqrt()
    assert images.shape[-2:]==(512,512)
    losses=[]
    for i in range(noise_pred.shape[0]):
        if int(timesteps[i])<=TMAX:
            def decode(z):
                return F.avg_pool2d(vae.decode(z/vae.config.scaling_factor).sample,kernel_size=8,stride=8)
            prediction=checkpoint(decode,clean[i:i+1],use_reentrant=False,preserve_rng_state=True)
            target=F.avg_pool2d(images[i:i+1].float(),kernel_size=8,stride=8)
            losses.append(F.l1_loss(prediction,target))
    loss=torch.stack(losses).sum()/noise_pred.shape[0] if losses else noise_pred.sum()*0.
    assert torch.isfinite(loss)
    return loss,len(losses)

def install(t,weight=LAMBDA,observer=None):
    import inspect
    original=t.process_training_batch
    source=inspect.getsource(original)
    assert source.count("    train_loss = loss.mean()")==1
    def added(pred,noisy,ts,scheduler,vae,images,base):
        if weight==0:
            if observer:observer(base,base*0,0,ts)
            return base
        extra,n=auxiliary(pred,noisy,ts,scheduler,vae,images)
        if observer:observer(base,extra,n,ts)
        return base+weight*extra
    t.spatial_auxiliary=added
    source=source.replace("    train_loss = loss.mean()","    train_loss = loss.mean()\n    train_loss = spatial_auxiliary(noise_pred,noisy_latents,timesteps,noise_scheduler,vae,img_ipadapter,train_loss)")
    exec(compile(source,"<spatial_process_training_batch>","exec"),t.__dict__)
    return original
