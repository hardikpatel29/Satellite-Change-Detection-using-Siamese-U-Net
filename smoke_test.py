"""
smoke_test.py - Pipeline validation script
Run: python smoke_test.py
"""
import sys, torch, yaml, os
sys.path.insert(0, 'src')

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

# 1. Dataset loading
print("1. Testing dataset loading ...")
from dataset import LEVIRCDDataset
ds = LEVIRCDDataset(
    zip_path=cfg['dataset']['zip_path'],
    split='train',
    image_size=256,
    use_registration=False,
    inner_root=cfg['dataset']['zip_inner_root']
)
print(f"   Train dataset: {len(ds)} samples")
sample = ds[0]
print(f"   img_a shape : {sample['img_a'].shape}")
print(f"   img_b shape : {sample['img_b'].shape}")
print(f"   mask shape  : {sample['mask'].shape}")
print(f"   mask unique : {sample['mask'].unique().tolist()}")
print("   [OK] Dataset loading")

# 2. Forward pass
print()
print("2. Testing model forward pass ...")
from model import build_model
model = build_model(cfg)
model.eval()
ia = sample['img_a'].unsqueeze(0)
ib = sample['img_b'].unsqueeze(0)
with torch.no_grad():
    out = model(ia, ib)
print(f"   Input  : {ia.shape}")
print(f"   Output : {out.shape}")
print("   [OK] Forward pass")

# 3. Loss
print()
print("3. Testing loss function ...")
from losses import build_loss
loss_fn = build_loss(cfg, pos_weight=23.5)
mask_t = sample['mask'].unsqueeze(0)
loss = loss_fn(out, mask_t)
print(f"   Loss value: {loss.item():.4f}")
print("   [OK] Loss computation")

# 4. Backward
print()
print("4. Testing backpropagation ...")
model.train()
out2 = model(ia, ib)
loss2 = loss_fn(out2, mask_t)
loss2.backward()
grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
print(f"   Gradient norm mean: {sum(grads)/len(grads):.6f}")
print("   [OK] Backpropagation")

# 5. Metrics
print()
print("5. Testing metrics ...")
from metrics import compute_metrics, MetricAccumulator
met = compute_metrics(out, mask_t, threshold=0.5)
print(f"   iou={met['iou']:.4f}, dice={met['dice']:.4f}")
acc = MetricAccumulator(threshold=0.5)
acc.update(out.detach(), mask_t)
gm = acc.compute()
print(f"   Accumulator iou={gm['iou']:.4f}")
print("   [OK] Metrics")

# 6. Checkpoint
print()
print("6. Testing checkpoint save/load ...")
os.makedirs('models', exist_ok=True)
torch.save({'epoch': 0, 'model_state': model.state_dict(),
            'val_dice': 0.0, 'cfg': cfg}, 'models/test_ckpt.pth')
ckpt = torch.load('models/test_ckpt.pth', map_location='cpu')
model2 = build_model(cfg)
model2.load_state_dict(ckpt['model_state'])
print("   [OK] Checkpoint save/load")

print()
print("========================================")
print("ALL PIPELINE CHECKS PASSED SUCCESSFULLY!")
print("========================================")
print(f"Model parameters: {model.count_parameters():,}")
