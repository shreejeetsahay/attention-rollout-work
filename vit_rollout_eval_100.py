import torch, torchvision, argparse, json, math, pathlib
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from vit_rollout import *

def main():
    train_dl, test_dl, test_ds = get_loaders()             
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    baseline = ViT(mode='baseline').to(device)
    rollout  = ViT(mode='rollout').to(device)
    baseline.load_state_dict(torch.load('baseline.pth', map_location=device))
    rollout.load_state_dict(torch.load('rollout.pth',  map_location=device))

    classes = test_ds.classes            
    per_cls = 100                        

    metrics = {}
    for name, model in [('baseline', baseline), ('rollout', rollout)]:
        feats, labels = collect_feats(model, test_ds, classes, per_cls, device)
        feats = F.normalize(feats, 2, 1)                
        intra, inter, ratio = centroid_stats(feats, labels)
        acc = KNeighborsClassifier(5, metric='cosine').fit(feats, labels).score(feats, labels)
        sil = silhouette_score(feats.numpy(), labels.numpy(), metric='euclidean')
        metrics[name] = {k: float(v) for k, v in
                         dict(intra=intra, inter=inter, ratio=ratio,
                              knn5=acc, silhouette=sil).items()}
        print(f"{name:8s}  knn5={acc:.3f}  intra={intra:.4f}  "
              f"inter={inter:.4f}  ratio={ratio:.2f}  sil={sil:.3f}")

    with open('results_100.json', 'w') as fp:
        json.dump(metrics, fp, indent=2)

if __name__ == '__main__':
    main()
