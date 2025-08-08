# sv_agreement_rollout.py  (with early stopping)
import os, gzip, csv, math, json, argparse, random, pathlib, copy
from collections import Counter
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# ---------------------- Data ----------------------
KEEP_POS = {'VBZ': 0, 'VBP': 1}  # 0=singular, 1=plural

def load_tsv_gz(path):
    with gzip.open(path, 'rt', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = [r for r in reader]
    return rows

def make_examples(rows, max_len=40):
    ex = []
    drop_non_vbz_vbp = 0
    misaligned = 0
    for r in rows:
        pos = r['verb_pos'].strip()
        if pos not in KEEP_POS:
            drop_non_vbz_vbp += 1
            continue
        try:
            v_idx = int(r['verb_index'])
        except:
            continue
        toks = r['orig_sentence'].strip().split()
        prefix_len = max(0, v_idx - 1)
        if prefix_len < 1:
            misaligned += 1
            continue
        toks = toks[:min(prefix_len, max_len-2)]
        if len(toks) == 0:
            misaligned += 1
            continue
        ex.append((toks, KEEP_POS[pos]))
    print(f"Loaded {len(ex)} examples "
          f"(dropped {drop_non_vbz_vbp} non-(VBZ/VBP), {misaligned} short/misaligned).")
    return ex

class Vocab:
    def __init__(self, counter, min_freq=1, specials=("[PAD]", "[UNK]", "[CLS]")):
        self.itos = list(specials)
        for tok, c in counter.most_common():
            if c >= min_freq and tok not in specials:
                self.itos.append(tok)
        self.stoi = {t:i for i,t in enumerate(self.itos)}
        self.pad_id, self.unk_id, self.cls_id = [self.stoi[s] for s in specials]
    def encode(self, toks):
        return [self.stoi.get(t, self.unk_id) for t in toks]

class SVADataset(Dataset):
    def __init__(self, examples, vocab, max_len=40):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        toks, y = self.examples[idx]
        ids = [self.vocab.cls_id] + self.vocab.encode(toks)
        ids = ids[:self.max_len]
        attn = [1]*len(ids)
        pad = self.max_len - len(ids)
        if pad > 0:
            ids = ids + [self.vocab.pad_id]*pad
            attn = attn + [0]*pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ---------------------- Model ----------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, p=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
        self.n2 = nn.LayerNorm(d_model)
        hidden = int(d_model*mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, d_model), nn.Dropout(p)
        )
    def forward(self, x, key_padding_mask=None, need_attn=False):
        q = k = v = self.n1(x)
        y, w = self.attn(q, k, v, key_padding_mask=key_padding_mask,
                         need_weights=need_attn, average_attn_weights=not need_attn)
        x = x + self.drop(y)
        x = x + self.mlp(self.n2(x))
        return (x, w) if need_attn else x

class TextTransformer(nn.Module):
    """
    mode='baseline' -> use CLS token representation
    mode='rollout'  -> use attention-rollout weighted sum over tokens
    """
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=4, p=0.1, mode='baseline', num_classes=2, max_len=64):
        super().__init__()
        self.mode = mode
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, p=p) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    @staticmethod
    def rollout(attn_list, pad_mask=None):
        B, H, S, _ = attn_list[0].shape
        I = torch.eye(S, device=attn_list[0].device).expand(B, S, S)
        P = I
        for A in attn_list:
            A = A.mean(1)
            A = A + I
            A = A / A.sum(-1, keepdim=True)
            if pad_mask is not None:
                mask = (~pad_mask).float()  # (B, S)
                A = A * mask.unsqueeze(1)
                A = A * mask.unsqueeze(2)
                denom = A.sum(-1, keepdim=True).clamp_min(1e-6)
                A = A / denom
            P = torch.bmm(A, P)
        return P  # (B, S, S)

    def forward_features(self, ids, attn_mask):
        x = self.tok(ids) + self.pos[:, :ids.size(1), :]
        x = self.drop(x)
        all_attn = []
        key_padding_mask = (attn_mask == 0)
        for blk in self.blocks:
            if self.mode == 'rollout':
                x, w = blk(x, key_padding_mask=key_padding_mask, need_attn=True)
                all_attn.append(w.detach())
            else:
                x = blk(x, key_padding_mask=key_padding_mask, need_attn=False)
        x = self.norm(x)
        if self.mode == 'baseline':
            return x[:, 0]
        P = self.rollout(all_attn, pad_mask=(attn_mask==1))
        W = P[:, 0, 1:]
        token_mask = attn_mask[:, 1:].float()
        W = W * token_mask
        W = W / (W.sum(-1, keepdim=True).clamp_min(1e-6))
        patches = x[:, 1:, :]
        rep = (W.unsqueeze(-1) * patches).sum(1)
        return rep

    def forward(self, ids, attn_mask):
        rep = self.forward_features(ids, attn_mask)
        return self.head(rep)

# ---------------------- Train / Eval ----------------------
def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@torch.no_grad()
def eval_loss_acc(model, loader, device, crit):
    model.eval()
    tot_loss = 0.0
    n = 0
    correct = 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        logits = model(ids, mask)
        loss = crit(logits, y)
        tot_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.numel()
    return tot_loss / max(1, len(loader)), correct / max(1, n)

def train_one(model, train_loader, epochs, device, lr_max=5e-4, lr_min=5e-6, warm=5, wd=1e-4,
              val_loader=None, early_stop=True, monitor='val_acc', mode='max', patience=10, min_delta=0.0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=wd)
    def lr_lambda(ep):
        if ep < warm: return (ep+1)/warm
        t = (ep - warm) / max(1, (epochs - warm))
        return (lr_min/lr_max) + (1 - lr_min/lr_max) * 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_metric = None
    best_state = None
    wait = 0
    comp = (lambda a,b: a > b + min_delta) if mode == 'max' else (lambda a,b: a < b - min_delta)

    model.train()
    train_losses=[]
    for ep in range(epochs):
        tot=0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                logits = model(ids, mask)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
        sched.step()
        ep_loss = tot/len(train_loader)
        train_losses.append(ep_loss)

        # validation / early stopping
        if val_loader is not None:
            val_loss, val_acc = eval_loss_acc(model, val_loader, device, crit)
            metric = val_acc if monitor == 'val_acc' else val_loss
            if best_metric is None or comp(metric, best_metric):
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
            if (ep+1) % 10 == 0 or ep < 5:
                print(f"Ep {ep+1:03d}  trn_loss={ep_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={opt.param_groups[0]['lr']:.1e}")
            if early_stop and wait >= patience:
                print(f"Early stopping at epoch {ep+1} (best {monitor}={best_metric:.4f}).")
                break
        else:
            if (ep+1) % 10 == 0 or ep < 5:
                print(f"Ep {ep+1:03d}  loss={ep_loss:.4f}  lr={opt.param_groups[0]['lr']:.1e}")

    # load best weights if we validated
    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
    return train_losses

@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    n=0; c=0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        pred = model(ids, mask).argmax(1)
        c += (pred==y).sum().item()
        n += y.numel()
    return c/n

def build_loaders(path, max_len=40, batch=128, seed=42):
    rows = load_tsv_gz(path)
    ex = make_examples(rows, max_len=max_len)
    set_seed(seed)
    random.shuffle(ex)
    n = len(ex)
    n_train = int(0.8*n); n_val = int(0.1*n); n_test = n - n_train - n_val
    train_ex, val_ex, test_ex = ex[:n_train], ex[n_train:n_train+n_val], ex[n_train+n_val:]

    counter = Counter(t for toks,_ in train_ex for t in toks)
    vocab = Vocab(counter, min_freq=1)
    ds_train = SVADataset(train_ex, vocab, max_len=max_len)
    ds_val   = SVADataset(val_ex,   vocab, max_len=max_len)
    ds_test  = SVADataset(test_ex,  vocab, max_len=max_len)

    train_dl = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(ds_val,   batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(ds_test,  batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return vocab, train_dl, val_dl, test_dl

def plot_losses(loss_a, loss_b, labels=('baseline','rollout'), out='sva_loss_curve.png'):
    plt.figure(figsize=(7,4))
    plt.plot(loss_a, label=labels[0])
    plt.plot(loss_b, label=labels[1])
    plt.xlabel('epoch'); plt.ylabel('train CE loss'); plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

# ---------------------- Sweep (optional) ----------------------
def quick_sweep(vocab_size, train_dl, val_dl, device, max_len, mode, epochs=30):
    cfgs = {
        'lr_max': [5e-4, 1e-3],
        'wd':     [1e-4, 5e-5],
        'drop':   [0.1, 0.2],
        'layers': [4, 6],
        'heads':  [4],
        'd_model':[256],
    }
    best = None
    tried = 0
    for lr_max, wd, drop, layers in product(cfgs['lr_max'], cfgs['wd'], cfgs['drop'], cfgs['layers']):
        tried += 1
        print(f"[{tried}] mode={mode} lr={lr_max} wd={wd} drop={drop} L={layers}")
        model = TextTransformer(vocab_size, d_model=cfgs['d_model'][0],
                                n_layers=layers, n_heads=cfgs['heads'][0],
                                p=drop, mode=mode, num_classes=2, max_len=max_len).to(device)
        _ = train_one(model, train_dl, epochs, device, lr_max=lr_max, wd=wd,
                      val_loader=val_dl, early_stop=True, monitor='val_acc', mode='max', patience=5)
        acc = accuracy(model, val_dl, device)
        print(f"val acc={acc:.4f}")
        cand = (acc, {'lr_max':lr_max,'wd':wd,'drop':drop,'layers':layers})
        if best is None or acc > best[0]:
            best = cand
    print(f"Best ({mode}): acc={best[0]:.4f} cfg={best[1]}")
    return best

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='agr_50_mostcommon_10K.tsv.gz')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--max_len', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sweep', action='store_true', help='run quick hyperparameter sweep (30 epochs)', default=True)
    args = ap.parse_args([])  # remove [] if running as script from CLI

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using Device: {device}")

    vocab, train_dl, val_dl, test_dl = build_loaders(args.data, max_len=args.max_len, batch=args.batch, seed=args.seed)
    common = dict(vocab_size=len(vocab.itos), d_model=256, n_layers=6, n_heads=4, p=0.1, num_classes=2, max_len=args.max_len)

    if args.sweep:
        best_base = quick_sweep(len(vocab.itos), train_dl, val_dl, device, args.max_len, mode='baseline', epochs=30)
        best_roll = quick_sweep(len(vocab.itos), train_dl, val_dl, device, args.max_len, mode='rollout',  epochs=30)
        with open('sweep_results.json','w') as f:
            json.dump({
                'baseline': {'val_acc': best_base[0], 'cfg': best_base[1]},
                'rollout':  {'val_acc': best_roll[0], 'cfg': best_roll[1]}
            }, f, indent=2)
        print("Saved sweep_results.json")

    # Train final models (you can paste best cfgs here after sweep if you want)
    base = TextTransformer(**common, mode='baseline').to(device)
    roll = TextTransformer(**common, mode='rollout').to(device)

    print("Training baseline")
    base_losses = train_one(base, train_dl, args.epochs, device,
                            val_loader=val_dl, early_stop=True, monitor='val_acc', mode='max', patience=10)
    print("Training rollout")
    roll_losses = train_one(roll, train_dl, args.epochs, device,
                            val_loader=val_dl, early_stop=True, monitor='val_acc', mode='max', patience=10)

    plot_losses(base_losses, roll_losses, out='sva_loss_curve.png')

    base_acc = accuracy(base, test_dl, device)
    roll_acc = accuracy(roll, test_dl, device)
    print(f"Test accuracy  baseline={base_acc:.4f}  rollout={roll_acc:.4f}")

    torch.save(base.state_dict(), 'baseline_sva.pt')
    torch.save(roll.state_dict(), 'rollout_sva.pt')
    with open('results_sva.json','w') as f:
        json.dump({'baseline': {'test_acc': float(base_acc)},
                   'rollout':  {'test_acc': float(roll_acc)}}, f, indent=2)
    print("Saved weights and results.")

if __name__ == '__main__':
    main()
