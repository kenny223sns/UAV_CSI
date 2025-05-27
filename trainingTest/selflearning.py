import torch, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
net_s  = LocNet(in_ch=4).to(device)   # student
net_t  = LocNet(in_ch=4).to(device)   # teacher
net_t.load_state_dict(net_s.state_dict())
for p in net_t.parameters(): p.requires_grad = False

opt   = torch.optim.AdamW(net_s.parameters(), lr=1e-4)
scaler= GradScaler()

def update_teacher(alpha=0.995):
    with torch.no_grad():
        for p_t, p_s in zip(net_t.parameters(), net_s.parameters()):
            p_t.data = alpha*p_t.data + (1-alpha)*p_s.data

# ---------- training loop ------------------------------------------------
sup_loader = DataLoader(train_sup,  batch_size=64, shuffle=True, drop_last=True)
all_loader = DataLoader(train_all,  batch_size=64, shuffle=True, drop_last=True)

for epoch in range(10):
    for (x_sup, y_sup), (x_unl, _) in zip(sup_loader, all_loader):
        x_sup, y_sup = x_sup.to(device), y_sup.to(device)
        x_unl = x_unl.to(device)

        # -------- forward -------------
        with autocast():
            y_pred_sup = net_s(x_sup)
            sup_loss = F.mse_loss(y_pred_sup, y_sup)

            # 一致性：teacher 目標、student 輸出
            with torch.no_grad():
                y_t = net_t(x_unl)
            y_s = net_s(x_unl)
            cons_loss = F.mse_loss(y_s, y_t)

            loss = sup_loss + 0.5*cons_loss

        # -------- backward ------------
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad()
        update_teacher()

    print(f"Epoch {epoch:2d} | Sup {sup_loss.item():.3f}  Cons {cons_loss.item():.3f}")
