import csv, os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

class CSIDataset(Dataset):
    def __init__(self, root, csv_path, supervised_only=False,
                 norm=True, aug_flip=False):
        self.root = root
        self.items = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                has_lbl = row['x'] != ''
                if supervised_only and not has_lbl:
                    continue
                path = os.path.join(root, f"sector{row['sector']}", row['file'])
                label = None
                if has_lbl:
                    label = np.array([float(row['x']), float(row['y'])],
                                     dtype=np.float32)
                self.items.append((path, label))
        self.norm = norm
        self.aug_flip = aug_flip
        self.normalize = Normalize(mean=[0], std=[1])  # 可換成真實統計量

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = np.load(path)            # (C,H,W), float32
        if self.aug_flip and np.random.rand()<0.5:
            x = x[..., ::-1]         # delay 軸翻轉 = 順序增強
        x = torch.from_numpy(x)
        if self.norm:
            x = (x - x.mean()) / (x.std() + 1e-6)
        if label is None:
            return x, None
        return x, torch.from_numpy(label)

root = "dataset_three_sector"
train_sup = CSIDataset(root, f"{root}/labels.csv", supervised_only=True)
train_all = CSIDataset(root, f"{root}/labels.csv", supervised_only=False,
                       aug_flip=True)
val_set   = train_sup   # 若你已預留另一份驗證 CSV 再替換
