


class FeatureDataset(Dataset):
    """
    Custom dataset to load InceptionV3 features for fine-tuning
    """

    def __init__(self, feature_dir):
        self.files = sorted(glob(f"{feature_dir}/FL*.npy"), key=self._extract_idx)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label, feature = np.load(self.files[idx], allow_pickle=True)
        return torch.from_numpy(feature), torch.from_numpy(label)

    def _extract_idx(self, filename):
        """Extract batch index from filename"""
        match = re.search(r"(\d+)\.npy$", filename)
        match = int(match.group(1)) if match else -1
        if match == -1:
            raise ValueError(f"Invalid filename {filename}")
        return match


class FineTuned(nn.Module):
    """Fine-tuned output layer for InceptionV3"""

    def __init__(self, config):
        super(FineTuned, self).__init__()

        self.clf = nn.Linear(config['embed-dim'], config['num-classes'])

    def forward(self, x):
        x = self.clf(x)
        return x