import time
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import glob
from PIL import Image
from classification import *

class testVerifyDataset(Dataset):
    def __init__(self, test_path, test_vrf_order_path, transforms):
        super().__init__()

        self.test_path = test_path + "/"
        self.transforms = transforms
        self.trial_order_list = np.loadtxt(test_vrf_order_path, dtype=str)
        
    def __len__(self):
        return len(self.trial_order_list)
      
    def __getitem__(self, index):
        imagea = transformations(Image.open(self.test_path + self.trial_order_list[index][0]))
        imageb = transformations(Image.open(self.test_path + self.trial_order_list[index][1]))
        
        return (imagea, imageb)

# In[ ]:


def get_similarity(a, b):
    return torch.sum(a * b, dim=1) / (torch.norm(a, dim=1)*torch.norm(b, dim=1))

def predict_similarity(model_vrf, save=False, filename=""):
    with torch.no_grad():
        similarity_scores = np.array([])

        for batch_idx, (data_a, data_b) in enumerate(tqdm(test_vrf_loader)):
            data_a, data_b = data_a.to(device), data_b.to(device)
            
            out_a = model_vrf(data_a)
            out_b = model_vrf(data_b)

            # average to get embedding
            emb_a = out_a.mean([2, 3])
            emb_b = out_b.mean([2, 3])

            similarities = get_similarity(emb_a, emb_b).cpu().detach().numpy()
            similarity_scores = np.concatenate([similarity_scores, similarities], axis=0)
        
        result = np.concatenate([test_vrf_dataset.trial_order_list, similarity_scores.reshape(-1, 1)], axis=1)
        np.savetxt(pred_vrf_filename, result, fmt="%s %s,%s", header="trial,score", comments="")

    return result



if __name__ == "__main__":
    if verbose:
        print("loading dataset...")

    test_vrf_dataset = testVerifyDataset(test_vrf, test_vrf_order_path, transformations)

    test_vrf_loader = DataLoader(
        test_vrf_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    if verbose:
        print("loading model...")

    model = MobileNetV2(in_shape=input_shape,
                        output_size=num_faceids, dropout=dropout)
    checkpoint_filename = "../checkpoint_13.tar"
    checkpoint = torch.load(checkpoint_filename, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # predict similarity
    if verbose:
        print("predicting verfication...")
    model_vrf = model.feature_extractor
    
    result = predict_similarity(model_vrf, save=True, filename=pred_vrf_filename)

    if verbose:
        print("finished")
