import torch
import torch.nn as nn
from torch.autograd import Variable
import mahalanobis.lib_generation as lg

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import os
import pickle
import os
import pickle

class MahalanobisDetector(nn.Module):
    def __init__(self, model, num_classes, device="cuda", net_type="resnet", dataset='cifar'):
        """
        model      : a PyTorch model implementing .feature_list() and .intermediate_forward()
        num_classes: number of in‐distribution classes
        device     : "cuda" or "cpu"
        net_type   : "resnet" or "densenet" (for gradient normalization)
        """
        super(MahalanobisDetector, self).__init__()
        self.device      = device
        self.model       = model.to(device)
        self.num_classes = num_classes
        self.net_type    = net_type
        self.dataset_mean = dataset.mean
        self.dataset_std = dataset.std
        
        # figure out how many feature‐layers and their dims
        self.model.eval()
        with torch.no_grad():
            if dataset.ds_name == 'mnist':
                dummy = torch.randn(1,1,28,28, device=device)
            elif dataset.ds_name == 'imagenet':
                dummy = torch.randn(1,3,224,224, device=device)
            else: # cifar, svhn, etc.
                # assume 32x32 input for CIFAR-like datasets
                dummy = torch.randn(1,3,32,32, device=device)
            _, feats = self.model.feature_list(dummy)
        self.layer_dims = [f.size(1) for f in feats]
        
        # place-holders for after fit()
        self.sample_mean = None
        self.precision   = None

    def save(self, path):
        """
        Saves all the variables needed to reconstruct the detector model at the given path.
        This includes the model architecture, sample means, and precision matrices.
        The model architecture is saved in a way that it can be reloaded later.

        Args:
            path (str): The directory where the model will be saved.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save the main model
        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        
        # Save sample means and precision matrices if they exist
        if self.sample_mean is not None:
            torch.save(self.sample_mean, os.path.join(path, 'sample_mean.pth'))
        if self.precision is not None:
            torch.save(self.precision, os.path.join(path, 'precision.pth'))
            
        # Save regressor if it exists
        if hasattr(self, 'regressor') and self.regressor is not None:
            torch.save(self.regressor.state_dict(), os.path.join(path, 'regressor.pth'))
        
        # Save metadata
        metadata = {
            'num_classes': self.num_classes,
            'net_type': self.net_type,
            'layer_dims': self.layer_dims
        }
        torch.save(metadata, os.path.join(path, 'metadata.pth'))
        
        print(f"Models saved to {path}")

    def load(self, path):
        """
        Loads all the variables needed to reconstruct the detector model from the given path.
        This includes the model architecture, sample means, and precision matrices.
        The model architecture is loaded in a way that it can be used immediately.
        It assumes that the model architecture is compatible with the saved parameters.

        Args:
            path (str): The directory from where the model will be loaded.
        """
        
        # Load the main model
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=self.device))
        
        # Load metadata
        metadata = torch.load(os.path.join(path, 'metadata.pth'), map_location=self.device)
        self.num_classes = metadata['num_classes']
        self.net_type = metadata['net_type']
        self.layer_dims = metadata['layer_dims']
        
        # Load sample means and precision matrices if they exist
        sample_mean_path = os.path.join(path, 'sample_mean.pth')
        if os.path.exists(sample_mean_path):
            self.sample_mean = torch.load(sample_mean_path, map_location=self.device)
        
        precision_path = os.path.join(path, 'precision.pth')
        if os.path.exists(precision_path):
            self.precision = torch.load(precision_path, map_location=self.device)
            
        # Load regressor if it exists
        regressor_path = os.path.join(path, 'regressor.pth')
        if os.path.exists(regressor_path):
            self.regressor = nn.Linear(len(self.layer_dims), 1, bias=True)
            self.regressor.load_state_dict(torch.load(regressor_path, map_location=self.device))
            self.regressor.to(self.device)
        else:
            self.regressor = None


        print(f"Models loaded from {path}")

    def fit(self, train_loader):
        """
        Compute and store (sample_mean, precision) for each feature‐layer.
        train_loader should yield (input, target) on in‐distribution data.
        """
        self.model.eval()
        self.sample_mean, self.precision = lg.sample_estimator(
            self.model,
            self.num_classes,
            self.layer_dims,
            train_loader
        )

    def mahalanobis_score(self, x, magnitude=0.001):
        """
        x:    a Tensor of shape (B,3,H,W) on the same device as the model
        magnitude: the noise magnitude for input‐processing
        
        Returns:
           scores: a Tensor of shape (B, num_layers), where each entry is
                   the Mahalanobis score for that sample & layer.
        """
        assert self.sample_mean is not None, "You must call .fit(...) first"
        self.model.eval()

        std = torch.tensor(self.dataset_std, device=self.device)
        
        x = x.to(self.device)
        if x.dim()==3:
            x = x.unsqueeze(0)
        x.requires_grad_(True)
        x_var = x
        
        all_scores = []
        for layer_idx in range(len(self.layer_dims)):
            # 1) forward to get features at this layer
            out = self.model.intermediate_forward(x_var, layer_idx)
            out = out.view(out.size(0), out.size(1), -1).mean(dim=2)
            
            # 2) compute Gaussian scores for each class
            #    G_ic = -½ (f - μ_ic)^T Σ^{-1} (f - μ_ic)
            gaussian_score = []
            for c in range(self.num_classes):
                zero_f = out - self.sample_mean[layer_idx][c].unsqueeze(0)
                term = -0.5 * torch.mm(torch.mm(zero_f, self.precision[layer_idx]), zero_f.t()).diag()
                gaussian_score.append(term.unsqueeze(1))
            gaussian_score = torch.cat(gaussian_score, dim=1)  # (B, num_classes)
            
            # 3) pick the class with the highest raw score
            sample_pred = gaussian_score.max(dim=1)[1]
            batch_mean  = self.sample_mean[layer_idx].index_select(0, sample_pred)
            
            # 4) recompute “pure” Gaussian score & backprop through it
            zero_f = out - batch_mean
            pure_gau = -0.5 * torch.mm(torch.mm(zero_f, self.precision[layer_idx]), zero_f.t()).diag()
            loss = torch.mean(-pure_gau)

            # compute gradient of `loss` w.r.t. x_var, keeping it in the graph
            grad_x = torch.autograd.grad(loss, x_var,
                                            create_graph=True,
                                            retain_graph=True)[0]
            # now build your “±1” mask in a way that’s still differentiable
            grad = (grad_x.ge(0).float() * 2 - 1) / std.view(1, -1, 1, 1)
            
            # 6) add/subtract the tiny noise
            perturbed = x_var - magnitude * grad
            perturbed.requires_grad_(True)
            
            # 7) re‐extract features & repeat scoring
            noise_feat = self.model.intermediate_forward(perturbed, layer_idx)
            noise_feat = noise_feat.view(noise_feat.size(0), noise_feat.size(1), -1).mean(dim=2)
            noise_gauss = []
            for c in range(self.num_classes):
                zero_f = noise_feat - self.sample_mean[layer_idx][c].unsqueeze(0)
                term = -0.5 * torch.mm(torch.mm(zero_f, self.precision[layer_idx]), zero_f.t()).diag()
                noise_gauss.append(term)
            noise_gauss = torch.stack(noise_gauss, dim=1)
            
            # final Mahalanobis score is the max across classes
            score, _ = noise_gauss.max(dim=1)   # (B,)
            all_scores.append(score)
            
            # zero‐out grads for next layer
            # x_var.grad.data.zero_()
        
        # (num_layers, B) → (B, num_layers)
        return torch.stack(all_scores, dim=1)
    
    def train_regressor(self, val_loader, norm, attacker):
        """
        Build X, y from val_loader by running mahalanobis_score on each batch,
        then fit a cross-validated LogisticRegressionCV.
        """
        i=0
        self.model.eval()
        feats_list, lbls_list = [], []
        for image, label in val_loader:
            # Add Natural image features
            feats = self.mahalanobis_score(norm(image.to(self.device))).detach()
            feats_list.append(feats.cpu().numpy())
            lbls_list.append(np.float32(0))

        # Add adversarial image features
            adv_image = attacker.attack(image.to(self.device), label.to(self.device))
            feats = self.mahalanobis_score(norm(adv_image)).detach()
            feats_list.append(feats.cpu().numpy())
            lbls_list.append(np.float32(1))

            # i += 1
            # if i >= 100:  # limit to 1000 batches
            #     break

        X = np.vstack(feats_list)   # (N, L)
        y = np.array(lbls_list)  # (N,)

        # same as ADV_Regression.py :contentReference[oaicite:0]{index=0}
        regressor = LogisticRegressionCV(n_jobs=-1).fit(X, y)

        w  = regressor.coef_      # shape (n_classes, n_features)
        b  = regressor.intercept_ # shape (n_classes,)
        n_out , n_in = w.shape            # w is 2-D even for binary
        self.regressor = nn.Linear(n_in, n_out, bias=True)
        with torch.no_grad():
            self.regressor.weight.copy_(torch.from_numpy(w).float())
            self.regressor.bias.copy_(torch.from_numpy(b).float())
        
        self.regressor.to(self.device)
        self.regressor.eval()
        print("Regressor trained and ready for use.")

    def forward(self, x, label=None):
        """
        Given raw inputs x (Tensor or batch), compute the final
        detection score (probability of being *positive*) via:

            1) mahalanobis_score(x) → shape (B, L)
            2) regressor.predict_proba(...)[:,1] → shape (B,)

        Returns a numpy array of length B.
        """
        if self.regressor is None:
            raise RuntimeError("You must call train_regressor(...) first")

        # 1) compute raw features
        feats = self.mahalanobis_score(x)  # (B, num_layers) torch.Tensor

        logits = self.regressor(feats)

        if logits.shape[1] == 1:                   # binary
            proba = torch.sigmoid(logits).squeeze(1)
        else:                                      # multi-class
            proba = torch.softmax(logits, dim=1)[:, 1]  # “positive” = class-1
        return proba

# First iteration of the LIDDetector class, not working yet
# It is a work in progress and does NOT function as intended.
class LIDDetector(nn.Module):
    def __init__(self, model, device="cuda", k=20, input_shape=(3,32,32)):
        """
        model       : a PyTorch model implementing .feature_list()
        device      : "cuda" or "cpu"
        k           : number of neighbors for LID estimation
        input_shape : tuple (C,H,W) for creating a dummy input
        """
        super(LIDDetector, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.k = k

        # figure out how many feature‐layers we have
        self.model.eval()
        with torch.no_grad():
            C,H,W = input_shape
            dummy = torch.randn(1, C, H, W, device=device)
            _, feats = self.model.feature_list(dummy)
        self.num_layers = len(feats)

        # will hold a sklearn NearestNeighbors for each layer
        self.neigh = [None] * self.num_layers

    def fit(self, train_loader, k=None):
        """
        Build a k-NN index on training features at each layer.
        train_loader should yield (inputs, _) on in‐distribution data.
        """
        if k is not None:
            self.k = k

        self.model.eval()
        # collect features per layer
        feats_acc = {i: [] for i in range(self.num_layers)}
        for x, _ in train_loader:
            x = x.to(self.device)
            with torch.no_grad():
                _, feats = self.model.feature_list(x)
            for i, f in enumerate(feats):
                # global‐average‐pool spatial dims
                f = f.view(f.size(0), f.size(1), -1).mean(dim=2)
                feats_acc[i].append(f.cpu().numpy())

        # fit a NearestNeighbors model per layer
        for i in range(self.num_layers):
            data = np.vstack(feats_acc[i])  # shape (N, D_i)
            nbrs = NearestNeighbors(n_neighbors=self.k+1, algorithm='auto')
            nbrs.fit(data)
            self.neigh[i] = nbrs

        print(f"LIDDetector: fitted {self.num_layers} layer(s) with k={self.k}")

    def forward(self, x):
        """
        Compute per-layer LID scores for input batch x.
        Returns a tensor of shape (B, num_layers) on self.device.
        """
        assert all(m is not None for m in self.neigh), "You must call .fit(...) first"
        self.model.eval()

        x = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # 1) extract features at all layers
        with torch.no_grad():
            _, feats = self.model.feature_list(x)

        lid_scores = []
        for i, f in enumerate(feats):
            # 2) flatten and global‐average‐pool
            f_flat = f.view(f.size(0), f.size(1), -1).mean(dim=2)
            f_np = f_flat.cpu().numpy()  # (B, D_i)

            # 3) query k+1 neighbors (first neighbor is itself at distance 0)
            dists, _ = self.neigh[i].kneighbors(f_np)  # => (B, k+1)
            dists = dists[:, 1:]                       # drop self‐match => (B, k)

            # 4) MLE estimator: ˆLID = -k / sum_j log(r_j / r_k)
            r_k = dists[:, -1].reshape(-1, 1)           # (B,1)
            logs = np.log(dists / r_k)                 # (B,k)
            lids = - (self.k / np.sum(logs, axis=1))   # (B,)

            lid_scores.append(torch.from_numpy(lids).to(self.device))

        # (num_layers, B) → (B, num_layers)
        return torch.stack(lid_scores, dim=1)
