from collections import defaultdict
import torch

class MahalanobisDetector:
    def __init__(self, model, device="cuda"):
        self.model = model.eval().to(device)
        self.device = device

        # to be filled by fit()
        self.class_means = []    # List[ Dict[label, Tensor] ]
        self.precisions = []     # List[ Tensor ]

    def fit(self, train_loader):
        # get # of layers
        sample_x, _ = next(iter(train_loader))
        sample_x = sample_x.to(self.device)
        with torch.no_grad():
            sample_feats = self.model.extract_features(sample_x)
        n_layers = len(sample_feats)

        # prepare structures
        self.class_means = [dict() for _ in range(n_layers)]
        self.precisions = [None] * n_layers

        # find all classes in the dataset
        all_labels = sorted({ lbl.item() for _, lbl in train_loader })

        # for each layer separately
        for L in range(3, n_layers):
            # we'll build the shared covariance Σ_b layerwise
            Σb = None
            C = 0  # number of classes processed

            # process one class at a time
            for cls in all_labels:
                # First pass: accumulate sum and sum of squares to compute class mean + cov
                sum_feats = None
                sum_outer = None
                N = 0

                # scan the loader once, only collect samples of this class
                for img, lbl in train_loader:
                    if lbl.item() != cls:
                        continue
                    img = img.to(self.device)
                    with torch.no_grad():
                        feat = self.model.extract_features(img)[L]   # Tensor [...,]
                    v = feat.view(-1)                                 # [D]
                    N += 1
                    sum_feats = v if sum_feats is None else sum_feats + v
                    # accumulate outer product for covariance
                    if sum_outer is None:
                        sum_outer = v.unsqueeze(1) @ v.unsqueeze(0)   # [D,D]
                    else:
                        sum_outer = sum_outer + (v.unsqueeze(1) @ v.unsqueeze(0))

                if N < 2:
                    raise ValueError(f"Not enough samples for class {cls} at layer {L}")

                # compute class mean μ_c and covariance Σ_c
                μc = sum_feats / N                                   # [D]
                # Σ_c = (1/(N-1)) * Σ_i (v_i - μ_c)(v_i - μ_c)^T
                # but Σ_i v_i v_i^T = sum_outer. So:
                Σc = (sum_outer - N * (μc.unsqueeze(1) @ μc.unsqueeze(0))) / (N - 1)

                # store this class mean
                self.class_means[L][cls] = μc

                # update global Σb via Algorithm 2:
                # Σb_new = (C/(C+1)) * Σb_old + (1/(C+1)) * Σc
                if Σb is None:
                    Σb = Σc
                else:
                    α = C / float(C + 1)
                    Σb = α * Σb + (1 - α) * Σc

                C += 1

            # done with all classes on this layer → compute and store precision
            self.precisions[L] = torch.inverse(Σb)

    def mahalanobis_score(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            feats = self.model.extract_features(x)
        score = 0.0
        for L, feat in enumerate(feats):
            v = feat.view(-1)
            prec = self.precisions[L]
            # compute min_c (v-μ_c)^T Σ^{-1} (v-μ_c)
            dists = []
            for μc in self.class_means[L].values():
                d = (v - μc).unsqueeze(0)
                dists.append( (d @ prec @ d.t()).item() )
            score += min(dists)
        return score

    def predict(self, x, threshold):
        return self.mahalanobis_score(x) < threshold
