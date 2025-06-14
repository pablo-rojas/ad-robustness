import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_utils import initialize_us_models, resnet18_classifier, get_patch_descriptor

class UninformedStudents(nn.Module):
    """
    A detector model that trains multiple student models to detect adversarial examples.

    Attributes:
        patch_size (int): The size of the patches.
        dataset (str): The name of the dataset.
        num_students (int): The number of student models.
        teacher (torch.nn.Module): The teacher model.
        students (list): The student models.
        criterion (torch.nn.Module): The loss criterion.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the models on.
    """

    def __init__(self, config, device='cpu'):
        """
        Initializes the UninformedStudents model.

        Args:
            config (dict): Configuration dictionary.
            device (torch.device, optional): The device to run the models on. Defaults to 'cpu'.
            norm (bool, optional): Whether to normalize the teacher outputs. Defaults to False.
        """
        super(UninformedStudents, self).__init__()
        self.patch_size     = config["patch_size"]
        self.dataset        = config["dataset"]
        self.num_students   = config["num_students"]

        # Initialize the teacher and student models.
        # self.teacher = get_patch_descriptor(self.patch_size, dim=3)
        # self.teacher.load_state_dict(torch.load('./models/p17descriptor.pth'))
        self.teacher, self.students = initialize_us_models(
            self.num_students, self.dataset, self.patch_size, device
            )

        # by default whitening is inactive: mean=0, std=1 (scalars)
        self.register_buffer('teacher_mean', torch.tensor(0., device=device))
        self.register_buffer('teacher_std',  torch.tensor(1., device=device))

        if "train" in config:
            self.optimizer = torch.optim.Adam(
                [p for student in self.students for p in student.parameters()],
                lr=config["train"]["learning_rate"],
                weight_decay=config["train"]["weight_decay"],
                betas=(config["train"]["beta1"], config["train"]["beta2"]),
                eps=config["train"]["eps"]
            )
            self.phard          = config["train"].get("phard", 0.0)
            self.batch_size     = config["train"]["batch_size"]
            self.subdivisions   = config["train"]["batch_subdivisions"]
        
        self.to(device)
    
    def save(self, path):
        """
        Saves the teacher and student models to the specified path.

        Args:
            path (str): The directory where the models will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.teacher.state_dict(), os.path.join(path, 'teacher.pth'))
        for idx, student in enumerate(self.students):
            torch.save(student.state_dict(), os.path.join(path, f'student_{idx}.pth'))
        
        # Save extra variables if they exist.
        extra_vars = {}
        for name in ('e_mean','e_std','v_mean','v_std'):
            if hasattr(self, name):
                extra_vars[name] = getattr(self, name)
        extra_vars['teacher_mean'] = self.teacher_mean.cpu()
        extra_vars['teacher_std']  = self.teacher_std.cpu()

        torch.save(extra_vars, os.path.join(path, 'extra_vars.pth'))

        print(f"Models saved to {path}")

    def load(self, path):
        """
        Loads the teacher and student models from the specified path.

        Args:
            path (str): The directory from where the models will be loaded.
        """
        self.teacher.load_state_dict(torch.load(os.path.join(path, 'teacher.pth')))
        for idx, student in enumerate(self.students):
            student.load_state_dict(torch.load(os.path.join(path, f'student_{idx}.pth')))
        
        # Load extra variables if the file exists.
        extra_path = os.path.join(path, 'extra_vars.pth')
        if os.path.exists(extra_path):
            extra_vars = torch.load(extra_path,
                                    map_location=self.teacher_mean.device)
            # restore teacher_mean & teacher_std via .data.copy_
            if 'teacher_mean' in extra_vars:
                self.teacher_mean.data.copy_(extra_vars['teacher_mean'])
            if 'teacher_std' in extra_vars:
                self.teacher_std.data.copy_(extra_vars['teacher_std'])
            for key, value in extra_vars.items():
                setattr(self, key, value)

        print(f"Models loaded from {path}")

    def save_pretrain(self, filename):
        """
        Saves the first student model as the pre-trained patch descriptor.
        Args:
            path (str): The directory where the model will be saved.
        """
        torch.save(self.students[0].state_dict(), filename)
        print(f"Pre-trained model saved to {filename}")

    def to(self, device):
        """
        Moves the teacher and student models to the specified device.

        Args:
            device (torch.device): The device to move the models to.
        """
        self.device = device
        self.teacher.to(device)
        for student in self.students:
            student.to(device)

    def train_step(self, x, labels=None):
        """
        Performs a training step on a batch of anomaly-free images.
        Extracts patches and trains the student models.

        Args:
            x (torch.Tensor): The input images.
            labels (torch.Tensor, optional): The class labels, unused, for compatibility reasons. Defaults to None.

        Returns:
            torch.Tensor: The total loss.
        """

        B = x.size(0)
        S = self.subdivisions
        assert B >= S, "Batch size must be >= subdivisions"
        
        # zero grads once
        self.optimizer.zero_grad()

        total_loss = 0.0
        # size of each chunk (last chunk may be larger if B % S != 0)
        for i in range(S):
            start = i * (B // S)
            end   = (i + 1) * (B // S) if i < S - 1 else B
            x_sub = x[start:end]

            t = self.teacher(x_sub).detach()

            # Normalize the teacher outputs if needed
            t_norm = (t - self.teacher_mean[None, ..., None, None]) \
                        / self.teacher_std[ None, ..., None, None]

            # Preallocate tensor for student outputs.
            s = torch.empty((len(self.students),) + t.shape, 
                                            device=x.device, dtype=t.dtype)
            
            # Fill the preallocated tensor with each student's output
            for i, student in enumerate(self.students):
                s[i] = student(x_sub).squeeze()

            t_exp = t_norm.unsqueeze(0).expand_as(s)

            if self.phard > 0:
                # Hard feature mining
                # D = (s - t)^2
                D = (s - t_exp).pow(2)                              # (M, B, C, W, H)
                M, B2, C, W, H = D.shape
                # flatten per sample
                D_flat = D.view(M*B2, C*W*H)                         # (M*B, C*W*H)
                N = D_flat.size(1)                                  # N = C*W*H
                # compute the ϕ-quantile rank (1-indexed):
                k = int(self.phard * N)
                k = min(max(k, 1), N)
                # kthvalue returns the k-th smallest element
                thresh, _ = D_flat.kthvalue(k, dim=1, keepdim=True)
                mask = D_flat >= thresh                             # (M*B, C*W*H)
                # count hard examples, avoid div0
                counts = mask.sum(dim=1).clamp(min=1).float()       # (M*B,)
                # sum only hard diffs
                loss_per = (D_flat * mask).sum(dim=1) / counts      # (M*B,)
                loss = loss_per.mean()
            else:
                # Plain MSE
                loss = F.mse_loss(s, t_exp)

            # scale the gradient contribution
            loss = loss / S
            loss.backward()
            total_loss += loss.item()

        self.optimizer.step()

        return total_loss

    def train(self, mode=True):
        """
        Sets the training mode for all student models.

        Args:
            mode (bool, optional): The training mode. Defaults to True.
        """
        for student in self.students:
            student.train(mode)

    def eval(self):
        """
        Sets the evaluation mode for all student models.
        """
        for student in self.students:
            student.eval()

    def remove_decoder(self):
        """
        Removes the decoder layer from the student and teacher models.
        """
        for student in self.students:
            student.decode = torch.nn.Identity()
        self.teacher.decode = torch.nn.Identity()  

    def forward(self, x, label=None, return_as=True):
        """
        Forward pass of the detector model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            torch.Tensor, torch.Tensor: The regression error and predictive uncertainty.
        """
        t = self.teacher(x).detach()

        # Normalize the teacher outputs if needed
        t_norm = (t - self.teacher_mean[None, ..., None, None]) \
                 / self.teacher_std[ None, ..., None, None]
        t_norm = t_norm.squeeze()  # remove any singleton dims

        t_exp = t_norm.expand(len(self.students), *t_norm.shape)
        
        s = torch.stack([student(x).squeeze() for student in self.students])
        
        students_mean = s.mean(dim=0)
        
        squared_diffs = (students_mean - t_exp) ** 2
        regression_error = squared_diffs.mean()
        
        predictive_uncertainty = s.var(dim=0).mean()
        
        if return_as:
            return (regression_error - self.e_mean) / self.e_std + (predictive_uncertainty - self.v_mean) / self.v_std
        else:
            return regression_error, predictive_uncertainty

class PetrainedDescriptor(UninformedStudents):
    """
    A detector model that uses a pre-trained descriptor for anomaly detection.

    Attributes:
        patch_size (int): The size of the patches.
        dataset (str): The name of the dataset.
        num_students (int): The number of student models.
        teacher (torch.nn.Module): The teacher model.
        students (list): The student models.
        criterion (torch.nn.Module): The loss criterion.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the models on.
    """

    def __init__(self, config, device='cpu'):
        super(PetrainedDescriptor, self).__init__(config, device)

    def save_pretrain(self, filename):
        """
        Saves the first student model as the pre-trained patch descriptor.
        Args:
            path (str): The directory where the model will be saved.
        """
        torch.save(self.students[0].state_dict(), filename)
        print(f"Pre-trained model saved to {filename}")



def init_model_cifar(device, dataset='cifar'):
    teacher = resnet18_classifier(device, dataset=dataset, pretrained=True)
    student = resnet18_classifier(device, dataset=dataset, pretrained=False)

    return teacher, student

class STFPM(nn.Module):
        
    def __init__(self, dataset, device='cpu', lr=0.0001):
        super(STFPM, self).__init__()
        self.teacher, self.student = init_model_cifar(device, dataset=dataset.ds_name)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [param  for param in self.student.parameters()], lr=lr
        )
        self.to(device)
        print("Teacher: " + str(self.teacher))
        print("Student: " + str(self.student))

    def save(self, path):
        """
        Saves the teacher, teacher feature extractor, and student models to the specified path.

        Args:
            path (str): The directory where the models will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.teacher.state_dict(), os.path.join(path, 'teacher.pth'))
        torch.save(self.student.state_dict(), os.path.join(path, f'student.pth'))

        print(f"Models saved to {path}")

    def load(self, path):
        """
        Loads the teacher, teacher feature extractor, and student models from the specified path.

        Args:
            path (str): The directory from where the models will be loaded.
        """
        self.teacher.load_state_dict(torch.load(os.path.join(path, 'teacher.pth')))
        self.student.load_state_dict(torch.load(os.path.join(path, f'student.pth')))

        print(f"Models loaded from {path}")

    def to(self, device):
        """
        Moves the teacher and student models to the specified device.

        Args:
            device (torch.device): The device to move the models to.
        """
        self.device = device
        self.teacher.to(device)
        self.student.to(device)

    def train(self, mode=True):
        """
        Sets the training mode for all student models.

        Args:
            mode (bool, optional): The training mode. Defaults to True.
        """
        self.student.train(mode)

    def eval(self):
        """
        Sets the evaluation mode for all student models.
        """
        self.student.eval()

    def extract_features(self, net, x, dataset='cifar'):
        """
        Extracts the pyramid features from the network.
        For ResNet-18, we use:
          - layer1 (conv2_x)
          - layer2 (conv3_x)
          - layer3 (conv4_x)
        """
        x = net.conv1(x)
        x = net.bn1(x)
        if dataset == 'cifar':
            x = F.relu(x)
        else:
            x = self.relu(x)
            x = net.maxpool(x)
        f1 = net.layer1(x)
        f2 = net.layer2(f1)
        f3 = net.layer3(f2)
        return [f1, f2, f3]

    def train_step(self, x, labels=None):
        """
        Performs a training step on a batch of anomaly-free images.
        For each image, features are extracted from the teacher and student at multiple scales.
        The loss is the sum (over scales) of the per-pixel MSE between L2-normalized features.
        """
        self.teacher.eval()  # teacher is fixed
        self.student.train()

        # Extract features and detach teacher features to avoid unnecessary gradient computation.
        teacher_feats = self.extract_features(self.teacher, x)
        teacher_feats = [feat.detach() for feat in teacher_feats]
        student_feats = self.extract_features(self.student, x)

        loss_total = 0
        # For each scale in the feature pyramid
        for ft, fs in zip(teacher_feats, student_feats):
            # Normalize along the channel dimension (L2 norm)
            ft_norm = F.normalize(ft, p=2, dim=1)
            fs_norm = F.normalize(fs, p=2, dim=1)
            # Compute the MSE loss between normalized feature maps
            loss = self.criterion(fs_norm, ft_norm)
            loss_total += loss

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return loss_total

    def forward(self, x, labels=None, return_map=False):
        """
        Forward pass for test-time inference.
        For each input image, we extract pyramid features from both teacher and student,
        compute the per-pixel loss (0.5 * squared L2 difference, as in Eq. 1),
        upsample each anomaly map to the original image size, and then combine them by element-wise product.
        The maximum value in the final anomaly map serves as the anomaly score.
        """
        self.teacher.eval()
        self.student.eval()

        teacher_feats = self.extract_features(self.teacher, x)
        student_feats = self.extract_features(self.student, x)

        anomaly_maps = []
        for ft, fs in zip(teacher_feats, student_feats):
            ft_norm = F.normalize(ft, p=2, dim=1)
            fs_norm = F.normalize(fs, p=2, dim=1)
            # Compute per-pixel loss: 0.5 * ||ft - fs||^2. Sum over channels.
            diff = 0.5 * (ft_norm - fs_norm) ** 2
            map_scale = diff.sum(dim=1, keepdim=True)  # shape: (N, 1, H, W)
            # Upsample to the input size (using bilinear interpolation)
            map_up = F.interpolate(map_scale, size=x.shape[2:], mode='bilinear', align_corners=False)
            anomaly_maps.append(map_up)

        # Combine anomaly maps from each scale via element-wise multiplication
        final_map = anomaly_maps[0]
        for m in anomaly_maps[1:]:
            final_map = final_map * m

        # Compute anomaly score as the maximum value in the final anomaly map (per image)
        anomaly_score, _ = final_map.view(final_map.size(0), -1).max(dim=1)
        if return_map:
            return anomaly_score, final_map
        else:
            return anomaly_score

class ClassConditionalUninformedStudents(UninformedStudents):
    
    """
    A detector model that trains multiple student models to detect adversarial examples with class conditioning.

    Attributes:
        patch_size (int): The size of the patches.
        teacher (torch.nn.Module): The teacher model.
        teacher_feature_extractor (torch.nn.Module): The teacher feature extractor.
        students (list): The student models.
        criterion (torch.nn.Module): The loss criterion.
        optimizer (torch.optim.Optimizer): The optimizer.
    """

    def __init__(self, num_students, dataset, patch_size=5, lr=0.0001, device='cpu'):
        # Derive the number of students from the number of classes in the dataset.
        self.num_students = num_students
        super(ClassConditionalUninformedStudents, self).__init__(num_students*dataset.num_classes, dataset, patch_size, lr, device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [param for student in self.students for param in student.parameters()], lr=lr
        )
        self.to(device)
        print("Teacher: " + str(self.teacher))
        print("Students: " + str(self.students))

    def train_patch(self, patches, label=None):
        """
        Trains the student models on a batch of patches with class labels.

        Args:
            patches (torch.Tensor): The input patches.
            label (torch.Tensor): The class label.

        Returns:
            torch.Tensor: The total loss.
        """
        if label is None:
            raise ValueError("Class label must be provided for class-conditional training.")
        # Forward pass of the teacher
        teacher_outputs = self.teacher(patches).detach().squeeze()

        # Select the student model corresponding to the given label
        students = self.students[label.cpu().item()*self.num_students:(label.cpu().item()+1)*self.num_students]

        # Accumulate losses from each student model
        total_loss = 0
        for student in students:
            student_outputs = student(patches).squeeze()
            loss = self.criterion(student_outputs, teacher_outputs)
            total_loss += loss

        # Perform a single backward pass and optimizer step for all students
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def forward(self, image, label=None):
        """
        Forward pass of the detector model using only the student corresponding to the given label.

        Args:
            image (torch.Tensor): The input image.
            label (torch.Tensor): The class label.

        Returns:
            torch.Tensor, torch.Tensor: The regression error and the predictive uncertainty (set to 0).
        """
        if label is None:
            raise ValueError("Class label must be provided for class-conditional training.")
        
        students = self.students[label.cpu().item()*self.num_students:(label.cpu().item()+1)*self.num_students]

        # Extract patches from the input image
        patches = extract_patches(image, self.patch_size)
        
        # Initialize lists to store teacher and student outputs
        teacher_outputs = []
        student_outputs = {idx: [] for idx in range(len(self.students))}

        # Iterate over each patch
        for patch in patches:
            patch = patch.to(self.device)
            # Forward pass of the teacher model
            teacher_output = self.teacher(patch).detach().squeeze()
            teacher_outputs.append(teacher_output)

            # Forward pass of each student model
            for student_idx, student in enumerate(students):
                student_output = student(patch)
                student_outputs[student_idx].append(student_output.squeeze())

        # Concatenate teacher and student outputs
        teacher_flat = torch.cat(teacher_outputs, dim=0)
        student_flat = torch.stack([torch.cat(student_outputs[i], dim=0) for i in range(len(students))])

        # Calculate mean of student outputs
        students_mean = student_flat.mean(dim=0)

        # Calculate squared differences for regression error
        squared_diffs = (students_mean - teacher_flat) ** 2

        # Regression error: Mean of squared differences across all students
        regression_error = squared_diffs.mean(dim=0)

        # Calculate predictive uncertainty (variance)
        predictive_uncertainty = student_flat.var(dim=0)

        return regression_error.item(), predictive_uncertainty.mean().item()
    
def initialize_detector(config, dataset, device):
    """
    Initialize the appropriate detector model based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        dataset: Dataset object
        device: PyTorch device
    
    Returns:
        detector: Initialized detector model
    """
    method = config['method']
    
    if method == 'STFPM':
        detector = STFPM(dataset, device, config['train']['learning_rate'])
    elif method == 'ClassConditionalUninformedStudents':
        detector = ClassConditionalUninformedStudents(
            config['num_students'], 
            dataset, 
            patch_size=config['patch_size'], lr=config['train']['learning_rate'],
            device=device
        )
    elif method == 'UninformedStudents':
        detector = UninformedStudents(config, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return detector