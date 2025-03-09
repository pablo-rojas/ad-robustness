import os
import torch
import torch.nn as nn
from src.model_utils import extract_patches, initialize_us_models, resnet18_classifier
import torchvision.models as models
import torch.nn.functional as F


class UninformedStudents(nn.Module):
    """
    A detector model that trains multiple student models to detect adversarial examples.

    Attributes:
        patch_size (int): The size of the patches.
        teacher (torch.nn.Module): The teacher model.
        teacher_feature_extractor (torch.nn.Module): The teacher feature extractor.
        students (list): The student models.
        criterion (torch.nn.Module): The loss criterion.
        optimizer (torch.optim.Optimizer): The optimizer.
    """

    def __init__(self, num_students, dataset, patch_size=5, lr=0.0001, weight_decay=0, device='cpu'):
        super(UninformedStudents, self).__init__()
        self.patch_size = patch_size
        self.teacher, self.teacher_feature_extractor, self.students = initialize_us_models(num_students, dataset, patch_size, device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [param for student in self.students for param in student.parameters()], lr=lr, weight_decay=weight_decay
        )
        self.to(device)
        print("Teacher: " + str(self.teacher_feature_extractor))
        print("Students: " + str(self.students))
    
    def save(self, path):
        """
        Saves the teacher, teacher feature extractor, and student models to the specified path.

        Args:
            path (str): The directory where the models will be saved.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.teacher.state_dict(), os.path.join(path, 'teacher.pth'))
        torch.save(self.teacher_feature_extractor.state_dict(), os.path.join(path, 'teacher_feature_extractor.pth'))
        for idx, student in enumerate(self.students):
            torch.save(student.state_dict(), os.path.join(path, f'student_{idx}.pth'))
        
        # Save extra variables if they exist.
        extra_vars = {}
        if hasattr(self, 'e_mean'):
            extra_vars['e_mean'] = self.e_mean
        if hasattr(self, 'e_std'):
            extra_vars['e_std'] = self.e_std
        if hasattr(self, 'v_mean'):
            extra_vars['v_mean'] = self.v_mean
        if hasattr(self, 'v_std'):
            extra_vars['v_std'] = self.v_std

        if extra_vars:
            torch.save(extra_vars, os.path.join(path, 'extra_vars.pth'))

        print(f"Models saved to {path}")

    def load(self, path):
        """
        Loads the teacher, teacher feature extractor, and student models from the specified path.

        Args:
            path (str): The directory from where the models will be loaded.
        """
        self.teacher.load_state_dict(torch.load(os.path.join(path, 'teacher.pth')))
        self.teacher_feature_extractor.load_state_dict(torch.load(os.path.join(path, 'teacher_feature_extractor.pth')))
        for idx, student in enumerate(self.students):
            student.load_state_dict(torch.load(os.path.join(path, f'student_{idx}.pth')))
        
        # Load extra variables if the file exists.
        extra_vars_path = os.path.join(path, 'extra_vars.pth')
        if os.path.exists(extra_vars_path):
            extra_vars = torch.load(extra_vars_path)
            for key, value in extra_vars.items():
                setattr(self, key, value)

        print(f"Models loaded from {path}")

    def to(self, device):
        """
        Moves the teacher and student models to the specified device.

        Args:
            device (torch.device): The device to move the models to.
        """
        self.device = device
        self.teacher.to(device)
        self.teacher_feature_extractor.to(device)
        for student in self.students:
            student.to(device)

    def train_patch(self, patches, label=None):
        """
        Trains the student models on a batch of patches.

        Args:
            patches (torch.Tensor): The input patches.
            label (torch.Tensor, optional): The class label. Defaults to None.

        Returns:
            torch.Tensor: The total loss.
        """
        # Forward pass of the teacher - compute once for all students
        teacher_outputs = self.teacher_feature_extractor(patches).detach().squeeze()
        
        # Preallocate tensor for student outputs.
        # We assume each student's output (after squeeze) matches teacher_outputs shape.
        num_students = len(self.students)
        output_shape = teacher_outputs.shape  # e.g., (batch_size, features)
        all_student_outputs = torch.empty((num_students,) + output_shape, 
                                        device=patches.device, dtype=teacher_outputs.dtype)
        
        # Fill the preallocated tensor with each student's output
        for i, student in enumerate(self.students):
            # Compute output and squeeze to match teacher_outputs shape.
            all_student_outputs[i] = student(patches).squeeze()

        # Instead of expanding teacher_outputs explicitly, unsqueeze so broadcasting happens.
        # This gives a shape [1, batch_size, features] which will broadcast to [num_students, batch_size, features].
        #teacher_outputs_unsqueezed = teacher_outputs.unsqueeze(0)

        teacher_outputs_unsqueezed = teacher_outputs.unsqueeze(0).expand(num_students, *teacher_outputs.shape)


        # Compute loss for all students at once.
        # If self.criterion is nn.MSELoss with default "mean" reduction, this returns a scalar.
        losses = self.criterion(all_student_outputs, teacher_outputs_unsqueezed)
        total_loss = losses.sum()  # If losses is scalar, .sum() is a no-op; adjust if using "none" reduction.

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def train_step(self, x, labels=None):
        """
        Performs a training step on a batch of anomaly-free images.
        Extracts patches and trains the student models.

        Args:
            x (torch.Tensor): The input images.
            labels (torch.Tensor, optional): The class labels. Defaults to None.

        Returns:
            torch.Tensor: The total loss.
        """
        # Extract patches from all images in the batch
        patches = extract_patches(x, self.patch_size)
        
        # Reshape to batch format
        patches = patches.view(-1, x.size(1), self.patch_size, self.patch_size)
        
        # Train on patches
        loss = self.train_patch(patches, labels)

        return loss

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

    def forward(self, image, label=None):
        """
        Forward pass of the detector model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            torch.Tensor, torch.Tensor: The regression error and predictive uncertainty.
        """
        # Extract patches from the input image
        # This returns a 4D tensor (num_patches, C, patch_size, patch_size)
        patches = extract_patches(image, self.patch_size)
        
        # We can process all patches in a single batch for efficiency
        # Forward pass of the teacher model
        teacher_outputs = self.teacher_feature_extractor(patches).detach().squeeze()
        
        # Forward pass of each student model
        all_student_outputs = torch.stack([student(patches).squeeze() for student in self.students])
        
        # Calculate mean of student outputs across all students
        students_mean = all_student_outputs.mean(dim=0)
        
        # Calculate regression error (squared difference between mean student output and teacher output)
        squared_diffs = (students_mean - teacher_outputs) ** 2
        regression_error = squared_diffs.mean()
        
        # Calculate predictive uncertainty (variance across student models)
        predictive_uncertainty = all_student_outputs.var(dim=0).mean()
        
        return regression_error.item(), predictive_uncertainty.item()

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
        print("Teacher: " + str(self.teacher_feature_extractor))
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
        teacher_outputs = self.teacher_feature_extractor(patches).detach().squeeze()

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
            teacher_output = self.teacher_feature_extractor(patch).detach().squeeze()
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