import os
import torch
import torch.nn as nn

from model_utils import extract_patches, initialize_model


class Detector(nn.Module):
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

    def __init__(self, num_students, dataset, patch_size=5, device='cpu'):
        super(Detector, self).__init__()
        self.patch_size = patch_size
        self.teacher, self.teacher_feature_extractor, self.students = initialize_model(num_students, dataset)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [param for student in self.students for param in student.parameters()], lr=0.0001
        )
        self.to(device)
        print("Teacher: " + str(self.teacher_feature_extractor))
        print("Students: " + str(self.students))

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


    def train_patch(self, patches):
        """
        Trains the student models on a batch of patches.

        Args:
            patches (torch.Tensor): The input patches.

        Returns:
            torch.Tensor: The total loss.
        """
        # Forward pass of the teacher
        teacher_outputs = self.teacher_feature_extractor(patches).detach().squeeze()

        # Accumulate losses from each student model
        total_loss = 0
        for student in self.students:
            student_outputs = student(patches).squeeze()
            loss = self.criterion(student_outputs, teacher_outputs)
            total_loss += loss

        # Perform a single backward pass and optimizer step for all students
        self.optimizer.zero_grad()
        total_loss.backward()
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

    def forward(self, image):
        """
        Forward pass of the detector model.

        Args:
            image (torch.Tensor): The input image.

        Returns:
            torch.Tensor, torch.Tensor: The average standard deviation map and average bias map.
        """
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
            for student_idx, student in enumerate(self.students):
                student_output = student(patch)
                student_outputs[student_idx].append(student_output.squeeze())

        # Concatenate teacher and student outputs
        teacher_flat = torch.cat(teacher_outputs, dim=0)
        student_flat = torch.stack([torch.cat(student_outputs[i], dim=0) for i in range(len(self.students))])

        # Calculate mean of student outputs
        students_mean = student_flat.mean(dim=0)

        # Calculate squared differences for regression error
        squared_diffs = (students_mean - teacher_flat) ** 2

        # Regression error: Mean of squared differences across all students
        regression_error = squared_diffs.mean(dim=0)

        # Calculate predictive uncertainty (variance)
        predictive_uncertainty = student_flat.var(dim=0)

        return regression_error.item(), predictive_uncertainty.mean().item()
