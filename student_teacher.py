import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from robustness import attacker, datasets
from utils import extract_patches, initialize_model


class Detector(nn.Module):
    """
    A detector model that trains multiple student models to detect adversarial examples.

    Args:
        num_students (int): The number of student models.
        attack_kwargs (dict): The keyword arguments for the attacker.
        patch_size (int, optional): The size of the patches. Defaults to 5.
        device (str, optional): The device to use for training. Defaults to 'cuda'.

    Attributes:
        device (torch.device): The device used for training.
        dataset (robustness.datasets.Dataset): The dataset object.
        patch_size (int): The size of the patches.
        teacher (torch.nn.Module): The teacher model.
        teacher_feature_extractor (torch.nn.Module): The teacher feature extractor.
        students (list): The student models.
        criterion (torch.nn.Module): The loss criterion.
        optimizer (torch.optim.Optimizer): The optimizer.
        attacker (robustness.attacker.Attacker): The attacker.
        attack_kwargs (dict): The keyword arguments for the attacker.
        writer (torch.utils.tensorboard.SummaryWriter): The TensorBoard writer.
    """

    def __init__(self, num_students, attack_kwargs, patch_size=5, device='cuda'):
        super(Detector, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset = datasets.CIFAR('../cifar10-challenge/cifar10_data')
        self.patch_size = patch_size
        self.teacher, self.teacher_feature_extractor, self.students = initialize_model(num_students, self.dataset, device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [param for student in self.students for param in student.parameters()], lr=0.000001
        )
        self.attacker = attacker.Attacker(self.teacher, self.dataset)
        self.attack_kwargs = attack_kwargs
        self.to(self.device)
        self.writer = SummaryWriter()  # TensorBoard writer
        print("Teacher: " + str(self.teacher_feature_extractor))
        print("Students: " + str(self.students))

    def evaluate(self, test_loader, eval_iter, num_batches=1000):
        """
        Evaluates the detector model on the test dataset.

        Args:
            test_loader (torch.utils.data.DataLoader): The test data loader.
            eval_iter (int): The evaluation iteration.
            num_batches (int, optional): The number of batches to evaluate. Defaults to 1000.
        """
        # Initialize lists to store average and maximum standard deviations
        avg_std_list = []
        max_std_list = []

        # Iterate over the test data loader
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            
            # Forward pass through the model to obtain standard deviation map
            std_map, _ = self.forward(inputs)
            
            # Calculate average and maximum standard deviations
            avg_std = torch.mean(std_map).item()
            max_std = torch.max(std_map).item()

            # Append the values to the respective lists
            avg_std_list.append(avg_std)
            max_std_list.append(max_std)

            # Break the loop if the desired number of batches is reached
            if ((batch_idx + 1) % num_batches == 0):
                break

        # Calculate the top 1% quantiles of average and maximum standard deviations
        avg_std_top_1_percent = torch.quantile(torch.tensor(avg_std_list), 0.99).item()
        max_std_top_1_percent = torch.quantile(torch.tensor(max_std_list), 0.99).item()

        # Log the top 1% quantiles to TensorBoard
        self.writer.add_scalar('Evaluation/Top_1_percent_avg_std', avg_std_top_1_percent, eval_iter)
        self.writer.add_scalar('Evaluation/Top_1_percent_max_std', max_std_top_1_percent, eval_iter)

        # Log histograms of average and maximum standard deviations to TensorBoard
        self.writer.add_histogram('Histograms/avg_std', torch.tensor(avg_std_list), eval_iter)
        self.writer.add_histogram('Histograms/max_std', torch.tensor(max_std_list), eval_iter)

        # Initialize lists to store average and maximum standard deviations for adversarial examples
        avg_std_list_adv = []
        max_std_list_adv = []
        acc_avg = 0
        acc_max = 0

        # Iterate over the test data loader again for adversarial examples
        for batch_idx, (inputs, label) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            
            # Generate adversarial examples using the attacker
            target_label = (label + torch.randint_like(label, high=9)) % 10
            adv_im = self.attacker(inputs.cuda(), target_label.cuda(), True, **self.attack_kwargs)
            
            # Forward pass through the model to obtain standard deviation map for adversarial examples
            std_map, _ = self.forward(adv_im)
            
            # Calculate average and maximum standard deviations for adversarial examples
            avg_std = torch.mean(std_map).item()
            max_std = torch.max(std_map).item()

            # Append the values to the respective lists
            avg_std_list_adv.append(avg_std)
            max_std_list_adv.append(max_std)

            # Count the number of adversarial examples that have higher standard deviations than the top 1% quantiles
            if avg_std > avg_std_top_1_percent:
                acc_avg += 1
            if max_std > max_std_top_1_percent:
                acc_max += 1

            # Break the loop if the desired number of batches is reached
            if ((batch_idx + 1) % num_batches == 0):
                break
        
        # Log the accuracy of adversarial examples to TensorBoard
        self.writer.add_scalar('Evaluation/acc_avg', acc_avg/num_batches, eval_iter)
        self.writer.add_scalar('Evaluation/acc_max', acc_max/num_batches, eval_iter)

        # Log histograms of average and maximum standard deviations for adversarial examples to TensorBoard
        self.writer.add_histogram('Histograms/adv_avg_std', torch.tensor(avg_std_list_adv), eval_iter)
        self.writer.add_histogram('Histograms/adv_max_std', torch.tensor(max_std_list_adv), eval_iter)

    def train_patch(self, patches):
        """
        Trains the student models on a batch of patches.

        Args:
            patches (torch.Tensor): The input patches.

        Returns:
            torch.Tensor: The total loss.
        """
        # Forward pass of the teacher
        teacher_outputs = self.teacher_feature_extractor(patches).detach()

        # Accumulate losses from each student model
        total_loss = 0
        for student in self.students:
            student_outputs = student(patches)
            loss = self.criterion(student_outputs, teacher_outputs)
            total_loss += loss

        # Perform a single backward pass and optimizer step for all students
        self.optimizer.zero_grad()
        total_loss.backward()  # This will accumulate gradients for all students
        self.optimizer.step()

        return total_loss

    def train_model(self, train_loader, test_loader, num_epochs=10, interval=10000):
        """
        Trains the detector model.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            test_loader (torch.utils.data.DataLoader): The test data loader.
            num_epochs (int, optional): The number of epochs to train. Defaults to 10.
            interval (int, optional): The interval for evaluation. Defaults to 10000.
        """
        eval_iter = 0
        for epoch in range(num_epochs):
            # Train loop
            for batch_idx, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                patches = extract_patches(inputs, self.patch_size)

                # Reshape patches to have sufficient batch size
                patches = patches.view(-1, inputs.size(1), self.patch_size, self.patch_size)
                patches = patches.to(self.device)
                loss = self.train_patch(patches)

                # Log the training loss
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

                if ((batch_idx + 1) % interval == 0):
                    self.eval()
                    self.evaluate(test_loader, eval_iter)
                    eval_iter += 1
                    self.train()

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

            # Compute teacher output and store it
            teacher_output = self.teacher_feature_extractor(patch).detach()
            teacher_outputs.append(teacher_output)

            # Compute student output for each student and store them
            for student_idx, student in enumerate(self.students):
                student_output = student(patch)
                student_outputs[student_idx].append(student_output)

        # Concatenate teacher outputs
        teacher_flat = torch.cat(teacher_outputs, dim=0)

        # Initialize lists to store standard deviation maps and bias maps
        std_maps = []
        bias_maps = []

        # Compute standard deviation map and bias map for each student
        for student_idx in range(len(self.students)):
            student_flat = torch.cat(student_outputs[student_idx], dim=0)
            std_map = torch.std(student_flat - teacher_flat, dim=0)
            bias_map = torch.mean(student_flat - teacher_flat, dim=0)
            std_maps.append(std_map)
            bias_maps.append(bias_map)

        # Compute average standard deviation map and average bias map
        avg_std_map = torch.mean(torch.stack(std_maps), dim=0)
        avg_bias_map = torch.mean(torch.stack(bias_maps), dim=0)

        return avg_std_map, avg_bias_map


# Test code
if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(description='Adversarial Example Generator')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon constraint (L-inf norm)')
    parser.add_argument('--linf', type=bool, default=False, help='L-inf constraint (True) or L2 (False)')
    parser.add_argument('--targeted', type=bool, default=False, help='Chose if the attach is targeted (True) or untargeted (False)')
    parser.add_argument('--iterations', type=int, default=100, help='Number of PGD steps')
    args = parser.parse_args()

    # Use parsed arguments
    epsilon = args.epsilon
    linf = args.linf
    iterations = args.iterations
    targeted = args.targeted

    if not linf:
        attack_kwargs = {
            'constraint': 'inf', # L-inf PGD 
            'eps': epsilon, # Epsilon constraint (L-inf norm)
            'step_size': 0.01, # Learning rate for PGD
            'iterations': iterations, # Number of PGD steps
            'targeted': targeted, # Targeted attack
            'custom_loss': None # Use default cross-entropy loss
        }
    else:
        attack_kwargs = {
            'constraint': '2', # L-inf PGD 
            'eps': epsilon, # Epsilon constraint (L-inf norm)
            'step_size': 2.5*epsilon/100, # Learning rate for PGD
            'iterations': iterations, # Number of PGD steps
            'targeted': targeted, # Targeted attack
            'custom_loss': None # Use default cross-entropy loss
        }

    # Example of how to use the Detector class
    detector = Detector(10, attack_kwargs, patch_size=9)
    train_loader, test_loader = detector.dataset.make_loaders(workers=4, batch_size=1)
    detector.train_model(train_loader, test_loader)
