import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import cv2

from robustness import attacker, datasets
from model_utils import extract_patches, initialize_model
from dataset_utils import get_dataset
from eval_utils import partial_auc


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

    def __init__(self, num_students, attack_kwargs, dataset_name='mnist', patch_size=5, device='cuda'):
        super(Detector, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset = get_dataset(dataset_name)
        print("Dataset: " + str(self.dataset))
        self.patch_size = patch_size
        self.teacher, self.teacher_feature_extractor, self.students = initialize_model(num_students, self.dataset, device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            [param for student in self.students for param in student.parameters()], lr=0.0001
        )
        self.attacker = attacker.Attacker(self.teacher, self.dataset)
        self.attack_kwargs = attack_kwargs
        self.to(self.device)
        self.writer = SummaryWriter(comment=f"_{dataset_name}_l{self.attack_kwargs['constraint']}_{self.attack_kwargs['eps']}")
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

    def evaluate(self, test_loader, eval_iter, num_batches=100, save=False):
        """
        Evaluates the detector model on the test dataset.

        Args:
            test_loader (torch.utils.data.DataLoader): The test data loader.
            eval_iter (int): The evaluation iteration.
            num_batches (int, optional): The number of batches to evaluate. Defaults to 1000.
        """
        # Initialize anomaly score list
        as_list = []
        e_list = []
        u_list = []
        nat_accuracy = 0

        # Iterate over the test data loader
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            
            # Forward pass through the model to obtain the anomaly score
            regression_error, predictive_uncertainty = self.forward(inputs)

            # Forward through the teacher model to obtain the prediction
            y = self.teacher(self.dataset.normalize(inputs)).detach().cpu()

            # Calculate the natural accuracy
            nat_accuracy += (y.argmax(1) == labels).sum().item()/num_batches

            # Append the values to the anomaly score list
            e_list.append(regression_error)
            u_list.append(predictive_uncertainty)

            # Break the loop if the desired number of batches is reached
            if ((batch_idx + 1) % num_batches == 0):
                break
        
        self.e_mean = torch.tensor(e_list).mean().item()
        self.e_std = torch.tensor(e_list).std().item()
        self.v_mean = torch.tensor(u_list).mean().item()
        self.v_std = torch.tensor(u_list).std().item()

        # Normalize e_list and u_list
        e_list = [(e - self.e_mean) / self.e_std for e in e_list]
        u_list = [(u - self.v_mean) / self.v_std for u in u_list]

        # Compute the anomaly score list
        as_list = [e + u for e, u in zip(e_list, u_list)]

        # Calculate the top 1% quantile 
        threshold = torch.quantile(torch.tensor(as_list), 0.90).item()

        # Log the top 1% quantiles to TensorBoard
        self.writer.add_scalar('Detector Evaluation/threshold', threshold, eval_iter)

        # Log histograms of average and maximum standard deviations to TensorBoard
        self.writer.add_histogram('Histograms/anomaly_score', torch.tensor(as_list), eval_iter)

        # Initialize lists to store average and maximum standard deviations for adversarial examples
        adv_as_list = []
        accuracy = 0
        adv_accuracy = 0

        # Iterate over the test data loader again for adversarial examples
        for batch_idx, (inputs, label) in enumerate(test_loader):
            inputs = inputs.to(self.device)
            
            # Generate adversarial examples using the attacker
            target_label = (label + torch.randint_like(label, high=9)) % 10
            adv_im = self.attacker(inputs.to(self.device), target_label.to(self.device), True, **self.attack_kwargs)

            # Forward through the teacher model to obtain the prediction
            y = self.teacher(self.dataset.normalize(adv_im)).detach().cpu()

            # Calculate the natural accuracy
            adv_accuracy += (y.argmax(1) == labels).sum().item()/num_batches
            
            # Forward pass through the model to obtain standard deviation map for adversarial examples
            regression_error, predictive_uncertainty = self.forward(adv_im)

            # Calculate the anomaly score for adversarial examples
            anomaly_score = (regression_error - self.e_mean) / self.e_std + (predictive_uncertainty - self.v_mean) / self.v_std
            
            # Append the values to the anomaly score list
            adv_as_list.append(anomaly_score)

            # Count the number of adversarial examples that have higher standard deviations than the top 1% quantiles
            if anomaly_score > threshold:
                accuracy += 1

            if (save):
                cv2.imwrite(str(batch_idx) + "_adv.png", adv_im)

            # Break the loop if the desired number of batches is reached
            if ((batch_idx + 1) % num_batches == 0):
                break

        self.writer.add_scalar('Classifier Evaluation/natural_accuracy', nat_accuracy, eval_iter)
        self.writer.add_scalar('Classifier Evaluation/adversarial_accuracy', adv_accuracy, eval_iter)
        
        # Log the accuracy of adversarial examples to TensorBoard
        self.writer.add_scalar('Detector Evaluation/accuracy', accuracy/num_batches, eval_iter)

        # Log histograms of average and maximum standard deviations for adversarial examples to TensorBoard
        self.writer.add_histogram('Histograms/adv_anomaly_score', torch.tensor(adv_as_list), eval_iter)

        pAUC = partial_auc(as_list, adv_as_list)

        # Log the partial AUC to TensorBoard
        self.writer.add_scalar('Detector Evaluation/pAUC', pAUC, eval_iter)

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
            student_outputs = student(patches)
            loss = self.criterion(student_outputs, teacher_outputs)
            total_loss += loss

        # Perform a single backward pass and optimizer step for all students
        self.optimizer.zero_grad()
        total_loss.backward()  # This will accumulate gradients for all students
        self.optimizer.step()

        return total_loss

    def train_model(self, train_loader, test_loader, num_epochs=10, interval=5000):
        """
        Trains the detector model.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            test_loader (torch.utils.data.DataLoader): The test data loader.
            num_epochs (int, optional): The number of epochs to train. Defaults to 10.
            interval (int, optional): The interval for evaluation. Defaults to 10000.
        """
        eval_iter = 0

        # Train the model for the specified number of epochs
        for epoch in range(num_epochs):
            for batch_idx, (inputs, _) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                patches = extract_patches(self.dataset.normalize(inputs), self.patch_size)

                # Reshape patches to have sufficient batch size
                patches = patches.view(-1, inputs.size(1), self.patch_size, self.patch_size)
                patches = patches.to(self.device)
                loss = self.train_patch(patches)

                # Log the training loss
                self.writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + batch_idx)

                # Evaluate the model at regular intervals
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
        patches = extract_patches(self.dataset.normalize(image), self.patch_size)
        
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



# Test code
if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(description='Adversarial Example Generator')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
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
    dataset_name = args.dataset

    if linf:
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
    detector = Detector(10, attack_kwargs, dataset_name=dataset_name, patch_size=9)
    train_loader, test_loader = detector.dataset.make_loaders(workers=4, batch_size=1)
    detector.train_model(train_loader, test_loader)
