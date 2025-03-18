import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flatten a nested dictionary.
    
    Args:
        d (dict): Dictionary to flatten
        parent_key (str): Key of the parent dictionary
        sep (str): Separator between nested keys
    
    Returns:
        dict: Flattened dictionary
    """
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def load_config(config_path):
    """
    Load a JSON configuration file.
    
    Args:
        config_path (str): Path to the JSON config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def setup_attack_kwargs(config):
    """
    Set up PGD attack parameters based on the configuration.
    
    Args:
        config (dict): Attack configuration dictionary
        
    Returns:
        dict: Attack keyword arguments
        
    Raises:
        ValueError: If attack type is not supported
    """
    if config['type'] == 'pgd':
        return {
            'constraint': config['constraint'],
            'eps': config['epsilon'],
            'step_size': 2.5 * config['epsilon'] / config['iterations'],  # Following original PGD paper
            'iterations': config['iterations'],
            'targeted': config['targeted'],
            'custom_loss': None
        }
    else:
        raise ValueError(f"Attack type '{config['type']}' not supported")


def get_targeted(attack_config):
    """
    Parse targeted attack configuration.
    
    Args:
        attack_config (dict): Attack configuration
        
    Returns:
        tuple: (targeted (bool), str_targeted (str))
        
    Raises:
        ValueError: If targeted value is invalid
    """
    if attack_config["targeted"] in [True, 1]:
        targeted = True
        str_targeted = "T"
    elif attack_config["targeted"] in [False, -1]:
        targeted = False  # Fixed bug: was incorrectly set to True
        str_targeted = "U"
    else:
        raise ValueError(f"Invalid targeted value: {attack_config['targeted']}")
    return targeted, str_targeted


class UnifiedSummaryWriter(SummaryWriter):
    """
    A custom SummaryWriter that logs hyperparameters using add_hparams
    directly into the same run as the scalar metrics.
    """
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, global_step=None):
        """
        Add hyperparameters and metrics to the same TensorBoard run.
        
        Args:
            hparam_dict (dict): Hyperparameters
            metric_dict (dict): Metrics
            hparam_domain_discrete (dict, optional): Discrete domain for hyperparameters
            global_step (int, optional): Global step value for the summary
        """
        # Generate the hyperparameter summary events
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)
        
        # Write summaries to the current file writer
        self.file_writer.add_summary(exp, global_step)
        self.file_writer.add_summary(ssi, global_step)
        self.file_writer.add_summary(sei, global_step)
        
        # Log metric values associated with the hyperparameters
        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step)
