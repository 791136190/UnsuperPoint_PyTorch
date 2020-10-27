from abc import ABCMeta, abstractmethod
import torch
import torch.utils.data as torch_data
from Unsuper.utils.utils import dict_update

class BaseDataset(torch_data.Dataset):
    """Base model class.

    Arguments:
        config: A dictionary containing the configuration parameters.

    Datasets should inherit from this class and implement the following methods:
        `_init_dataset` and `_get_data`.
    Additionally, the following static attributes should be defined:
        default_config: A dictionary of potential default configuration values (e.g. the
            size of the validation set).
    """
    split_names = ['training', 'validation', 'test']

    def init_dataset(self):
        """Prepare the dataset for reading.

        This method should configure the dataset for later fetching through `_get_data`,
        such as downloading the data if it is not stored locally, or reading the list of
        data files from disk. Ideally, especially in the case of large images, this
        method shoudl NOT read all the dataset into memory, but rather prepare for faster
        seubsequent fetching.

        Arguments:
            config: A configuration dictionary, given during the object instantiantion.

        Returns:
            An object subsequently passed to `_getitem__`, e.g. a list of file paths and
            set splits.
        """
        raise NotImplementedError
    
    def __getitem__(self,index):
        raise NotImplementedError

    def __len__(self):
        return self.len

    def __init__(self, config, is_training):
        # Update config
        super().__init__()
        self.len = 0 # length of the dataset
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self.is_training = is_training # bool 
        self.len, self.train_files = self.init_dataset()  # image files & names