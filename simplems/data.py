import numpy as np
from .autograd import Tensor
import os
import pickle
import gzip
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from simplems import backend_ndarray as nd

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        
        if flip_img:
            img = img[:, ::-1, :]
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        
        H, W, C = img.shape
        pad = np.zeros((H + 2 * self.padding, W + 2 * self.padding, C))
        pad[self.padding:self.padding + H, self.padding:self.padding + W, :] = img
        newx, newy = self.padding + shift_x, self.padding + shift_y
        crop = pad[newx:newx + H, newy:newy + W, :]
        return crop


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = len(dataset)
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        
        self.index = 0
        if self.shuffle:
            indexes = np.arange(self.n)
            np.random.shuffle(indexes)
            self.ordering = np.array_split(indexes,
                                           range(self.batch_size, self.n, self.batch_size))
        
        return self

    def __next__(self):
        
        if self.index == len(self.ordering):
            raise StopIteration

        res = [Tensor(x) for x in self.dataset[self.ordering[self.index]]]
        self.index += 1

        return tuple(res)
        

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        self.X, self.y = self.parse_mnist(image_filename, label_filename)
        

    def __getitem__(self, index) -> object:
        x = self.X[index]
        y = self.y[index]
        n = len(x.shape)
        if n == 1:
            # 单索引情形
            x = x.reshape(28, 28, -1)
            x = self.apply_transforms(x)
            x = x.reshape(28, 28, 1)
        else:
            # 多索引情形
            m = x.shape[0]
            x = x.reshape(m, 28, 28, -1)
            for i in range(m):
                x[i] = self.apply_transforms(x[i])
        return x, y
        

    def __len__(self) -> int:
        return self.X.shape[0]
        

    def parse_mnist(self, image_filename, label_filename):
        """ Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format

        Returns:
            Tuple (X,y):
                X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                    data.  The dimensionality of the data should be
                    (num_examples x input_dim) where 'input_dim' is the full
                    dimension of the data, e.g., since MNIST images are 28x28, it
                    will be 784.  Values should be of type np.float32, and the data
                    should be normalized to have a minimum value of 0.0 and a
                    maximum value of 1.0.

                y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                    labels of the examples.  Values should be of type np.int8 and
                    for MNIST will contain the values 0-9.
        """
    
    #使用gzip打开文件
        with gzip.open(image_filename, "rb") as f:
            X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype('float32') / 255
        with gzip.open(label_filename, "rb") as f:
            y = np.frombuffer(f.read(), np.uint8, offset=8)
        return X, y
    


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        super().__init__(transforms)
        if train:
            files = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5"
            ]
        else:
            files = [
                "test_batch"
            ]
        X = []
        y = []
        for file in files:
            data, labels = self.unpickle(os.path.join(base_folder, file))
            X.append(data)
            y.append(labels)
        X = np.concatenate(X) / 255.0
        y = np.concatenate(y)
        self.n = X.shape[0]
        X = X.reshape(self.n, -1, 32, 32)
        self.X = X
        self.y = y
        

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict[b'data'], dict[b'labels']

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        x = self.apply_transforms(self.X[index])
        y = self.y[index]
        return x, y
        

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        return self.n


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.n = 0

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.n
            self.idx2word.append(word)
            self.n += 1

        return self.word2idx[word]
        

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        return self.n


class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    eos = "<eos>"
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ids = []
        with open(path) as f:
            for i, sentence in enumerate(f.readlines()):
                if i == max_lines:
                    break
                for word in sentence.split():
                    id = self.dictionary.add_word(word)
                    ids.append(id)
                # add <eos>
                id = self.dictionary.add_word(Corpus.eos)
                ids.append(id)
        return ids
        


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    n = len(data)
    m = n // batch_size * batch_size
    data = data[:m]
    # [1,2,3,4,5,6] -> [[1,2],[3,4],[5,6]] -> [[1,3,5], [2,4,6]]
    array = np.array(data).reshape(batch_size, -1).transpose(1, 0)
    return array


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    data = batches[i : i+bptt]
    target  = batches[i+1 : i+bptt+1]
    n = min(data.shape[0], target.shape[0])
    data = data[:n]
    target = target[:n].flatten()

    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype)
    