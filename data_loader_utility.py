from torchvision.datasets.mnist import FashionMNIST
from torch.utils.data import DataLoader, random_split
from torch import manual_seed
from torchvision.transforms import ToTensor

### TODO -- UPDATE SO THAT YOU CAN JUST LOAD THE DATA WO A PROBLEM

def get_dataloaders():
    # set random seed so that all splits from train/val will be the same
    manual_seed(64)

    dataset_60k = FashionMNIST(root='data', 
                           train=True,  # setting this parameter equal to True loads the train dataset, and False will load the test one
                           download=True,  # only downloads the data if you don't already have it
                           transform=ToTensor()  # necessary to load a tensor object and not an image object; will allow us to add a more thorough transformation function later
                           )

    # Split the data into train and val sets (50k/10k split to match the 10k test set)
    train_set, val_set = random_split(dataset_60k, [50000, 10000])

    # Create dataloaders for the train and val sets
    train_loader = DataLoader(train_set, batch_size=32)
    val_loader = DataLoader(val_set, batch_size=32)

    # load the test set
    test_dataset = FashionMNIST(root='data', 
                           train=False,  # set to False to download the test set
                           download=True,
                           transform=ToTensor()
                           )
    
    # test dataloader
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, val_loader, test_loader


def get_tabular_data():  # in case we decide to do any traditional ML stuff, we can have a regular data matrix (n x 784)
    """Return train/test dataset in a matrix format (n x 784), with a return statement like the sklearn train_test_split function (X_train, X_test, y_train, y_test).
    """

    manual_seed(64)

    dataset_60k = FashionMNIST(root='data', 
                           train=True,  # setting this parameter equal to True loads the train dataset, and False will load the test one
                           download=False,  # only downloads the data if you don't already have it
                           transform=ToTensor()  # necessary to load a tensor object and not an image object; will allow us to add a more thorough transformation function later
                           )

    train_data = dataset_60k.data.reshape(60000, 784)
    train_labels = dataset_60k.targets

    # load the test set
    test_dataset = FashionMNIST(root='data', 
                           train=False,  # set to False to download the test set
                           download=False,
                           transform=ToTensor()
                           )
    
    test_data = test_dataset.data.reshape(10000, 784)
    test_labels = test_dataset.targets
    
    return train_data, train_labels, test_data, test_labels


# labels for the classes
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


if __name__ == '__main__':
    get_dataloaders()