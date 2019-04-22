import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
import numpy as np
from scipy.linalg import sqrtm

SVHN_PATH = "Dataset\\SVHN"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=2,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    To be implemented by you!
    """
    sample_features = np.array(list(sample_feature_iterator)) # (1000, 512)
    testset_features = np.array(list(testset_feature_iterator)) # (26032, 512)

    sample_mean = np.mean(sample_features, axis=0, keepdims=True) # (1, 512)
    testset_mean = np.mean(testset_features, axis=0, keepdims=True) # (1, 512)

    sample_cov = np.cov(sample_features - sample_mean, rowvar=False) # (512, 512), subtracted mean for numerical stability
    testset_cov = np.cov(testset_features - testset_mean, rowvar=False) # (512, 512), subtracted mean for numerical stability

    mean_diff = np.linalg.norm(testset_mean - sample_mean) **2

    num_features = sample_cov.shape[0]
    epsilon_list = [1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-12, 0]
    dist = None
    for i, epsilon in enumerate(epsilon_list):
        cov_diff = np.trace(testset_cov + sample_cov - (2 * sqrtm(np.dot(testset_cov, sample_cov) + (np.eye(num_features) * epsilon))))
        if np.isreal(cov_diff):
            dist = mean_diff + cov_diff
        else:
            if i == 0: raise ValueError('Epsilon choices not large enough')
            print('Used epsilon: {:.1E}'.format(epsilon_list[i-1]))
            break
    return dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()
    
    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)

    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
