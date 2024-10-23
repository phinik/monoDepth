import torchvision
from torch.utils.data import DataLoader

from dataset_feldraum import FeldraumDataset
from dataset_torso import TorsoReal, TorsoSim, TorsoSimCycled, TorsoSimAndCycled
from transforms import *

def get_torso_real_train_loader(config, batch_size, input_shape):
    train_transforms = torchvision.transforms.Compose([
        #RandomColorJitter(hue=0.25, saturation=0.25, vibrance=0.25),
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    training_data = TorsoReal("train", train_transforms)

    return DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=config["general"]["num_workers"])

def get_torso_real_eval_loader(config, input_shape):
    test_transforms = torchvision.transforms.Compose([
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    test_data = TorsoReal("test", test_transforms)

    return DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["general"]["num_workers"])

def get_torso_sim_train_loader(config, batch_size, input_shape):
    train_transforms = torchvision.transforms.Compose([
        RandomCrop(p=0.5, min_size=512),
        #RandomColorJitter(hue=0.25, saturation=0.25, vibrance=0.25),
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    training_data = TorsoSim("train", train_transforms)

    return DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=config["general"]["num_workers"])

def get_torso_sim_eval_loader(config, input_shape):
    test_transforms = torchvision.transforms.Compose([
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    test_data = TorsoSim("test", test_transforms)

    return DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["general"]["num_workers"])

def get_torso_sim_cycled_train_loader(config, batch_size, input_shape):
    train_transforms = torchvision.transforms.Compose([
        RandomCrop(p=0.5, min_size=512),
        #RandomColorJitter(hue=0.25, saturation=0.25, vibrance=0.25),
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    training_data = TorsoSimCycled("train", train_transforms)

    return DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=config["general"]["num_workers"])

def get_torso_sim_cycled_eval_loader(config, input_shape):
    test_transforms = torchvision.transforms.Compose([
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    test_data = TorsoSimCycled("test", test_transforms)

    return DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["general"]["num_workers"])

def get_torso_sim_and_cycled_train_loader(config, batch_size, input_shape):
    train_transforms = torchvision.transforms.Compose([
        RandomCrop(p=0.5, min_size=512),
        #RandomColorJitter(hue=0.25, saturation=0.25, vibrance=0.25),
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    training_data = TorsoSimAndCycled("train", train_transforms)

    return DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=config["general"]["num_workers"])

def get_torso_sim_and_cycled_eval_loader(config, input_shape):
    test_transforms = torchvision.transforms.Compose([
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    test_data = TorsoSimAndCycled("test", test_transforms)

    return DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["general"]["num_workers"])

def get_feldraum_eval_dataloader(config, input_shape):
    test_transforms = torchvision.transforms.Compose([
        Resize(input_shape),
        Pad2Square(input_shape),
        ToTensor()
    ])

    test_data = FeldraumDataset("test", test_transforms)

    return DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["general"]["num_workers"])
