
"""
Created on Sun Nov 13 16:23:15 2022
# this is the file to load cifar 100 as id and corresponding oods and model from OpenOOD framework
@author: saiful
"""
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '..')
import pandas as pd
from models import cnn, resnet, densenet, lenet
from models.resnet_react import resnet50, resnet18, resnet34, resnet101
from models.transformer import ImageNetTransformer
from models.cifar100_transformer import Cifar100Transformer
from models.cifar10_transformer import Cifar10Transformer
import torch
import pickle
from torchvision import transforms
from pathlib import Path
from utils import get_torch_device
from os import chdir
from document_id_ood_n_model_loader import load_resnet50_model_for_document_dataset
sys.path.insert(0, '..')
sys.path.insert(0, '..')  # Enable import from parent folder.
sys.path.insert(0, '..')  # Enable import from parent folder.

# =============================================================================
# openood
# =============================================================================

# model, trans= load_openood_model for cifar10()


def load_openood_model():
    import OpenOOD
    old_path = Path.cwd()
    # provide the directory of the /confidence-magesh/OpenOOD folder
    chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    temp_path = Path.cwd()
    # loading ood data for cifar10 from openood
    # change directory to //home/saiful/confidence-master/OpenOOD

    sys.path.append(
        "/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    from OpenOOD.openood_id_ood_and_model_cifar10 import get_model_from_openood
    resnet18_on_cifar = get_model_from_openood()
    chdir(old_path)

    # transform = transforms.Normalize(
        # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Normalize((0, 0, 0), (1, 1, 1))
    # transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return resnet18_on_cifar.eval(), transform


def load_openood_model_for_cifar100():
    import OpenOOD
    old_path = Path.cwd()
    chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    temp_path = Path.cwd()
    # loading ood data for cifar10 from openood
    # change directory to //home/saiful/confidence-master/OpenOOD

    sys.path.append(
        "/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    from OpenOOD.openood_id_ood_and_model_cifar100 import get_model_from_openood_for_cifar100
    resnet18_on_cifar100 = get_model_from_openood_for_cifar100()
    chdir(old_path)

    # transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Normalize((0, 0, 0), (1, 1, 1))
    # transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    print("##   ##   ##  ##  ##  #")
    return resnet18_on_cifar100.eval(), transform


def load_openood_model_for_imagenet():
    import OpenOOD
    old_path = Path.cwd()
    chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    temp_path = Path.cwd()
    # loading ood data for cifar10 from openood
    # change directory to //home/saiful/confidence-master/OpenOOD

    sys.path.append(
        "/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    from OpenOOD.openood_id_ood_and_model_imagenet import get_model_from_openood_for_imagenet
    resnet50_on_imagenet = get_model_from_openood_for_imagenet()
    chdir(old_path)

    # transform = transforms.Normalize(
    # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Normalize((0, 0, 0), (1, 1, 1))
    # transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    print("##   ##   ##  ##  ##  #")
    return resnet50_on_imagenet.eval(), transform

def load_openood_model_for_mnist():
    import OpenOOD
    old_path=Path.cwd()
    chdir("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    temp_path=Path.cwd()
    # loading ood data for cifar10 from openood
    # change directory to //home/saiful/confidence-master/OpenOOD

    sys.path.append("/home/saiful/confidence-magesh_MR/confidence-magesh/OpenOOD")
    
    from OpenOOD.openood_id_ood_and_model_mnist import get_model_from_openood_for_mnist
    lenet_on_mnist = get_model_from_openood_for_mnist()
    chdir(old_path)
    
    # transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Normalize((0, 0, 0), (1, 1, 1))
    # transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    print("##   ##   ##  ##  ##  #")
    return lenet_on_mnist.eval(), transform
# =============================================================================
#
# =============================================================================


def load_model(dataset, model):
    device = get_torch_device()
    
    if dataset == "imagenet":
        if model == "resnet18":
            m = resnet18(pretrained=True, num_classes=1000).to(device)
            transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif model == "resnet34":
            m = resnet34(pretrained=True, num_c=1000).to(device)
            transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif model == "resnet50":

            # m = resnet50(pretrained=True, num_classes=1000).to(device)
            # transform = transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            m, transform = load_openood_model_for_imagenet()
        elif model == "resnet101":
            m = resnet101(pretrained=True, num_classes=1000).to(device)
            transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif model == "transformer":
            m = ImageNetTransformer(device)
            transform = transforms.Compose([])  # No input transform
        else:
            raise Exception(f"Unknown model: {model}")
    
    
    elif dataset == "document":
        if model == "resnet50_docu":
            m, transform = load_resnet50_model_for_document_dataset()
            
    elif dataset == "mnist":
        m, transform = load_openood_model_for_mnist()
        
    elif model == "cifar10transformer":
        m = Cifar10Transformer(device)
        transform = transforms.Compose([])  # No input transform
        
    elif model == "cifar100transformer":
        m = Cifar100Transformer(device)
        transform = transforms.Compose([])  # No input transform
        
    else:
        path = Path(__file__).parent / f"{dataset}/{model}/"
        # # It will work only for cifar 10 and cofar 100
        # state_dict = torch.load(path / "state_dict.pt", map_location=device)
        
        if dataset == "mnist_original":
          
            checkpoint = torch.load(
                path / "mnist_lenet_acc99.60.ckpt", map_location=device)
            if model == "lenet":
                m = lenet.LeNet(num_classes=10).to(device)
                print(type(checkpoint))
                print(checkpoint.keys())
                m.load_state_dict(checkpoint)
                transform = transforms.Compose([])

        elif model == "cnn":  # Small CNN, mostly for testing.
            m = cnn.ReluNet().to(device)
            # m.load_state_dict(state_dict)
            transform = transforms.Compose([])
        elif model == "resnet":  # ResNet34
            """
            # Changed to ResNet18 for OpenOOD
            #checkpoint = torch.load(path / "cifar10_res18_acc94.30.ckpt", map_location=device)
            #sys.path.insert(0, '/home/magesh/confidence/OpenOOD/')
            #with open(path / "18_32x32_for_cifar10.pickle", 'rb') as handle:
             #   m=pickle.load(handle)
            m = resnet.ResNet18(100 if dataset == "cifar100" else 10).to(device)
            #m = resnet.ResNet18(100 if dataset == "cifar100" else 10).to(device)
            m.load_state_dict(state_dict, strict=False) # Removed this for ResNet 18
            #m.load_state_dict(checkpoint, strict=False)
            #print("Model: ", m)
            print("Load state dict for ResNet18 is passed")
            transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transform = transforms.Normalize((0, 0, 0), (1, 1, 1))
            """

        # elif model=="resnet18":
            ##
            if dataset == "cifar10":
                m, transform = load_openood_model() #.to(device)
                print("m :", m)
            elif dataset == "cifar100":
                m, transform = load_openood_model_for_cifar100()
            ##
        elif model == "densenet":  # DenseNet3
            m = densenet.DenseNet3(100, 100 if dataset ==
                                   "cifar100" else 10).to(device)
            # m.load_state_dict(state_dict, strict=False)
            transform = transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                             (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0))
        else:
            raise Exception(f"Unknown model: {model}")
        try:
            m.threshold = pd.read_csv(
                path / "percentiles.csv", index_col=0).loc[90].iloc[0]
        except FileNotFoundError:
            print("No threshold found.")
    return m.eval(), transform


def save_state_dict(path, name):
    device = torch.device("cpu")
    model = torch.load(path / name, device).eval()
    torch.save(model.state_dict(), path / "state_dict.pt")


if __name__ == "__main__":
    # path = Path("cifar100/densenet")
    # save_state_dict(path, "densenet_cifar100.pth")

    # load_openood_model_for_cifar100()
    load_openood_model_for_imagenet()
