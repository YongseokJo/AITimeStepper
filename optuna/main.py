import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from optuna.artifacts import FileSystemArtifactStore

import argparse
from joblib import Parallel, delayed, parallel_backend 
from functools import partial


import sys
sys.path.append("../src/")
from structures import *
from losses import *
from trainer import *

dtype = torch.double
torch.set_default_dtype(dtype)
#torch.autograd.set_detect_anomaly(True)

# Set the device to CUDA if available, otherwise CPU

# Constants
BATCHSIZE = 32
EPOCHS = 10
TargetEnergyError = 1e-2
DIR = os.getcwd()
#print("Using device:", DEVICE)

# Define the base path for storing artifacts
artifact_base_path = "../data/optuna/artifacts"
os.makedirs(artifact_base_path, exist_ok=True)

# Initialize the artifact store
artifact_store = FileSystemArtifactStore(base_path=artifact_base_path)



def get_dataset(device):
    # Example unsupervised data: 100 samples with 10 features each
    data = np.load("../data/three_body_train_data.npy")
    num_samples = data.shape[0]
    num_samples_start = int(data.shape[0]*0.9)
    #num_samples_start = 0
    num_samples_end   = data.shape[0]
    num_samples = num_samples_end - num_samples_start
    num_features = data.shape[1]

    # Placeholder for input (magnitudes of velocities and accelerations)
    data_org = torch.tensor(data[num_samples_start:num_samples_end,:],
                            dtype=dtype).to(device)
    data = torch.empty((num_samples,6+num_features), dtype=dtype).to(device)
    data[:,6:] = data_org


    # Magnitudes vel, acc, mass
    vel = torch.norm(data_org[:,4:7], p=2, dim=1)
    acc = torch.empty((num_samples,4), dtype=dtype).to(device)
    acc[:,0] = torch.norm(data_org[:,7:10], p=2, dim=1)
    acc[:,1] = torch.norm(data_org[:,10:13], p=2, dim=1)
    acc[:,2] = torch.norm(data_org[:,13:16], p=2, dim=1)
    acc[:,3] = torch.norm(data_org[:,16:19], p=2, dim=1)
    mass = data_org[:,0]

    data[:,0] = vel
    data[:,1:5] = acc
    data[:,5] = mass

    # Normalize the data
    data_min = data[:,:6].min(axis=0, keepdim=True).values
    data_max = data[:,:6].max(axis=0, keepdim=True).values
    data[:,:6] = (data[:,:6] - data_min) / (data_max - data_min)


    # Wrap the feature tensor in a TensorDataset
    # Each item from the dataset will be a tuple containing one tensor (X[i],)
    dataset = TensorDataset(data)

    # Define the proportion for the test set (e.g., 20%)
    test_ratio = 0.2
    num_test = int(num_samples * test_ratio)
    num_train = num_samples - num_test

    # Use random_split to split the dataset into train and test subsets
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test])

    # Create DataLoaders for the training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)

    print(data.shape)
    print(data)

    """
    with open(base_path+"c++/normalization_factors.txt", "w") as f:
        f.write(f"{data_min[0][0]} {data_min[0][1]} {data_min[0][2]} {data_min[0][3]} {data_min[0][4]} {data_min[0][5]}\n")    
        f.write(f"{data_max[0][0]} {data_max[0][1]} {data_max[0][2]} {data_max[0][3]} {data_max[0][4]} {data_max[0][5]}\n")    
        """

    return train_loader, test_loader, data_min, data_max


class Optimization():
    # Objective function for Optuna
    big_number = 1e5

    def __init__(self, study_name, storage_name, load_if_exists=True,
                 directions="minimize"):

        self.study_name     = study_name
        self.storage_name   = storage_name
        self.load_if_exists = load_if_exists
        self.base_lr        = 1e-9 #base_lr
        self.directions     = directions



        sampler = optuna.samplers.TPESampler()
        self.study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = self.directions,
            sampler=sampler,
        )


    def objective(self, gpu_id, trial):
        device = torch.device("cuda:{}".format(gpu_id))
        print(device)

        train_loader, valid_loader, data_min, data_max = get_dataset(device)

        input_size = 6     # Number of input features
        input_mask = np.r_[0:6]
        hidden_dims = [32,64,64,32]     # Number of hidden neurons
        output_size = 2     # Number of output 

        model = FullyConnectedNN(input_dim=input_size,
                                 output_dim=output_size,
                                 hidden_dims=hidden_dims,
                                 activation='relu',
                                 dropout=0.1,
                                 output_positive=True).to(device)
        model.double()


        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        #optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        lr = trial.suggest_float("lr", 1e-9, 1e-2, log=True)
        lr_tensor = torch.tensor(lr, dtype=dtype)
        optimizer = getattr(optim, optimizer_name)(model.parameters(),
                                                   lr=lr_tensor)
        #tw = trial.suggest_float("tw", 1e-5, 1e-1, log=True)  # timestep weight
        #ew = trial.suggest_float("te", 1e-5, 1e-1, log=True)  # energy weight
        tw = 0.1; ew = 0.1;
        weights = {"time_step":tw, "energy_loss":ew}


        # Define the loss function (CrossEntropyLoss is common for classification tasks)
        criterion = CustomizableLoss3DM(nParticle=3, nAttribute=20,
                                        nBatch=BATCHSIZE,
                                        alpha=weights['time_step'],
                                        beta=weights['energy_loss'],
                                        gamma=weights['energy_loss'],
                                        TargetEnergyError=TargetEnergyError,
                                        data_min=data_min,
                                        data_max=data_max,
                                        device=device)


        for epoch in range(EPOCHS):
            train_loss = \
                train_one_epoch(model, optimizer, criterion, train_loader, input_mask, weights, device)

            valid_loss, energy_error, energy_error_std, energy_error_fiducial, energy_error_fiducial_std, energy_pred, energy_init, time_step, time_step_fiducial = \
                validate(model, criterion, valid_loader, input_mask, weights, device)

            """
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                """

        # Save the trained model locally
        model_filename = f"model_trial_{trial.number}.pt"
        model_path = os.path.join(artifact_base_path, model_filename)
        torch.save(model.state_dict(), model_path)

        # Upload the model file to the artifact store and record its artifact ID
        artifact_id = optuna.artifacts.upload_artifact(
            artifact_store=artifact_store,
            file_path=model_path,
            study_or_trial=trial,
        )
        trial.set_user_attr("artifact_id", artifact_id)

        return valid_loss, time_step


    def run(self, n_trials, n_gpus, total_jobs, job_id):
        print("Optimization Starts!")
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(
            study_name     = self.study_name,
            storage        = "sqlite:///"+self.storage_name,
            load_if_exists = self.load_if_exists,
            directions     = self.directions,
            sampler=sampler,
        )

        gpu_id = job_id % n_gpus

        study.optimize(partial(self.objective, gpu_id),
                       n_trials=n_trials, n_jobs=1)
                       #n_trials=n_trials, n_jobs=total_jobs)



# Main function to run the optimization
if __name__ == "__main__":

    n_trials=100; n_jobs=1; n_gpus=1;
    total_jobs = n_jobs * n_gpus;

    directions = ['minimize', 'maximize']

    study_name = "three_body_preliminary"
    print(study_name)

    storage_name = "./database/{}.db".format(study_name)
    print(storage_name)

    opt = Optimization(study_name, storage_name, directions=directions)
    print("Optimization Prepared!")


    if total_jobs == 1:
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(
            study_name     = study_name,
            storage        = "sqlite:///"+storage_name,
            load_if_exists = True,
            directions     = directions,
            sampler=sampler,
        )
        study.optimize(partial(opt.objective, 0),
                       n_trials=n_trials, n_jobs=total_jobs)
    else:
         #r = Parallel(n_jobs=n_gpu)([delayed(opt.run)(n_trials, job_id=job_id, n_jobs=n_jobs)\
                 #                            for job_id in range(n_gpu)])
         r = Parallel(n_jobs=total_jobs)(
             [delayed(opt.run)(n_trials, n_gpus, total_jobs, job_id) for job_id in
              range(total_jobs)]
         )


    pruned_trials = [t for t in opt.study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in opt.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("Best trial:")
    trial = opt.study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in opt.trial.params.items():
        print(f"    {key}: {value}")





