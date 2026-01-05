import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
#torch.set_default_dtype(torch.float64)
#torch.set_default_dtype(torch.float32)

# Training loop for one epoch
def train_one_epoch_deepspeed(model_engine, optimizer, criterion, train_loader, input_mask, weights=None, device='cuda'):

    model_engine.train()
    result={"train_loss":0., "energy_error":0., "energy_error_fiducial":0., "time_step":0., "time_step_fiducial":0.,
            "time_step_relative_error":0.,"time_step_MSE":0.}
    N = 0
    for batch_idx, (X,) in enumerate(train_loader):
        #optimizer.zero_grad()       # Clear gradients
        #print(X.shape)
        #print(X[:,:12].shape)
        output = model_engine(X[:,input_mask])          # Forward pass
        #print(output)
        loss, loss_terms = criterion(model_engine, output, X, weights)  # Compute loss between output and input
        #loss.backward(retain_graph=True)               # Backpropagation
        #loss.backward()               # Backpropagation
        #optimizer.step()              # Update parameters
        model_engine.backward(loss, retain_graph=True)
        model_engine.step()
        

        dt = torch.pow(10, output[:,0])
        result['train_loss']            += loss.item() * X.size(0)
        result['energy_error']          += torch.sum(torch.log10(loss_terms["energy_error"])).item()
        result['energy_error_fiducial'] += torch.sum(torch.log10(X[:,-2])).item()
        result['time_step']             += dt.sum().item() # convert to seconds
        result['time_step_fiducial']    += torch.sum(X[:,25]).item() ## for magnitude, 25 and for vectors, 19
        result['time_step_relative_error'] += torch.mean(torch.abs(dt-X[:,25])/X[:,25]).item()*X.size(0) ## for magnitude, 25 and for vectors, 19
        result['time_step_MSE'] += torch.mean((dt-X[:,25])**2).item()*X.size(0) ## for magnitude, 25 and for vectors, 19
        N += X.shape[0]


    result['train_loss']            /= N
    result['energy_error']          /= N
    result['energy_error_fiducial'] /= N
    result['time_step']             /= N
    result['time_step_fiducial']    /= N
    result['time_step_relative_error'] /= N
    result['time_step_MSE'] /= N
    return result

def validate_deepspeed(model_engine, criterion, val_loader, input_mask, weights=None, device='cuda'):
    # Evaluate on the test set
    model_engine.eval()

    result={"val_loss":0., "energy_error":0., "energy_error_fiducial":0.,
            "time_step":0., "time_step_fiducial":0., "time_step_std":0., "time_step_fiducial_std":0.,
            "time_step_relative_error":0,"time_step_MSE":0.,}
    N = 0

    energy_error_fiducial = 0.0
    #energy_error_std = 0.0
    #energy_error_fiducial_std = 0.0
    with torch.no_grad():
        for batch_idx, (X,) in enumerate(val_loader):
            output = model_engine(X[:,input_mask])
            loss, loss_terms = criterion(model_engine, output, X, weights)

            #print(output[:,0]) 
            dt = torch.pow(10, output[:,0])
            result['val_loss']              += loss.item() * X.size(0)
            result['energy_error']          += torch.sum(torch.log10(loss_terms["energy_error"])).item()
            result['energy_error_fiducial'] += torch.sum(torch.log10(X[:,-2])).item()
            result['time_step']             += dt.sum().item() # convert to seconds
            result['time_step_fiducial']    += torch.sum(X[:,25]).item() ## for magnitude, 25 and for vectors, 19
            result['time_step_std']          += torch.std(dt).item()*X.size(0) # convert to seconds
            result['time_step_fiducial_std'] += torch.std(X[:,25]).item()*X.size(0) ## for magnitude, 25 and for vectors, 19
            result['time_step_relative_error'] += torch.mean(torch.abs(dt-X[:,25])/X[:,25]).item()*X.size(0) ## for magnitude, 25 and for vectors, 19
            result['time_step_MSE'] += torch.mean((dt-X[:,25])**2).item()*X.size(0) ## for magnitude, 25 and for vectors, 19
            N += X.shape[0]

            #test_loss += loss.item() * X.size(0)

            #energy_error_std +=  torch.norm(loss_terms["energy_error"], p=2).item()
            #energy_init += torch.mean(loss_terms["energy_init"]).item()
            #energy_pred += torch.mean(loss_terms["energy_pred"]).item()
            #energy_error_fiducial_std +=  torch.norm(X[:,-2],p=2).item()
            #energy_loss += output[:,1].mean().item()
            #print(output[:,0])

    result['val_loss']              /= N
    result['energy_error']          /= N
    result['energy_error_fiducial'] /= N
    result['time_step']             /= N
    result['time_step_fiducial']    /= N
    result['time_step_std']          /= N
    result['time_step_fiducial_std'] /= N
    result['time_step_relative_error'] /= N
    result['time_step_MSE'] /= N

    return result


# Training loop for one epoch
def train_one_epoch(model, optimizer, criterion, train_loader, input_mask, weights=None, device='cuda'):

    model.train()
    train_loss = 0.0
    energy_loss = 0.0
    for batch_idx, (X,) in enumerate(train_loader):
        optimizer.zero_grad()       # Clear gradients
        #print(X.shape)
        #print(X[:,:12].shape)
        output = model(X[:,input_mask])          # Forward pass
        #print(output)
        loss, loss_terms = criterion(output, X, weights)  # Compute loss between output and input
        loss.backward(retain_graph=True)               # Backpropagation
        #loss.backward()               # Backpropagation
        optimizer.step()              # Update parameters
        

        energy_loss += torch.mean(loss_terms["energy_error"]).item()
        train_loss += loss.item() * X.size(0)
    
    energy_loss /= len(train_loader.dataset)
    train_loss /= len(train_loader.dataset)
    return train_loss, energy_loss

def validate(model, criterion, val_loader, input_mask, weights=None, device='cuda'):
    # Evaluate on the test set
    model.eval()
    test_loss = 0.0
    energy_error = 0.0
    energy_error_fiducial = 0.0
    energy_error_std = 0.0
    energy_error_fiducial_std = 0.0
    energy_pred = 0.
    energy_init = 0.
    time_step = 0.0
    time_step_fiducial = 0.0
    num_test = len(val_loader.dataset)
    with torch.no_grad():
        for batch_idx, (X,) in enumerate(val_loader):
            output = model(X[:,input_mask])
            loss, loss_terms = criterion(output, X, weights)
            test_loss += loss.item() * X.size(0)

            energy_error += torch.mean(loss_terms["energy_error"]).item()
            energy_error_std +=  torch.norm(loss_terms["energy_error"], p=2).item()
            energy_init += torch.mean(loss_terms["energy_init"]).item()
            energy_pred += torch.mean(loss_terms["energy_pred"]).item()
            energy_error_fiducial += torch.mean(X[:,-2]).item()
            energy_error_fiducial_std +=  torch.norm(X[:,-2],p=2).item()

            #energy_loss += output[:,1].mean().item()
            time_step += output[:,0].mean().item() # convert to seconds
            time_step_fiducial = torch.mean(X[:,25]).item() ## for magnitude, 25 and for vectors, 19
            #print(output[:,0])
    test_loss /= num_test
    energy_error /= num_test
    energy_error_std /= num_test
    energy_error_fiducial /= num_test
    energy_error_fiducial_std /= num_test
    energy_pred /= num_test
    energy_init /= num_test
    time_step /= num_test
    time_step_fiducial /= num_test

    return test_loss, energy_error, energy_error_std, energy_error_fiducial, energy_error_fiducial_std, energy_pred, energy_init, time_step, time_step_fiducial
    
    #print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4e}, Test Loss: {test_loss:.4e}, Energy Loss: {energy_error:.4e}/{energy_error_fiducial:.4e}, Time step: {time_step:.4e}/{time_step_fiducial:.4e}")




def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    loss,
    info: dict | None = None,
    extra: dict | None = None,
):
    """
    Save model/optimizer state + some training diagnostics.

    path   : str or Path to the .pt file
    epoch  : current epoch (int)
    model  : nn.Module
    optimizer : torch.optim.Optimizer
    loss   : scalar tensor
    info   : dict of tensors (e.g. your info dict)
    extra  : any extra metadata you want (hyperparams, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert info tensors -> Python scalars for convenience
    if info is not None:
        info_out = {}
        for k, v in info.items():
            if torch.is_tensor(v):
                # detach just in case
                info_out[k] = v.detach().cpu().item()
            else:
                info_out[k] = v
    else:
        info_out = None

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss.detach().cpu().item(),
        "info": info_out,
        "extra": extra,
    }

    torch.save(state, path)