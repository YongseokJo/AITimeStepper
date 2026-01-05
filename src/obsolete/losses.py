
# Customizable loss function combining MSE and MAE
class CustomizableLoss2D(nn.Module):
    Dim = 2
    def __init__(self, nParticle=2, nAttribute=13, nBatch=32, alpha=0.1, beta=0.1, gamma=0.10, TargetEnergyError=1e-8, 
                data_min=None, data_max=None, device='cpu'):
        """
        mse_weight: weight for the mean squared error component
        mae_weight: weight for the mean absolute error component
        """
        super(CustomizableLoss3D, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.TargetEnergyError = TargetEnergyError
        self.device=device
        self.data_min = data_min
        self.data_m2m = data_max-data_min
        self.nAttribute = nAttribute
        self.nParticle = nParticle
        self.nBatch = nBatch
        self.particle = torch.empty((nBatch, nParticle, nAttribute), requires_grad=True).to(device)

    def forward(self, output, data):
        self.nBatch = data.shape[0]
        energy_init = data[:self.nBatch,-2].clone()
        for i in range(self.nParticle):
            self.particle[:self.nBatch,i,:] = data[:,self.nAttribute*i:self.nAttribute*(i+1)]*self.data_m2m[0,self.nAttribute*i:self.nAttribute*(i+1)]+self.data_min[0,self.nAttribute*i:self.nAttribute*(i+1)]
        energy_init2 = self.compute_energy()
        print("init0=",energy_init,"\ninit1=",energy_init2)
        print("diff=",energy_init-energy_init2)
        self.predict_positions(output[:,0].clone())
        energy_pred = self.compute_energy()
        print("pred=",energy_pred, "\ndiff=",energy_pred-energy_init)
        print("pred-init=",energy_pred-energy_init)
        energy_error = torch.abs((energy_pred - energy_init)/energy_init)
        energy_loss = torch.mean((energy_error - self.TargetEnergyError)**2)
        #energy_diff = torch.mean(((energy_pred - energy_init)/energy_init)**2)
        #energy_loss = torch.mean(energy_diff - self.TargetEnergyError**2)
        loss = -self.alpha*torch.log(torch.mean(output[:,0]**2))+ self.gamma*torch.log(energy_loss)  #+ self.beta*energy_loss 
        return loss, {"energy_error": torch.mean(energy_error)}

        

    def predict_positions(self, dt):
        #print(data.shape)
        #print(self.particle1.shape)
        self.new_particle = self.particle.clone()
        for i in range(self.nParticle):
            for dim in range(Dim):
            #print(dt)
                self.new_particle[:self.nBatch,i,dim] = \
                    ((self.particle[:self.nBatch,i,Dim*3+dim]*dt/3 + self.particle[:self.nBatch,i,Dim*2+dim])*dt/2 + self.particle[:self.nBatch,i,Dim+dim])*dt\
                        + self.particle[:self.nBatch,i,dim]
                self.new_particle[:self.nBatch,i,Dim+dim] =\
                    (self.particle[:self.nBatch,i,Dim*3+dim]*dt/2 + self.particle[:self.nBatch,i,Dim*2+dim])*dt\
                        + self.particle[:self.nBatch,i,Dim+dim]
            #self.particle2[:,dim] = ((self.particle[:,Dim*9+dim]*dt/3 + self.particle[:,Dim*8+dim])*dt/2 + self.particle[:,Dim*7+dim])*dt + self.particle[:,Dim*6+dim]
            #self.particle2[:,Dim+dim] =  (data[:,Dim*9+dim]*dt/2 + data[:,Dim*8+dim])*dt   + data[:,Dim*7+dim]
            #print(self.particle1)
            #print(self.particle2)
        #self.particle1[:,-1] = data[:,12]
        #self.particle2[:,-1] = data[:,25]
        #self.particle = self.new_particle

    def compute_energy(self): 
        #kinetic_energy   = 0.5 * (self.particle1[:,-1] * torch.norm(self.particle1[:,2:4], p=2, dim=1)**2+self.particle2[:,-1] * torch.norm(self.particle2[:,2:4], p=2, dim=1)**2)
        kinetic_energy   = 0.5 * torch.sum(self.new_particle[:self.nBatch,:,-1] * torch.norm(self.new_particle[:self.nBatch,:,Dim:Dim*2], p=2, dim=2)**2, dim=1)
        #print(torch.norm(self.particle[:self.nBatch,:,Dim:Dim*2], p=2, dim=2)**2)
        #print(self.particle[:self.nBatch,:,-1]*torch.norm(self.particle[:self.nBatch,:,Dim:Dim*2], p=2, dim=2)**2)
        potential_energy = torch.zeros((self.nBatch,), requires_grad=True).to(self.device)
        for i in range(self.nParticle):
            for j in range(i+1,self.nParticle):
                r = torch.norm(self.new_particle[:self.nBatch,i,:Dim] - self.new_particle[:self.nBatch,j,:Dim], p=2, dim=1)
                potential_energy -= self.new_particle[:self.nBatch,i,-1] * self.new_particle[:self.nBatch,j,-1] / r
        #print("kin=",kinetic_energy)
        #print("pot=",potential_energy)
        return kinetic_energy + potential_energy


    def compute_momentum(self): 
        momentum = torch.sum(self.new_particle[:self.nBatch,:,-1] * torch.norm(self.new_particle[:self.nBatch,:,Dim:Dim*2], p=2, dim=1))
        return momemtum ## (samples, dimensions)

class CustomizableLoss3D(nn.Module):
    Dim = 3
    def __init__(self, nParticle=3, nAttribute=20, nBatch=32, alpha=0.1, beta=0.1, gamma=0.10, TargetEnergyError=1e-8, 
                data_min=None, data_max=None, device='cpu'):
        """
        mse_weight: weight for the mean squared error component
        mae_weight: weight for the mean absolute error component
        """
        super(CustomizableLoss3D, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.TargetEnergyError = TargetEnergyError
        self.device=device
        self.data_min = data_min
        self.data_m2m = data_max-data_min
        self.nAttribute = nAttribute
        self.nParticle = nParticle
        self.nBatch = nBatch
        self.particle = torch.empty((nBatch, nParticle, nAttribute), requires_grad=True).to(device)

    def forward(self, output, data):
        self.nBatch = data.shape[0]
        energy_init = data[:self.nBatch,-1].clone()
        #print("init=",energy_init)
        for i in range(self.nParticle):
            self.particle[:self.nBatch,i,:] = data[:,self.nAttribute*i:self.nAttribute*(i+1)]*self.data_m2m[0,self.nAttribute*i:self.nAttribute*(i+1)]+self.data_min[0,self.nAttribute*i:self.nAttribute*(i+1)]
        #self.new_particle = self.particle.clone()
        #energy_init2 = self.compute_energy()
        #print("init0=",energy_init,"\ninit1=",energy_init2)
        #print("diff=",energy_init-energy_init2)
        self.predict_positions(output[:,0].clone())
        energy_pred = self.compute_energy()
        #print("pred=",energy_pred, "\ndiff=",energy_pred-energy_init)
        #print("pred-init=",energy_pred-energy_init)
        energy_error = torch.abs((energy_pred - energy_init)/energy_init)
        energy_loss = torch.mean((energy_error - self.TargetEnergyError)**2)
        #energy_loss2 = torch.mean(torch.log(torch.abs(energy_pred/energy_init)))
        energy_loss3 = torch.mean((energy_pred - energy_init)**2)
        #energy_diff = torch.mean(((energy_pred - energy_init)/energy_init)**2)
        #energy_loss = torch.mean(energy_diff - self.TargetEnergyError**2)
        loss = -self.alpha*torch.log(torch.mean(output[:,0]**2))+ self.beta*energy_loss3 + self.gamma*torch.log(energy_loss)  # + self.beta*energy_loss2 
        #print(energy_loss)
        return loss, {"energy_error": energy_error, "energy_pred": energy_pred, "energy_init": energy_init}

        

    def predict_positions(self, dt):
        #print(data.shape)
        #print(self.particle1.shape)
        self.new_particle = self.particle.clone()
        for i in range(self.nParticle):
            for dim in range(self.Dim):
            #print(dt)
                self.new_particle[:self.nBatch,i,dim] = \
                    ((self.particle[:self.nBatch,i,self.Dim*3+dim+1]*dt/3 + self.particle[:self.nBatch,i,self.Dim*2+dim+1])*dt/2 + self.particle[:self.nBatch,i,self.Dim+dim+1])*dt\
                        + self.particle[:self.nBatch,i,dim+1]
                self.new_particle[:self.nBatch,i,self.Dim+dim+1] =\
                    (self.particle[:self.nBatch,i,self.Dim*3+dim+1]*dt/2 + self.particle[:self.nBatch,i,self.Dim*2+dim+1])*dt\
                        + self.particle[:self.nBatch,i,self.Dim+dim+1]
            #self.particle2[:,dim] = ((self.particle[:,self.Dim*9+dim]*dt/3 + self.particle[:,self.Dim*8+dim])*dt/2 + self.particle[:,self.Dim*7+dim])*dt + self.particle[:,self.Dim*6+dim]
            #self.particle2[:,self.Dim+dim] =  (data[:,self.Dim*9+dim]*dt/2 + data[:,self.Dim*8+dim])*dt   + data[:,self.Dim*7+dim]
            #print(self.particle1)
            #print(self.particle2)
        #self.particle1[:,-1] = data[:,12]
        #self.particle2[:,-1] = data[:,25]
        #self.particle = self.new_particle

    def compute_energy(self): 
        #kinetic_energy   = 0.5 * (self.particle1[:,-1] * torch.norm(self.particle1[:,2:4], p=2, dim=1)**2+self.particle2[:,-1] * torch.norm(self.particle2[:,2:4], p=2, dim=1)**2)
        kinetic_energy   = 0.5 * torch.sum(self.new_particle[:self.nBatch,:,0] * torch.norm(self.new_particle[:self.nBatch,:,self.Dim+1:self.Dim*2+1], p=2, dim=2)**2, dim=1)
        #print(torch.norm(self.particle[:self.nBatch,:,self.Dim:self.Dim*2], p=2, dim=2)**2)
        #print(self.particle[:self.nBatch,:,-1]*torch.norm(self.particle[:self.nBatch,:,self.Dim:self.Dim*2], p=2, dim=2)**2)
        potential_energy = torch.zeros((self.nBatch,), requires_grad=True).to(self.device)
        for i in range(self.nParticle):
            for j in range(i+1,self.nParticle):
                r = torch.norm(self.new_particle[:self.nBatch,i,1:self.Dim+1] - self.new_particle[:self.nBatch,j,1:self.Dim+1], p=2, dim=1)
                potential_energy -= self.new_particle[:self.nBatch,i,0] * self.new_particle[:self.nBatch,j,0] / r
        #print("kin=",kinetic_energy)
        #print("pot=",potential_energy)
        return kinetic_energy + potential_energy


    def compute_momentum(self): 
        momentum = torch.sum(self.new_particle[:self.nBatch,:,0] * torch.norm(self.new_particle[:self.nBatch,:,self.Dim+1:self.Dim*2+1], p=2, dim=1))
        return momemtum ## (samples, dimensions)


class CustomizableLoss3DM(nn.Module):
    Dim = 3
    def __init__(self, nParticle=3, nAttribute=20, nBatch=32, alpha=0.1, beta=0.1, gamma=0.10, TargetEnergyError=1e-8, 
                data_min=None, data_max=None, input_dim=6, device='cpu'):
        super(CustomizableLoss3DM, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.TargetEnergyError = TargetEnergyError
        self.device=device
        self.data_min = data_min
        self.data_m2m = data_max-data_min
        self.nAttribute = nAttribute
        self.nParticle = nParticle
        self.nBatch = nBatch
        self.particle = torch.empty((nBatch, nParticle, nAttribute), requires_grad=True).to(device)
        self.EnergyErrorMax = 1e-8
        self.EnergyErrorMin = 1e-3
        self.input_dim = input_dim
        #self.new_particle = torch.empty((nBatch, nParticle, nAttribute), requires_grad=True).to(device)


    def forward(self, model, output, data, weights=None):
        if weights is not None:
            self.alpha = weights['time_step']
            self.beta  = weights['energy_loss']
            self.gamma = weights['energy_loss']
    
    
        self.nBatch = data.shape[0]
        #self.TargetEnergyError = data[:,-2]
        #energy_init = data[:self.nBatch,-1].clone()
        #print("dt=",torch.mean(output[:,0]))
        #print("init=",energy_init)
        for i in range(self.nParticle):
            self.particle[:self.nBatch,i,:] = data[:,self.input_dim+self.nAttribute*i:self.input_dim+self.nAttribute*(i+1)]

        self.new_particle = self.particle.clone()
        energy_init = self.compute_total_energy()
        #print("init=",energy_init)
        #print("init0=",energy_init,"\ninit1=",energy_init2)
        #print("diff=",energy_init-energy_init2)

        #print("data=",data[:,1:5])
        #print("acc=",acc)
        #print("dt_new=",dt)
        #print("dt_old=",data[:,25])
        #dt = torch.pow(10, output[:,0])*dt
        #dt = torch.pow(10, output[:,0])
        #print("dt_new=",dt)
        self.predict_positions(torch.pow(10, output[:self.nBatch,0]))

        energy_pred = self.compute_total_energy()

        energy_error = torch.abs(energy_pred/energy_init-1)
        #print("energy_error=",energy_error)
        nan_check = torch.isnan(energy_error)
        if nan_check.any():
            print("NaN Found")
            print("dt=",output[:self.nBatch,0])
            print("energy_error=",energy_error)
            raise

        zero_check = energy_error==0
        if zero_check.any():
            zero_check = torch.where(energy_error==0)
            print("zero Found", zero_check)
            print("dt=",output[zero_check,0])
            print("energy_error=",energy_error[zero_check])
            print("energy_init=",energy_init[zero_check])
            print("energy_pred=",energy_pred[zero_check])
            energy_error = torch.clamp(energy_error, min=eps)

        """
            Regions:
                1. delta1 <= |error| < delta2 : loss = 0
                2. otherwise                  : loss = (|error|-delta1)(|error|-delta2)
        """

        if False:
            energy_loss = (torch.log(energy_error+eps)-np.log(self.EnergyErrorMax))*(torch.log(energy_error+eps)-np.log(self.EnergyErrorMin))
            #energy_loss = torch.where(energy_loss<=0, eps, energy_loss)
            energy_loss = torch.where(energy_loss<=0, 0, energy_loss)
            #print(energy_loss)
            #print(torch.log(energy_loss))
            #energy_loss = torch.sum(torch.log(energy_loss))
        else:
            energy_loss = (torch.log(energy_error+eps)-np.log(self.EnergyErrorMin))**2
            #energy_loss = (energy_error-self.EnergyErrorMin)**2
        #print(energy_loss)
        energy_loss = torch.mean(energy_loss)
        #print(energy_loss)



        if False:
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.sum(param ** 2)
            loss = -self.alpha*torch.mean(torch.log(output[:,0]+eps)) + energy_loss + l2_loss #)/(self.alpha+self.beta+self.gamma)
        else:
            loss = -self.alpha*torch.mean(output[:,0]) + energy_loss #+ l2_loss #)/(self.alpha+self.beta+self.gamma)
            #loss = energy_loss #+ l2_loss #)/(self.alpha+self.beta+self.gamma)

        return loss, {"energy_error": energy_error, "energy_pred": energy_pred, "energy_init": energy_init}





        

    def predict_positions(self, dt):
        """
        Predict new particle positions and velocities given a time step dt.
        
        Args:
            dt (torch.Tensor): Tensor of shape (nBatch, nParticle) representing the time step for each batch and particle.
        
        Assumes self.particle is structured as:
        - [:, :, 1:Dim+1] -> Position
        - [:, :, Dim+1:2*Dim+1] -> Velocity
        - [:, :, 2*Dim+1:3*Dim+1] -> Acceleration
        - [:, :, 3*Dim+1:4*Dim+1] -> Jerk
        """
        # Clone the original particle tensor to update it
        self.new_particle = self.particle.clone()
        
        # Expand dt for broadcasting: from shape [nBatch, nParticle] to [nBatch, nParticle, 1]
        dt_exp = dt.view(dt.shape[0], 1, 1)
        
        # Extract slices (using 1-indexed positions as per your convention)
        pos  = self.particle[:self.nBatch, :, 1:self.Dim+1]
        vel  = self.particle[:self.nBatch, :, self.Dim+1:2*self.Dim+1]
        acc  = self.particle[:self.nBatch, :, 2*self.Dim+1:3*self.Dim+1]
        jerk = self.particle[:self.nBatch, :, 3*self.Dim+1:4*self.Dim+1]
        
        # Compute the new positions and velocities using vectorized operations
        new_pos = (((jerk * (dt_exp / 3)) + acc) * (dt_exp / 2) + vel) * dt_exp + pos
        new_vel = ((jerk * (dt_exp / 2)) + acc) * dt_exp + vel

        # Update the corresponding slices in the new particle tensor
        self.new_particle[:self.nBatch, :, 1:self.Dim+1] = new_pos
        self.new_particle[:self.nBatch, :, self.Dim+1:2*self.Dim+1] = new_vel



    def compute_total_energy(self, G=1.0, eps=1e-20):
        """
        Compute the total energy (kinetic + gravitational potential) for a batch of particles.

        Assumptions for self.new_particle tensor shape [nBatch, nParticle, 4*Dim+?]:
            - Index 0 is the mass.
            - Indices 1:Dim+1 are positions.
            - Indices Dim+1:2*Dim+1 are velocities.
        
        Args:
            G (float): Gravitational constant.
            eps (float): A small constant to avoid division by zero.
        
        Returns:
            total_energy (torch.Tensor): Tensor of shape [nBatch] with the total energy for each batch.
        """
        # Get device and dimensions
        nParticle = self.new_particle.shape[1]

        # -----------------------------
        # 1. Compute Kinetic Energy
        # -----------------------------
        # mass: shape [nBatch, nParticle]
        mass = self.new_particle[:self.nBatch, :, 0]
        # velocity: shape [nBatch, nParticle, Dim]
        vel = self.new_particle[:self.nBatch, :, self.Dim+1:2*self.Dim+1]
        # Compute squared speed without an explicit torch.norm for speed:
        v2 = (vel ** 2).sum(dim=2)  # shape [nBatch, nParticle]
        # Kinetic energy for each batch: sum(0.5 * mass * speed^2) over particles
        kinetic_energy = 0.5 * (mass * v2).sum(dim=1)  # shape [nBatch]

        # -----------------------------
        # 2. Compute Gravitational Potential Energy
        # -----------------------------
        # Positions: shape [nBatch, nParticle, Dim]
        pos = self.new_particle[:self.nBatch, :, 1:self.Dim+1]
        # Compute squared norms of positions for each particle (batched):
        x2 = (pos ** 2).sum(dim=2, keepdim=True)  # shape [nBatch, nParticle, 1]
        # Compute batched pairwise squared distances:
        # Using broadcasting: dist2 = ||x_i||^2 + ||x_j||^2 - 2 * (x_i dot x_j)
        dist2 = x2 + x2.transpose(1, 2) - 2 * torch.bmm(pos, pos.transpose(1, 2))  # shape [nBatch, nParticle, nParticle]
        dist2 = torch.clamp(dist2, min=eps)  # Avoid negatives or division by zero
        dist = torch.sqrt(dist2)  # shape [nBatch, nParticle, nParticle]

        # Compute the outer product of masses for each batch
        # mass: [nBatch, nParticle] -> unsqueeze to [nBatch, nParticle, 1] and [nBatch, 1, nParticle]
        mass_outer = mass.unsqueeze(2) * mass.unsqueeze(1)  # shape [nBatch, nParticle, nParticle]

        # Compute gravitational potential energy matrix:
        # U_ij = -G * (m_i * m_j) / r_ij for each pair (i, j)
        potential_matrix = -G * mass_outer / dist  # shape [nBatch, nParticle, nParticle]

        # Remove self-interactions by zeroing the diagonal for each batch:
        mask = torch.eye(nParticle, device=self.device, dtype=torch.bool).unsqueeze(0)  # shape [1, nParticle, nParticle]
        potential_matrix = potential_matrix.masked_fill(mask, 0.0)

        # Since the interaction matrix is symmetric, sum over all elements and divide by 2:
        potential_energy = 0.5 * potential_matrix.sum(dim=(1, 2))  # shape [nBatch]

        # -----------------------------
        # 3. Total Energy
        # -----------------------------
        total_energy = kinetic_energy + potential_energy  # shape [nBatch]
        #print("kin=",kinetic_energy)
        #print("pot=",potential_energy)
        return total_energy



    def compute_momentum(self): 
        momentum = torch.sum(self.new_particle[:self.nBatch,:,0] * torch.norm(self.new_particle[:self.nBatch,:,self.Dim+1:self.Dim*2+1], p=2, dim=1))
        return momemtum ## (samples, dimensions)



    def forward_old(self, model, output, data, weights=None):
        if weights is not None:
            self.alpha = weights['time_step']
            self.beta  = weights['energy_loss']
            self.gamma = weights['energy_loss']
    
    
        self.nBatch = data.shape[0]
        self.TargetEnergyError = data[:,-2]
        #energy_init = data[:self.nBatch,-1].clone()
        #print("dt=",torch.mean(output[:,0]))
        #print("init=",energy_init)
        for i in range(self.nParticle):
            self.particle[:self.nBatch,i,:] = data[:,6+self.nAttribute*i:6+self.nAttribute*(i+1)]

        self.new_particle = self.particle.clone()
        energy_init = self.compute_total_energy()
        #print("init=",energy_init)
        #print("init0=",energy_init,"\ninit1=",energy_init2)
        #print("diff=",energy_init-energy_init2)

        self.predict_positions(output[:self.nBatch,0])

        energy_pred = self.compute_total_energy()
        #print("pred=",energy_pred, "\ndiff=",energy_pred-energy_init)
        #print("pred-init=",energy_pred-energy_init)
        
        #energy_init_log = torch.log(np.abs(energy_init))
        #energy_pred_log = torch.log(np.abs(energy_pred))

        energy_error = torch.abs(energy_pred/energy_init-1)
        #energy_error = torch.clamp(energy_error, min=eps)
        #energy_error[energy_error<self.TargetEnergyError] =  self.TargetEnergyError
        #energy_error = torch.clamp_min(energy_error, self.TargetEnergyError)
        #print(energy_error)
        #energy_error = torch.clamp(energy_error, min=self.TargetEnergyError)
        #energy_loss = torch.sum(torch.log10(energy_error))
        #print(energy_error)
        #print(output[:,0], energy_error)
        #print(torch.mean(energy_error))

        """
            Regions:
                1. delta1 <= |error| < delta2 : loss = 0
                2. otherwise                  : loss = (|error|-delta1)(|error|-delta2)
        """
        
        energy_loss = (torch.log(energy_error+eps)-np.log(self.EnergyErrorMax))*(torch.log(energy_error+eps)-np.log(self.EnergyErrorMin))
        #energy_loss = torch.where(energy_loss<=0, eps, energy_loss)
        energy_loss = torch.where(energy_loss<=0, 0, energy_loss)
        #print(energy_loss)
        #print(torch.log(energy_loss))
        #energy_loss = torch.sum(torch.log(energy_loss))
        energy_loss = torch.mean(energy_loss)
        #print(energy_loss)

        #energy_loss10 = torch.mean((output[:,1] - torch.log10(energy_error))**2)

        #print(self.TargetEnergyError)
        #energy_loss = torch.abs(energy_error - self.TargetEnergyError)
        #energy_loss = torch.where(energy_error > self.TargetEnergyError, torch.abs(energy_error - self.TargetEnergyError), (energy_error - self.TargetEnergyError) ** 4)
        #energy_loss = torch.log(torch.mean(energy_loss))
        #energy_loss = torch.mean(torch.log(energy_loss+eps))
        #print(energy_loss)

        #energy_loss4 = torch.mean((torch.log(energy_error+eps) - np.log(self.TargetEnergyError))**2)
        #energy_loss4 = torch.mean((torch.log(energy_error+eps) - torch.log(self.TargetEnergyError))**2)
        #print(energy_loss4)


        #energy_loss2 = torch.mean(torch.log(torch.abs(energy_pred/energy_init))**2)
        #energy_loss2 = torch.mean(torch.log(torch.abs(energy_init/energy_pred))+torch.log(torch.abs(energy_pred/energy_init)))
        #energy_loss3 = torch.mean((energy_pred - energy_init)**2)
        #energy_diff = torch.mean(((energy_pred - energy_init)/energy_init)**2)
        #energy_loss = torch.mean(energy_diff - self.TargetEnergyError**2)
        #loss = -self.alpha*torch.log(torch.mean(output[:,0]**2))+ self.beta*energy_loss3 + self.gamma*torch.log(energy_loss) + self.beta*energy_loss2 
        #loss = self.beta*torch.log(energy_loss3+double_tiny) + self.beta*energy_loss2 
        #loss = -self.alpha*torch.mean(torch.log(output[:,0]+double_tiny)**2) + self.beta*torch.log(energy_loss3) + self.beta*energy_loss2 
        #loss = -self.alpha*torch.mean(output[:,0]**2) + self.beta*torch.log(energy_loss3+double_tiny) + self.beta*energy_loss2   + self.gamma*torch.log(energy_loss)
        #loss = -self.alpha*torch.mean(output[:,0]**2) + self.beta*energy_loss2   + self.gamma*torch.log(energy_loss)
        #loss = -self.alpha*torch.mean(output[:,0]**2)  + self.gamma*torch.log(energy_loss)
        #loss = (-self.alpha*torch.log(torch.mean(output[:,0]**2))  + self.gamma*torch.log(energy_loss)) #+ self.gamma*energy_loss4)/(self.alpha+self.beta+self.gamma)
        #loss = -self.alpha*torch.log(torch.mean(output[:,0]**2)) + torch.log(energy_loss) #+ self.gamma*energy_loss4 #)/(self.alpha+self.beta+self.gamma)
        #loss = torch.log(energy_loss) #+ self.gamma*energy_loss4 #)/(self.alpha+self.beta+self.gamma)
        #loss = energy_loss #+ self.gamma*energy_loss4 #)/(self.alpha+self.beta+self.gamma)
        #loss = energy_loss + energy_loss4 +energy_loss10 #)/(self.alpha+self.beta+self.gamma)
        #loss = -self.alpha*torch.log(torch.mean(output[:,0]**2)) + energy_loss + energy_loss4 +energy_loss10 #)/(self.alpha+self.beta+self.gamma)

        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.sum(param ** 2)

        #loss =  energy_loss + energy_loss4 + energy_loss10 #)/(self.alpha+self.beta+self.gamma)
        #loss = -self.alpha*torch.log(torch.mean(output[:,0]**2)) + energy_loss + energy_loss4 + energy_loss10 + l2_loss #)/(self.alpha+self.beta+self.gamma)
        #loss = -self.alpha*torch.log(torch.mean(output[:,0]**2)) + energy_loss + energy_loss10 + l2_loss #)/(self.alpha+self.beta+self.gamma)
        #loss = -self.alpha*torch.log(torch.mean(output[:,0]**2)) + energy_loss + l2_loss #)/(self.alpha+self.beta+self.gamma)
        loss = -self.alpha*torch.mean(torch.log(output[:,0]+eps)) + energy_loss + l2_loss #)/(self.alpha+self.beta+self.gamma)

        #loss = energy_loss #+ energy_loss10 #)/(self.alpha+self.beta+self.gamma)
        #print(output[:,0])
        #print(energy_error,energy_loss2, energy_loss3)
        #print(energy_loss)
        # Compute L2 regularization (squared L2 norm of all parameters)

        return loss, {"energy_error": energy_error, "energy_pred": energy_pred, "energy_init": energy_init}





    def forward_old_old(self, output, data):
        self.nBatch = data.shape[0]
        energy_init = data[:self.nBatch,-1].clone()
        #print("init=",energy_init)
        for i in range(self.nParticle):
            self.particle[:self.nBatch,i,:] = data[:,6+self.nAttribute*i:6+self.nAttribute*(i+1)]
        #self.new_particle = self.particle.clone()
        #energy_init2 = self.compute_energy()
        #print("init0=",energy_init,"\ninit1=",energy_init2)
        #print("diff=",energy_init-energy_init2)
        self.predict_positions(output[:self.nBatch,0])
        energy_pred = self.compute_energy()
        #print("pred=",energy_pred, "\ndiff=",energy_pred-energy_init)
        #print("pred-init=",energy_pred-energy_init)
        


        energy_error = torch.abs((energy_pred - energy_init)/energy_init)


        energy_loss = torch.mean((energy_error - self.TargetEnergyError)**2)
        #energy_loss4 = torch.mean((torch.log(energy_error) - np.log(self.TargetEnergyError))**2)
        print(energy_loss4)


        energy_loss2 = torch.mean(torch.log(torch.abs(energy_pred/energy_init))**2)
        #energy_loss2 = torch.mean(torch.log(torch.abs(energy_init/energy_pred))+torch.log(torch.abs(energy_pred/energy_init)))
        energy_loss3 = torch.mean((energy_pred - energy_init)**2)
        #energy_diff = torch.mean(((energy_pred - energy_init)/energy_init)**2)
        #energy_loss = torch.mean(energy_diff - self.TargetEnergyError**2)
        #loss = -self.alpha*torch.log(torch.mean(output[:,0]**2))+ self.beta*energy_loss3 + self.gamma*torch.log(energy_loss) + self.beta*energy_loss2 
        #loss = self.beta*torch.log(energy_loss3+double_tiny) + self.beta*energy_loss2 
        #loss = -self.alpha*torch.mean(torch.log(output[:,0]+double_tiny)**2) + self.beta*torch.log(energy_loss3) + self.beta*energy_loss2 
        #loss = -self.alpha*torch.mean(output[:,0]**2) + self.beta*torch.log(energy_loss3+double_tiny) + self.beta*energy_loss2   + self.gamma*torch.log(energy_loss)
        #loss = -self.alpha*torch.mean(output[:,0]**2) + self.beta*energy_loss2   + self.gamma*torch.log(energy_loss)
        #loss = -self.alpha*torch.mean(output[:,0]**2)  + self.gamma*torch.log(energy_loss)
        loss = -self.alpha*torch.log(torch.mean(output[:,0]**2))  + self.gamma*torch.log(energy_loss) #+ self.gamma*energy_loss4
        #print(output[:,0])
        #print(energy_error,energy_loss2, energy_loss3)
        return loss, {"energy_error": energy_error, "energy_pred": energy_pred, "energy_init": energy_init}

        
        
class CustomizableLoss3DMAdjusted(nn.Module):
    Dim = 3
    def __init__(self, nParticle=3, nAttribute=20, nBatch=32, alpha=0.1, beta=0.1, gamma=0.10, TargetEnergyError=1e-8, 
                data_min=None, data_max=None, device='cpu'):
        super(CustomizableLoss3DMAdjusted, self).__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.TargetEnergyError = TargetEnergyError
        self.device=device
        self.data_min = data_min
        self.data_m2m = data_max-data_min
        self.nAttribute = nAttribute
        self.nParticle = nParticle
        self.nBatch = nBatch
        self.particle = torch.empty((nBatch, nParticle, nAttribute), requires_grad=True).to(device)
        self.EnergyErrorMax = 1e-9
        self.EnergyErrorMin = 1e-9
        #self.new_particle = torch.empty((nBatch, nParticle, nAttribute), requires_grad=True).to(device)


    def forward(self, model, output, data, weights=None):
        if weights is not None:
            self.alpha = weights['time_step']
            self.beta  = weights['energy_loss']
            self.gamma = weights['energy_loss']
    
    
        self.nBatch = data.shape[0]
        #self.TargetEnergyError = data[:,-2]
        #energy_init = data[:self.nBatch,-1].clone()
        #print("dt=",torch.mean(output[:,0]))
        #print("init=",energy_init)
        for i in range(self.nParticle):
            self.particle[:self.nBatch,i,:] = data[:,6+self.nAttribute*i:6+self.nAttribute*(i+1)]

        self.new_particle = self.particle.clone()
        energy_init = self.compute_total_energy()
        #print("init=",energy_init)
        #print("init0=",energy_init,"\ninit1=",energy_init2)
        #print("diff=",energy_init-energy_init2)

        #print("data=",data[:,1:5])
        acc = data[:,1:5]*self.data_m2m[:,1:5] + self.data_min[:,1:5]
        #print("acc=",acc)
        dt = torch.sqrt((acc[:,0]*acc[:,2]+acc[:,1]**2)/(acc[:,1]*acc[:,3]+acc[:,2]**2))
        #print("dt_new=",dt)
        #print("dt_old=",data[:,25])
        #dt = torch.pow(10, output[:,0])*dt
        dt = output[:,0]*dt
        #print("dt_new=",dt)
        self.predict_positions(dt)

        energy_pred = self.compute_total_energy()

        energy_error = torch.abs(energy_pred/energy_init-1)
        #print("energy_error=",energy_error)


        """
            Regions:
                1. delta1 <= |error| < delta2 : loss = 0
                2. otherwise                  : loss = (|error|-delta1)(|error|-delta2)
        """

        if False:
            energy_loss = (torch.log(energy_error+eps)-np.log(self.EnergyErrorMax))*(torch.log(energy_error+eps)-np.log(self.EnergyErrorMin))
            #energy_loss = torch.where(energy_loss<=0, eps, energy_loss)
            energy_loss = torch.where(energy_loss<=0, 0, energy_loss)
            #print(energy_loss)
            #print(torch.log(energy_loss))
            #energy_loss = torch.sum(torch.log(energy_loss))
        else:
            #energy_loss = (torch.log(energy_error+eps)-np.log(self.EnergyErrorMin))**2
            energy_loss = (energy_error-self.EnergyErrorMin)**2
        print(energy_loss)
        energy_loss = torch.mean(energy_loss)
        print(energy_loss)



        if False:
            l2_loss = 0
            for param in model.parameters():
                l2_loss += torch.sum(param ** 2)
            loss = -self.alpha*torch.mean(torch.log(output[:,0]+eps)) + energy_loss + l2_loss #)/(self.alpha+self.beta+self.gamma)
        else:
            #loss = -self.alpha*torch.mean(torch.log(output[:,0]+eps)) + energy_loss #+ l2_loss #)/(self.alpha+self.beta+self.gamma)
            loss = energy_loss #+ l2_loss #)/(self.alpha+self.beta+self.gamma)

        return loss, {"energy_error": energy_error, "energy_pred": energy_pred, "energy_init": energy_init}




        

    def predict_positions(self, dt):
        """
        Predict new particle positions and velocities given a time step dt.
        
        Args:
            dt (torch.Tensor): Tensor of shape (nBatch, nParticle) representing the time step for each batch and particle.
        
        Assumes self.particle is structured as:
        - [:, :, 1:Dim+1] -> Position
        - [:, :, Dim+1:2*Dim+1] -> Velocity
        - [:, :, 2*Dim+1:3*Dim+1] -> Acceleration
        - [:, :, 3*Dim+1:4*Dim+1] -> Jerk
        """
        # Clone the original particle tensor to update it
        self.new_particle = self.particle.clone()
        
        # Expand dt for broadcasting: from shape [nBatch, nParticle] to [nBatch, nParticle, 1]
        dt_exp = dt.view(dt.shape[0], 1, 1)
        
        # Extract slices (using 1-indexed positions as per your convention)
        pos  = self.particle[:self.nBatch, :, 1:self.Dim+1]
        vel  = self.particle[:self.nBatch, :, self.Dim+1:2*self.Dim+1]
        acc  = self.particle[:self.nBatch, :, 2*self.Dim+1:3*self.Dim+1]
        jerk = self.particle[:self.nBatch, :, 3*self.Dim+1:4*self.Dim+1]
        
        # Compute the new positions and velocities using vectorized operations
        new_pos = (((jerk * (dt_exp / 3)) + acc) * (dt_exp / 2) + vel) * dt_exp + pos
        new_vel = ((jerk * (dt_exp / 2)) + acc) * dt_exp + vel

        # Update the corresponding slices in the new particle tensor
        self.new_particle[:self.nBatch, :, 1:self.Dim+1] = new_pos
        self.new_particle[:self.nBatch, :, self.Dim+1:2*self.Dim+1] = new_vel



    def compute_total_energy(self, G=1.0, eps=1e-20):
        """
        Compute the total energy (kinetic + gravitational potential) for a batch of particles.

        Assumptions for self.new_particle tensor shape [nBatch, nParticle, 4*Dim+?]:
            - Index 0 is the mass.
            - Indices 1:Dim+1 are positions.
            - Indices Dim+1:2*Dim+1 are velocities.
        
        Args:
            G (float): Gravitational constant.
            eps (float): A small constant to avoid division by zero.
        
        Returns:
            total_energy (torch.Tensor): Tensor of shape [nBatch] with the total energy for each batch.
        """
        # Get device and dimensions
        nParticle = self.new_particle.shape[1]

        # -----------------------------
        # 1. Compute Kinetic Energy
        # -----------------------------
        # mass: shape [nBatch, nParticle]
        mass = self.new_particle[:self.nBatch, :, 0]
        # velocity: shape [nBatch, nParticle, Dim]
        vel = self.new_particle[:self.nBatch, :, self.Dim+1:2*self.Dim+1]
        # Compute squared speed without an explicit torch.norm for speed:
        v2 = (vel ** 2).sum(dim=2)  # shape [nBatch, nParticle]
        # Kinetic energy for each batch: sum(0.5 * mass * speed^2) over particles
        kinetic_energy = 0.5 * (mass * v2).sum(dim=1)  # shape [nBatch]

        # -----------------------------
        # 2. Compute Gravitational Potential Energy
        # -----------------------------
        # Positions: shape [nBatch, nParticle, Dim]
        pos = self.new_particle[:self.nBatch, :, 1:self.Dim+1]
        # Compute squared norms of positions for each particle (batched):
        x2 = (pos ** 2).sum(dim=2, keepdim=True)  # shape [nBatch, nParticle, 1]
        # Compute batched pairwise squared distances:
        # Using broadcasting: dist2 = ||x_i||^2 + ||x_j||^2 - 2 * (x_i dot x_j)
        dist2 = x2 + x2.transpose(1, 2) - 2 * torch.bmm(pos, pos.transpose(1, 2))  # shape [nBatch, nParticle, nParticle]
        dist2 = torch.clamp(dist2, min=eps)  # Avoid negatives or division by zero
        dist = torch.sqrt(dist2)  # shape [nBatch, nParticle, nParticle]

        # Compute the outer product of masses for each batch
        # mass: [nBatch, nParticle] -> unsqueeze to [nBatch, nParticle, 1] and [nBatch, 1, nParticle]
        mass_outer = mass.unsqueeze(2) * mass.unsqueeze(1)  # shape [nBatch, nParticle, nParticle]

        # Compute gravitational potential energy matrix:
        # U_ij = -G * (m_i * m_j) / r_ij for each pair (i, j)
        potential_matrix = -G * mass_outer / dist  # shape [nBatch, nParticle, nParticle]

        # Remove self-interactions by zeroing the diagonal for each batch:
        mask = torch.eye(nParticle, device=self.device, dtype=torch.bool).unsqueeze(0)  # shape [1, nParticle, nParticle]
        potential_matrix = potential_matrix.masked_fill(mask, 0.0)

        # Since the interaction matrix is symmetric, sum over all elements and divide by 2:
        potential_energy = 0.5 * potential_matrix.sum(dim=(1, 2))  # shape [nBatch]

        # -----------------------------
        # 3. Total Energy
        # -----------------------------
        total_energy = kinetic_energy + potential_energy  # shape [nBatch]
        #print("kin=",kinetic_energy)
        #print("pot=",potential_energy)
        return total_energy


    def compute_momentum(self): 
        momentum = torch.sum(self.new_particle[:self.nBatch,:,0] * torch.norm(self.new_particle[:self.nBatch,:,self.Dim+1:self.Dim*2+1], p=2, dim=1))
        return momemtum ## (samples, dimensions)

