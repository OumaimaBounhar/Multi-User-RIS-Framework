import numpy as np 
import torch 

# Learn function for DQN
def dqn_learn_update_step(
        eval_net,
        target_net,
        optimizer,
        current_state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
        done_batch,
        gamma,
        do_gradient_clipping,
        max_norm,
):
        """ 
        One DQN update step. This function doesn't update the target network, it only computes the Bellman equation to calculate the loss function. 
        
        Works on CPU or GPU depending on where eval_net lives.
        """

        # Infer device from model (CPU or GPU)
        device = next(eval_net.parameters()).device # Tensors and models must live on the same device.
        
        # PyTorch rule: Any tensor participating in gradient-based math must be floating-point

        # Convert batches to tensors on correct device
        current_state_tensor = torch.as_tensor(current_state_batch, dtype= torch.float32, device= device)

        next_state_tensor = torch.as_tensor(next_state_batch, dtype= torch.float32, device= device)

        actions_tensor = torch.as_tensor(action_batch, dtype= torch.long, device= device).view(-1) #long for integers in PyTorch, dim(actions) = (memory_size=B,1) and gather() in torch expects dim like (X,) that's why we need to flatten here. Don't use squeeze() because if B=1, (B,1) -> scalar.

        rewards_tensor = torch.as_tensor(reward_batch, dtype= torch.float32, device= device).view(-1) #Even if rewards only take 0 or -1 a values, they are input to a continuous function, NN operates in floats, the update of the target would return an error if the reward is an integer.

        done_tensor = torch.as_tensor(done_batch, dtype= torch.float32, device= device).view(-1)

        # Calculate Q(s,a) from evaluation network
        eval_net.train()
        q_values_current = eval_net(current_state_tensor)  # shape (B, n_actions)
        selected_q_values = q_values_current.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Calculate the target Q-values using the bellman equation and target network
        # Gradient calculation is disabled to save memory and speed up the process. 
        with torch.no_grad():
                # Compute Q(s,a) from target network without calculating the gradients
                target_net.eval() 
                q_values_next = target_net(next_state_tensor)
                max_q_values_next = q_values_next.max(dim=1)[0]
                bootstrap_mask = 1 - done_tensor
                target_q_values = rewards_tensor + gamma * bootstrap_mask * max_q_values_next

        # Calculate the loss between the predicted and target Q-values
        loss = eval_net.loss_fct(selected_q_values, target_q_values)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        if do_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(eval_net.parameters(), max_norm)
        
        optimizer.step()

        return loss.item()

def dqn_target_update(
                eval_net,
                target_net,
                tau,
                targetNet_update_method,
):
        """ 
        Soft update for target model parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        If τ = 1 then it's a hard update.
        =======
        Args:
        =======
        @ eval_net : Evaluation network trained locally from which weights will be copied.
        @ target_net : Weights are copied to the frozen Target Network.
        @ tau : Interpolation parameter, for e.g. take the value of 1e-3
        """
        if targetNet_update_method.lower() == "hard":
                # Hard update
                target_net.load_state_dict(eval_net.state_dict())
        elif targetNet_update_method.lower() == "soft":
                # Soft update
                with torch.no_grad():
                        for target_param, eval_param in zip(target_net.parameters(), eval_net.parameters()):
                                target_param.copy_( tau*eval_param + (1-tau)*target_param ) 
