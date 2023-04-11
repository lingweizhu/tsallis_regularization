import numpy as np
from core.agent import base
from collections import namedtuple
import os
import torch

from core.network.policy_factory import MLPCont, MLPDiscrete
from core.network.network_architectures import DoubleCriticNetwork, DoubleCriticDiscrete, FCNetwork

class TsallisInAC(base.Agent):
    def __init__(self,
                 device,
                 discrete_control,
                 state_dim,
                 action_dim,
                 hidden_units,
                 learning_rate,
                 tau,
                 polyak,
                 exp_path,
                 seed,
                 env_fn,
                 timeout,
                 gamma,
                 offline_data,
                 batch_size,
                 use_target_network,
                 target_network_update_freq,
                 evaluation_criteria,
                 logger
                 ):
        super(TsallisInAC, self).__init__(
            exp_path=exp_path,
            seed=seed,
            env_fn=env_fn,
            timeout=timeout,
            gamma=gamma,
            offline_data=offline_data,
            action_dim=action_dim,
            batch_size=batch_size,
            use_target_network=use_target_network,
            target_network_update_freq=target_network_update_freq,
            evaluation_criteria=evaluation_criteria,
            logger=logger
        )
        
        def get_policy_func():
            if discrete_control:
                pi = MLPDiscrete(device, state_dim, action_dim, [hidden_units]*2)
            else:
                pi = MLPCont(device, state_dim, action_dim, [hidden_units]*2)
            return pi

        def get_critic_func():
            if discrete_control:
                q1q2 = DoubleCriticDiscrete(device, state_dim, [hidden_units]*2, action_dim)
            else:
                q1q2 = DoubleCriticNetwork(device, state_dim, action_dim, [hidden_units]*2)
            return q1q2
            
        pi = get_policy_func()
        q1q2 = get_critic_func()
        AC = namedtuple('AC', ['q1q2', 'pi'])
        self.ac = AC(q1q2=q1q2, pi=pi)
        pi_target = get_policy_func()
        q1q2_target = get_critic_func()
        q1q2_target.load_state_dict(q1q2.state_dict())
        pi_target.load_state_dict(pi.state_dict())
        ACTarg = namedtuple('ACTarg', ['q1q2', 'pi'])
        self.ac_targ = ACTarg(q1q2=q1q2_target, pi=pi_target)
        self.ac_targ.q1q2.load_state_dict(self.ac.q1q2.state_dict())
        self.ac_targ.pi.load_state_dict(self.ac.pi.state_dict())
        self.beh_pi = get_policy_func()
        self.value_net = FCNetwork(device, np.prod(state_dim), [hidden_units]*2, 1)

        self.pi_optimizer = torch.optim.Adam(list(self.ac.pi.parameters()), learning_rate)
        self.q_optimizer = torch.optim.Adam(list(self.ac.q1q2.parameters()), learning_rate)
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), learning_rate)
        self.beh_pi_optimizer = torch.optim.Adam(list(self.beh_pi.parameters()), learning_rate)
        self.exp_threshold = 10000
        if discrete_control:
            self.get_q_value = self.get_q_value_discrete
            self.get_q_value_target = self.get_q_value_target_discrete
        else:
            self.get_q_value = self.get_q_value_cont
            self.get_q_value_target = self.get_q_value_target_cont

        self.tau = tau
        self.polyak = polyak
        self.q = 2.0
        self.fill_offline_data_to_buffer()
        self.offline_param_init()
        return


    def compute_loss_beh_pi(self, data):
        """L_{\omega}, learn behavior policy"""
        states, actions = data['obs'], data['act']
        beh_log_probs = self.beh_pi.get_logprob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss, beh_log_probs
    
    def compute_loss_value(self, data):
        """L_{\phi}, learn z for state value, v = tau log z"""
        states = data['obs']
        v_phi = self.value_net(states).squeeze(-1)
        with torch.no_grad():
            actions, _ = self.ac.pi(states)
            logq_probs = self.ac.pi.get_logq_prob(states, actions)
            min_Q, _, _ = self.get_q_value_target(states, actions)
        target = min_Q - self.tau * logq_probs
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        return value_loss, v_phi.detach().numpy(), logq_probs.detach().numpy()
    
    def get_state_value(self, state):
        with torch.no_grad():
            value = self.value_net(state).squeeze(-1)
        return value

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_actions, _ = self.ac.pi(next_states)
            logq_probs = self.ac.pi.get_logq_prob(next_states, next_actions)
        next_q_values, _, _ = self.get_q_value_target(next_states, next_actions)
        q_target = rewards + self.gamma * (1 - dones) * (next_q_values - self.tau * logq_probs)
    
        minq, q1, q2 = self.get_q_value(states, actions, with_grad=True)
    
        critic1_loss = (0.5 * (q_target - q1) ** 2).mean()
        critic2_loss = (0.5 * (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        q_info = minq.detach().numpy()
        return loss_q, q_info

    def compute_loss_pi(self, data):
        """L_{\psi}, extract learned policy"""
        states, actions = data['obs'], data['act']
        log_probs = self.ac.pi.get_logprob(states, actions)
        min_Q, _, _ = self.get_q_value(states, actions, with_grad=False)
        '''
        Tsallis, 
        update towards the policy: pi_D exp_q(Q - Psi) pi_D^{-1}
        the first pi_D is absorbed into expectation, the rest becomes
        [exp_q( Q - Psi + ln_q(pi_D^-1) )^(q-1) + (q-1)^2 (Q-Psi) ln_q(pi_D^-1)]^(1/(q-1))
        approximate the normalization Psi by sqrt((Q/tau)^2 - (V-0.5*tau)/0.5*tau)
        '''
        with torch.no_grad():
            value = self.get_state_value(states)
            beh_prob = torch.clip(torch.exp(self.beh_pi.get_logprob(states, actions)), min=self.eps)
            Psi = torch.sqrt((min_Q/self.tau)**2 - (value - 0.5*self.tau)/0.5*self.tau)
        '''
        x = Q-Psi, y = ln_q pi_D^-1
        '''
        x = min_Q/self.tau - Psi
        y = self.ac.pi.logq_x(beh_prob**(-1), self.q)
        tsallis_policy = torch.pow(self.ac.pi.expq_x(x + y, self.q)**(self.q-1) + (self.q - 1)**2 * x * y, 1/(self.q-1))
        clipped = torch.clip(tsallis_policy, self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        return pi_loss, ""
    
    def update_beta(self, data):
        loss_beh_pi, _ = self.compute_loss_beh_pi(data)
        self.beh_pi_optimizer.zero_grad()
        loss_beh_pi.backward()
        self.beh_pi_optimizer.step()
        return loss_beh_pi

    def update(self, data):
        loss_beta = self.update_beta(data).item()
        
        self.value_optimizer.zero_grad()
        loss_vs, v_info, logp_info = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()

        loss_q, qinfo = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        
        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

        return {"beta": loss_beta,
                "actor": loss_pi.item(),
                "critic": loss_q.item(),
                "value": loss_vs.item(),
                "q_info": qinfo.mean(),
                "v_info": v_info.mean(),
                "logp_info": logp_info.mean(),
                }


    def get_q_value_discrete(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o)
                q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_target_discrete(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o)
            q1_pi, q2_pi = q1_pi[np.arange(len(a)), a], q2_pi[np.arange(len(a)), a]
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_cont(self, o, a, with_grad=False):
        if with_grad:
            q1_pi, q2_pi = self.ac.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        else:
            with torch.no_grad():
                q1_pi, q2_pi = self.ac.q1q2(o, a)
                q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def get_q_value_target_cont(self, o, a):
        with torch.no_grad():
            q1_pi, q2_pi = self.ac_targ.q1q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
        return q_pi.squeeze(-1), q1_pi.squeeze(-1), q2_pi.squeeze(-1)

    def sync_target(self):
        with torch.no_grad():
            for p, p_targ in zip(self.ac.q1q2.parameters(),
                                 self.ac_targ.q1q2.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.ac.pi.parameters(),
                                 self.ac_targ.pi.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def save(self):
        parameters_dir = self.parameters_dir
        path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.ac.pi.state_dict(), path)

        path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.ac.q1q2.state_dict(), path)

        path = os.path.join(parameters_dir, "vs_net")
        torch.save(self.value_net.state_dict(), path)



