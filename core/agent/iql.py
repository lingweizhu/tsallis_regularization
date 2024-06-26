from core.agent import base
import os
import torch
from torch.nn.utils import clip_grad_norm_
import core.utils.format_path as fp

"""
Changed based on https://github.com/BY571/Implicit-Q-Learning/blob/main/discrete_iql/agent.py
"""
def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)

class IQL(base.ActorCritic):
    def __init__(self, cfg):
        super(IQL, self).__init__(cfg)

        self.clip_grad_param = cfg.clip_grad_param
        self.temperature = cfg.temperature
        self.expectile = cfg.expectile
        
        self.value_net = cfg.state_value_fn()
        if 'load_params' in self.cfg.svalue_config and self.cfg.svalue_config['load_params']:
            cfg.svalue_config = fp.fill_run_number(cfg.svalue_config, cfg.run, cfg.param_setting, cfg.data_root, cfg.data_starts, cfg.data_ends)
            self.load_state_value_fn(cfg.svalue_config['path'])
        self.value_optimizer = cfg.vs_optimizer_fn(list(self.value_net.parameters()))
        
        if cfg.actor_cosin_schedule:
            self.actor_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.pi_optimizer, cfg.max_steps)

    def compute_loss_pi(self, data):
        states, actions = data['obs'], data['act']
        with torch.no_grad():
            v = self.value_net(states).squeeze(-1)
        min_Q, _, _ = self.get_q_value_target(states, actions)
        exp_a = torch.exp((min_Q - v) * self.temperature)
        exp_a = torch.min(exp_a, torch.FloatTensor([100.0]).to(states.device)).squeeze(-1)
        log_probs = self.ac.pi.get_logprob(states, actions)
        actor_loss = -(exp_a * log_probs).mean()
        return actor_loss, log_probs
    
    def compute_loss_value(self, data):
        states, actions = data['obs'], data['act']
        min_Q, _, _ = self.get_q_value_target(states, actions)

        value = self.value_net(states).squeeze(-1)
        value_loss = expectile_loss(min_Q - value, self.expectile).mean()
        return value_loss
    
    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_v = self.value_net(next_states).squeeze(-1)
            q_target = rewards + (self.gamma * (1 - dones) * next_v)
        
        _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
        critic1_loss = (0.5* (q_target - q1) ** 2).mean()
        critic2_loss = (0.5* (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())
        return loss_q, q_info
        
    def update(self, data):
        self.value_optimizer.zero_grad()
        loss_vs = self.compute_loss_value(data)
        loss_vs.backward()
        self.value_optimizer.step()
        
        loss_q, q_info = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.ac.q1q2.parameters(), self.clip_grad_param)
        self.q_optimizer.step()
        
        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.cfg.actor_cosin_schedule:
            self.actor_schedule.step()

        return loss_pi.item(), loss_q.item(), loss_vs.item()

    def save(self, early=False):
        parameters_dir = self.cfg.get_parameters_dir()
        if early:
            path = os.path.join(parameters_dir, "actor_net_earlystop")
        elif self.cfg.checkpoints:
            path = os.path.join(parameters_dir, "actor_net_{}".format(self.total_steps))
        else:
            path = os.path.join(parameters_dir, "actor_net")
        torch.save(self.ac.pi.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "critic_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "critic_net")
        torch.save(self.ac.q1q2.state_dict(), path)

        if early:
            path = os.path.join(parameters_dir, "vs_net_earlystop")
        else:
            path = os.path.join(parameters_dir, "vs_net")
        torch.save(self.value_net.state_dict(), path)


# class IQLOffline(IQLOnline):
#     def __init__(self, cfg):
#         super(IQLOffline, self).__init__(cfg)
#         self.offline_param_init()
#
#     def get_data(self):
#         return self.get_offline_data()
#
#     def feed_data(self):
#         self.update_stats(0, None)
#         return
