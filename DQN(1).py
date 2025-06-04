#coding = utf-8
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import numpy as np

#parameters
Batch_size = 32
Lr = 0.01
Epsilon = 0.9           #greedy policy
Gamma = 0.9             #reward discount
Target_replace_iter = 100  #target update frequency
Memory_capacity = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped

N_actions = env.action_space.n  # 2, 代表左、右两个动作
N_states = env.observation_space.shape[0] # 4, 代表4个状态分别为：小车在轨道上的位置\杆子与竖直方向的夹角\小车速度\角度变化率

ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
print(ENV_A_SHAPE) # 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_actions)
        self.out.weight.data.normal_(0,0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value =self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net,self.target_net = Net(),Net()
        self.learn_step_counter = 0  #for target updating
        self.memory_counter = 0  #for storing memory
        self.memory = np.zeros((Memory_capacity, N_states*2 + 2)) #innitialize memory
        self.optimizer = optim.Adam(self.eval_net.parameters(),lr=Lr)
        self.loss_func = nn.MSELoss()

    def choose_action(self,x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        if np.random.uniform() < Epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0,N_actions)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self,s,a,r,s_):
        transition = np.hstack((s,[a,r],s_))
        index = self.memory_counter % Memory_capacity
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        #target net update
        if self.learn_step_counter % Target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(Memory_capacity, Batch_size) # 随机选择BATCH 个(s, a, r, s_)索引
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_states]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_states:N_states+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_states+1:N_states+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_states:]))

        # 预测结果
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach() # t+1时刻
        # 当前动作的目标结果
        q_target = b_r + Gamma * q_next.max(1)[0].view(Batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
reward_list = []
epoch_list = []
print('\nCollecting experience...')
for i_episode in range(300):
    s = env.reset()  # 重置环境
    while True:
        env.render()  # 渲染出当前的智能体以及环境的状态
        ep_r = 0
        a = dqn.choose_action(s)

        #take action
        # 返回四个值：observation（object）、reward（float）、done（boolean）、info（dict），其中done表示是否应该reset环境
        s_, r, done, info = env.step(a)
        #modify the reward
        x, x_dot, theta, theta_dat = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        print(r)
        #
        epoch_list.append(i_episode)
        reward_list.append(r)
        dqn.store_transition(s, a, r, s_)  # 保存 当前状态\当前动作\当前奖励\下一个状态
        ep_r += r    # acculate reward
        if dqn.memory_counter > Memory_capacity:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2)) # 四舍五入，保留2位小数

        if done:
            break
        s = s_
env.close()
print(reward_list)
print(epoch_list)
# 生成图形
plt.scatter(epoch_list, reward_list, color='g', marker='o', s = 1)
# 显示图形
plt.show()
