# rnn-sac
Many popular implementations of (Discrete) Soft Actor-Critic (SAC) lack the implementation of Recurrent Neural networks (RNN) for the policy and value functions. Therefore, I combined SAC with an RNN policy, as done to precursors of SAC, such as DDPG and TD3. These RNN policies are seen as memory-based control methods and have seen application in various domains such as meta-learning and multi-task learning. However, SAC with RNN has not been explored yet to the best of my knowledge. The implementation is based on the implementation of [SpinningUp](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) by OpenAI.

## Install requirements
The `./install_script.sh` contains the instructions to create the conda-environment and install the right versions of the libraries.

## Implementation
It should be noted that off-policy agents can't naively be combined with meta-learning, as the off-policy optimization from past experience does not guarantee to optimize for the current task. This is confirmed by the extension of PEARL, a work which extends meta-learning to off-policy learning algorithm SAC with probabilistic context variables. Currently, the agent can solve the CartPole task.

## TODO:
- [ ] Extend SAC with the Probabilistic Context Variables (PEARL) for meta-learning


<br/>

**References**
- [RL2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)
- [Some Considerations on Learning to Explore via Meta-Reinforcement Learning](https://arxiv.org/abs/1803.01118)
- [Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](https://arxiv.org/abs/1910.10897)