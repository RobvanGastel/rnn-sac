# rnn-sac
Many popular implementations of (Discrete) Soft Actor-Critic (SAC) lack the implementation of Recurrent Neural networks (RNN) for the policy and value functions. Therefore, I combined SAC with an RNN policy, as done to precursors of SAC, such as DDPG and TD3. These RNN policies are seen as memory-based control methods and have seen application in various domains such as meta-learning and multi-task learning. However, SAC with RNN has not been explored yet to the best of my knowledge. The implementation is based on the implementation of [SpinningUp](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) by OpenAI.

## Setup
The `./install_script.sh` contains the instructions to create the conda-environment and install the right versions of the libraries.
Install the following packages.

## Implementation
It should be noted that off-policy agents can't naively be combined with meta-learning, as the off-policy optimization from past experience does not guarantee to optimize for the current task. This is confirmed by the extension of PEARL, a work which extends meta-learning to off-policy learning algorithm SAC with probabilistic context variables. Currently, the agent can solve the CartPole task.

**TODO:**
- [ ] Add option to use the a continuous policy
- [ ] Extend SAC with the Probabilistic Context Variables (PEARL) for meta-learning

<br/>

## References
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2019). Soft Actor-Critic Algorithms and Applications (arXiv:1812.05905). arXiv. https://doi.org/10.48550/arXiv.1812.05905
- Wang, J. X., Kurth-Nelson, Z., Tirumala, D., Soyer, H., Leibo, J. Z., Munos, R., Blundell, C., Kumaran, D., & Botvinick, M. (2017). Learning to reinforcement learn. ArXiv:1611.05763 [Cs, Stat]. http://arxiv.org/abs/1611.05763
- Rakelly, K., Zhou, A., Quillen, D., Finn, C., & Levine, S. (2019). Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables (arXiv:1903.08254). arXiv. https://doi.org/10.48550/arXiv.1903.08254
