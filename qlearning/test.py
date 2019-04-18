from qlearning import QLearning
import gym


def main():
    env = gym.make('FrozenLake-v0')
    model = QLearning(env, epsilon=0.2, gamma=1, alpha=0.01)
    model.train(1000000)
    model.test(10, render=False)


if __name__ == '__main__':
    main()
