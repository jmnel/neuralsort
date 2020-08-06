from sac import SAC
from trainer import Trainer

config = Config()
config.seed = 1
config.environment =

config.hyperparameters = {
    'learning_rate': 5e-3
    'gradient_clipping_norm': 5.0,
    'discount_rate': 0.99,
    'epsilon_decay_rate_denominator': 1.0,
    'normalize_reward': True,
    'exploration_worker_difference': 2.0,
    'clip_rewards': False,

    'actor': {
        'learning_rate': 3e-4,
        'batch_norm': False,
        'tau': 5e-3,
        'gradient_clipping_norm': 5,
        'initializer': 'xavier'
    },
    'critic': {
        'learning_rate': 3e-4,
        'batch_norm': False,
        'buffer_size': 1e6,
        'tau': 5e-3,
        'gradient_clipping_norm': 5,
        'initializer': 'xavier'
    },
    'min_steps_before_learning': 400,
    'batch_size': 256,
    'mu': 0.0,
    'theta': 0.15,
    'sigma': 0.25,
    'action_noise_std': 0.2,
    'action_noise_clipping_range': 0.5,
    'update_every_n_steps': 1,
    'learning_updates_per_learning_session': 1,
    'entropy_term_weight': None,
    'add_extra_noise': False,
    'do_evaluation_iterations': True
}


def main():
    pass


if __name__ == '__main__':
    main()
