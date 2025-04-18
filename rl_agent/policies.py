import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import make_proba_distribution

class TCN(nn.Module):
    def __init__(self, input_channels, num_filters, num_layers, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_filters
            layers.append(
                nn.Conv1d(in_channels, num_filters, kernel_size, dilation=dilation_size)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TcnPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(TcnPolicy, self).__init__(*args, **kwargs)
        self.tcn = TCN(input_channels=self.features_dim, num_filters=64, num_layers=4)
        self.mlp_extractor = self._build_mlp_extractor()

    def _build_mlp_extractor(self):
        # Reshape the input to (batch_size, num_input_channels, sequence_length)
        def forward(features):
            batch_size = features.size(0)
            sequence_length = self.features_dim // self.features_dim
            x = features.view(batch_size, self.features_dim, sequence_length)
            x = self.tcn(x)
            x = x.view(batch_size, -1)  # Flatten
            return x
        return forward

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features), self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        value = self.value_net(latent_vf)
        return distribution, value

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std) 