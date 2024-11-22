import torch
import torch.nn as nn

class BilinearFeatureModule(nn.Module):
    def __init__(self, k_emb_dim, s_emb_dim, m_emb_dim, out_dim):
        super(BilinearFeatureModule, self).__init__()
        self.bilinear1 = nn.Bilinear(k_emb_dim, m_emb_dim, out_dim)
        self.bilinear2 = nn.Bilinear(k_emb_dim, s_emb_dim, out_dim)
        self.bilinear3 = nn.Bilinear(out_dim, out_dim, out_dim)
    
    def forward(self, k1_emb, k2_emb, s_emb, m_emb):
        combined1 = self.bilinear1(k1_emb, s_emb)
        combined2 = self.bilinear2(k2_emb, m_emb)
        combined = self.bilinear3(combined1, combined2)
        return combined
    
class MultiLayerDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MultiLayerDNN, self).__init__()
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.5))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.network(x)
        return torch.sigmoid(x)


class BilinearDNNModel(nn.Module):
    def __init__(self, k_emb_dim, s_emb_dim, m_emb_dim, dnn_hidden_sizes, dnn_output_size, bilinear_output_size=100):
        super(BilinearDNNModel, self).__init__()
        self.bilinear_module = BilinearFeatureModule(k_emb_dim, s_emb_dim, m_emb_dim, bilinear_output_size)
        self.dnn = MultiLayerDNN(bilinear_output_size, dnn_hidden_sizes, dnn_output_size)
    
    def forward(self, k1_emb, k2_emb, s_emb, m_emb):
        combined_features = self.bilinear_module(k1_emb, k2_emb, s_emb, m_emb)
        output = self.dnn(combined_features)
        return output