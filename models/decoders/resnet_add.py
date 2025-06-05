import torch
import torch.nn as nn


class ResnetBlockConv1d(nn.Module):
    """ 1D-Convolutional ResNet block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if norm_method == 'batch_norm':
            norm = nn.BatchNorm1d
        elif norm_method == 'sync_batch_norm':
            norm = nn.SyncBatchNorm
        else:
             raise Exception("Invalid norm method: %s" % norm_method)

        self.bn_0 = norm(size_in)
        self.bn_1 = norm(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.fc_c = nn.Conv1d(c_dim, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c_global_local):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        
        out = x_s + dx + self.fc_c(c_global_local)

        return out


class Decoder(nn.Module):
    """ Decoder conditioned by adding.

    Example configuration for scorenet_cfg (cfg.models.scorenet):
        z_dim: 128 # This is decoder's internal z_dim, not directly used for concat layer sizing if different from encoder.
        dim: 3 # xyz dimension
        hidden_size: 256
        n_blocks: 5
        out_dim: 3
    Additional needed from overall_cfg (cfg):
        overall_cfg.models.encoder.zdim
    Implicit:
        Local feature dimension from encoder is 512.
    """
    def __init__(self, overall_cfg, scorenet_cfg): # overall_cfg is the full config, scorenet_cfg is cfg.models.scorenet
        super().__init__()

        # Get dimensions from relevant config sources
        xyz_dim = scorenet_cfg.dim 
        encoder_z_dim = overall_cfg.models.encoder.zdim
        
        # The local feature dimension from the l3dp_encoder.py is fixed at 512 (output of conv4)
        # It's good practice to make this configurable in overall_cfg.models.encoder if possible,
        # but for now, we use the known fixed value.
        encoder_local_feature_dim = 512

        # These are other parameters for the decoder structure
        self.dim = xyz_dim # Store for use in forward method for 'x'
        self.out_dim = scorenet_cfg.out_dim
        hidden_size = scorenet_cfg.hidden_size
        n_blocks = scorenet_cfg.n_blocks
        
        # Calculate the actual dimension of the concatenated conditional vector c_xyz_local
        # c_xyz_local = cat([p_xyz (xyz_dim), 
        #                    c_global_expanded (encoder_z_dim + 1 for sigma), 
        #                    x_local_features (encoder_local_feature_dim)], dim=1)
        actual_concatenated_cond_dim = xyz_dim + (encoder_z_dim + 1) + encoder_local_feature_dim
        
        self.conv_p = nn.Conv1d(actual_concatenated_cond_dim, hidden_size, 1)
        
        # ResnetBlockConv1d's c_dim is the dimension of the conditional vector it receives,
        # which is also actual_concatenated_cond_dim.
        self.blocks = nn.ModuleList([
            ResnetBlockConv1d(actual_concatenated_cond_dim, hidden_size) for _ in range(n_blocks)
        ])
        self.bn_out = nn.BatchNorm1d(hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, self.out_dim, 1)
        self.actvn_out = nn.ReLU()

    def forward(self, x, c_global, x_local_features):
        """
        :param x: (bs, npoints, self.dim) Input coordinate (xyz). self.dim is scorenet_cfg.dim.
        :param c_global: (bs, overall_cfg.models.encoder.zdim + 1) Global Shape latent code from encoder + sigma.
        :param x_local_features: (bs, 512, npoints) Local features from encoder.
        :return: (bs, npoints, self.out_dim) Gradient.
        """
        p = x.transpose(1, 2)  # (bs, self.dim, n_points)
        batch_size, D_p, num_points = p.size() # D_p is self.dim (xyz_dim)

        # c_global is (bs, encoder_z_dim + 1)
        c_global_expanded = c_global.unsqueeze(2).expand(-1, -1, num_points) # (bs, encoder_z_dim + 1, n_points)
        
        # x_local_features is (bs, encoder_local_feature_dim=512, n_points)
        
        # Concatenate: p, c_global_expanded, and x_local_features
        # Resulting dim: self.dim + (encoder_z_dim + 1) + encoder_local_feature_dim
        c_xyz_local = torch.cat([p, c_global_expanded, x_local_features], dim=1) 
        
        net = self.conv_p(c_xyz_local) # self.conv_p now expects actual_concatenated_cond_dim
        for block in self.blocks:
            # ResNet blocks also receive the full c_xyz_local as conditional input
            net = block(net, c_xyz_local)
        out = self.conv_out(self.actvn_out(self.bn_out(net))).transpose(1, 2)
        return out

