import torch
import torch.nn.functional as F
from dgl.nn import GATConv  # Import GATConv from DGL


class AveragedGATConv(torch.nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop):
        super(AveragedGATConv, self).__init__()
        self.num_heads = num_heads
        self.gatconv = GATConv(
            in_feats, out_feats, num_heads, feat_drop=feat_drop
        )  # No concat flag, just concatenation

    def forward(self, g, feat):
        # Apply GATConv, which will concatenate heads
        h, attn = self.gatconv(g, feat, get_attention=True)

        # Average across the attention heads
        h = h.view(h.size(0), self.num_heads, -1)  # Reshape for averaging
        h = h.mean(dim=1)  # Average over the heads

        return h, attn


NUM_HEADS = 8


class ST_GAT(torch.nn.Module):
    """
    Spatio-Temporal Graph Attention Network as presented in https://ieeexplore.ieee.org/document/8903252
    """

    def __init__(
        self, in_channels, out_channels, n_nodes, heads=NUM_HEADS, dropout=0.0
    ):
        """
        Initialize the ST-GAT model
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param n_nodes: Number of nodes in the graph
        :param heads: Number of attention heads to use in graph
        :param dropout: Dropout probability on output of Graph Attention Network
        """
        super(ST_GAT, self).__init__()
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 9
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # single graph attentional layer with multiple attention heads using DGL's GATConv
        self.gat = AveragedGATConv(
            in_feats=in_channels,
            out_feats=in_channels,
            num_heads=heads,
            feat_drop=dropout,
        )

        # add two LSTM layers
        self.lstm1 = torch.nn.LSTM(
            input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1
        )
        for name, param in self.lstm1.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)
        self.lstm2 = torch.nn.LSTM(
            input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1
        )
        for name, param in self.lstm2.named_parameters():
            if "bias" in name:
                torch.nn.init.constant_(param, 0.0)
            elif "weight" in name:
                torch.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = torch.nn.Linear(lstm2_hidden_size, self.n_nodes * self.n_pred)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, graph, device):
        """
        Forward pass of the ST-GAT model
        :param graph: DGL graph object
        :param device: Device to operate on (e.g., 'cpu' or 'cuda')
        """
        # Get node features and edge weights from the DGL graph
        x = graph.ndata["feat"]  # Node features

        # apply dropout
        if isinstance(x, torch.Tensor):
            x = x.clone().detach().to(device)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=device)

        # gat layer: output of gat: [11400, 12]
        x, attn = self.gat(graph, x)
        
        # # Alternative Method: Apply GAT for each individual time step
        # # Reshape x to [batch_size, n_nodes, seq_length]
        # batch_size = graph.batch_size if hasattr(graph, "batch_size") else 1
        # x = x.view(batch_size, self.n_nodes, -1)
        # seq_len = x.shape[2]  # Sequence length

        # gat_outputs = []
        # attn_matrices = []

        # # Apply GAT layer to each time step separately
        # for t in range(seq_len):
        #     xt, attn = self.gat(graph, x[:, :, t])
        #     gat_outputs.append(xt.unsqueeze(0))  # Maintain time dimension
        #     attn_matrices.append(attn.unsqueeze(0))

        # # Stack over time dimension
        # x = torch.cat(gat_outputs, dim=0)  # Shape: [seq_length, batch_size, n_nodes]
        # attn_matrices = torch.cat(attn_matrices, dim=0)  # Shape: [seq_length, batch_size, n_nodes, n_nodes]
        
        x = F.dropout(x, self.dropout, training=self.training)

        # RNN: 2 LSTM
        batch_size = graph.batch_size if hasattr(graph, "batch_size") else 1
        n_node = self.n_nodes

        # Reshape x to [batch_size, n_nodes, seq_length]
        x = torch.reshape(x, (batch_size, n_node, -1))

        # For LSTM: x should be [seq_length, batch_size, n_nodes]
        x = torch.movedim(x, 2, 0)

        # Pass through LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Output contains h_t for each timestep, only the last one has all input's accounted for
        x = torch.squeeze(x[-1, :, :])

        # Linear layer: [batch_size, 128] -> [batch_size, n_nodes * n_pred]
        x = self.linear(x)

        # Reshape into final output: [batch_size, n_nodes, n_pred]
        x = torch.reshape(x, (batch_size, self.n_nodes, self.n_pred))
        x = torch.reshape(x, (batch_size * self.n_nodes, self.n_pred))

        return x, attn
