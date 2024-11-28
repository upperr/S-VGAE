import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, encoding, surrogate
from layers import SpikingFC, PotentialOutput, GraphConvolution, SigmoidSurrogate, WeightedInnerProduct_Link, WeightedInnerProduct_Graph
from utils import kl_bernoulli, count_propagation, count_transformation, count_inner_product_link, count_inner_product_graph, count_readout

class SGAE(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, n_features, inputs_encoding,
                 task = 'link_prediction', T = 10, dropout = 0., tau = 2., threshold = 0.2, reset = None, bn = False):
        super().__init__()

        self.input_dim = n_features
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.T = T
        self.inputs_encoding = inputs_encoding
        self.task = task
        # build Encoder
        input_dim = n_features
        encoder = []
        for layer, layer_dim in enumerate(encoder_dim):
            # propagation
            layers = [GraphConvolution()]
            # use Poisson encoding for the first layer
            if layer == 0:
                layers.append(encoding.PoissonEncoder())
            else:
                layers.append(neuron.LIFNode(tau = tau, decay_input = False, v_threshold = threshold, v_reset = reset))
            # transformation
            layers.append(SpikingFC(input_dim, layer_dim, bias = False, dropout = dropout, bn = bn))
            layers.append(neuron.LIFNode(tau = tau, decay_input = False, v_threshold = threshold, v_reset = reset, surrogate_function = surrogate.PiecewiseLeakyReLU(w = 0.5, c = 0.)))
            input_dim += layer_dim
            encoder.append(nn.Sequential(*layers))
        self.encoder = nn.Sequential(*encoder)
        # build Decoder
        layers = [SpikingFC(encoder_dim[-1], decoder_dim, bias = True, dropout = dropout, bn = bn)]
        layers.append(neuron.LIFNode(tau = tau, decay_input = False, v_threshold = threshold, v_reset = reset, surrogate_function = surrogate.PiecewiseLeakyReLU(w = 0.5, c = 0.)))
        self.decoder = nn.Sequential(*layers)
        if task == 'link_prediction':
            self.reconstruct = WeightedInnerProduct_Link(decoder_dim)
        elif task == 'graph_generation':
            self.reconstruct = WeightedInnerProduct_Graph(decoder_dim)
        self.readout = PotentialOutput()

    def forward(self, x, adj, nodes_s = None, nodes_t = None, flops = False):
        self.muls = 0
        self.acs = 0
        if self.task == 'link_prediction':
            # spiking GCN encoder
            h_s, h_t = self.encode_link(x, adj, nodes_s, nodes_t, flops)
            # probabilistic spiking decoder
            self.reconstruction = self.decode_link(h_s, h_t, act = torch.sigmoid, flops = flops)
        elif self.task == 'graph_generation':
            # spiking GCN encoder
            h_g = self.encode_graph(x, adj, flops)
            # probabilistic spiking decoder
            self.reconstruction = self.decode_graph(h_g, act = torch.sigmoid, flops = flops)
        return self.reconstruction
    
    def encode_link(self, x, adj, nodes_s, nodes_t, flops = False):
        h_s = torch.zeros([nodes_s.size(0), self.encoder_dim[-1], self.T], device = x.device) # (N, C)
        h_t = torch.zeros([nodes_t.size(0), self.encoder_dim[-1], self.T], device = x.device) # (N, C)
        for t in range(self.T):
            # encode inputs as spiking inputs
            inputs = self.inputs_encoding(x)
            for layer in range(len(self.encoder_dim)):
                # calculate FLOPs
                if flops:
                    muls, acs = count_propagation(self.encoder[layer][0], inputs, adj)
                    self.muls += 0
                    self.acs += 2 * int(acs / inputs.size(0))
                # propagation
                h = self.encoder[layer][0]([inputs, adj])
                h_spikes = self.encoder[layer][1](h)
                # calculate FLOPs
                if flops:
                    muls, acs = count_transformation(self.encoder[layer][2], h_spikes)
                    self.muls += 0
                    self.acs += 2 * int(acs / inputs.size(0))
                # transformation
                z = self.encoder[layer][2](h_spikes)
                z_spikes = self.encoder[layer][3](z)
                # skip-connections
                inputs = torch.cat([z_spikes, h_spikes], dim = 1)
            h_s[:, :, t] = z_spikes[nodes_s]
            h_t[:, :, t] = z_spikes[nodes_t]
        return h_s, h_t

    def decode_link(self, h_s, h_t, act = lambda x: x, flops = False):
        self.q_logit = torch.zeros([h_s.size(0) + h_t.size(0), self.decoder_dim, self.T], device = h_s.device) # (N, C)
        self.z = torch.zeros([h_s.size(0) + h_t.size(0), self.decoder_dim, self.T], device = h_s.device) # (N, C, T)
        y = torch.zeros([h_s.size(0), self.T], device = h_s.device) # (N, T)
        for t in range(self.T):
            # calculate FLOPs
            if flops:
                muls, acs = count_transformation(self.decoder[0], torch.cat([h_s[:, :, t], h_t[:, :, t]], dim = 0))
                self.muls += 0
                self.acs += int(acs / (h_s.size(0) + h_t.size(0)))
            self.z[:, :, t] = self.decoder(torch.cat([h_s[:, :, t], h_t[:, :, t]], dim = 0))
            self.q_logit[:, :, t] = self.decoder[1].v
            # calculate FLOPs
            if flops:
                muls, acs = count_inner_product_link(self.reconstruct, [self.z[:h_s.size(0), :, t], self.z[h_s.size(0):, :, t]])
                self.muls += 0
                self.acs += int(acs / h_s.size(0))
            y[:, t] = self.reconstruct([self.z[:h_s.size(0), :, t].clone(), self.z[h_s.size(0):, :, t].clone()])
        # calculate FLOPs
        if flops:
            muls, acs = count_readout(self.reconstruct, y)
            self.muls += int(muls / y.size(0))
            self.acs += int(acs / y.size(0))
        reconstruction = self.readout(y)  # N
        return act(reconstruction)
    
    def encode_graph(self, x, adj, flops = False):
        h_g = torch.zeros([adj.size(0), self.encoder_dim[-1], self.T], device = x.device) # (N, C)
        for t in range(self.T):
            # encode inputs as spiking inputs
            inputs = self.inputs_encoding(x)
            for layer in range(len(self.encoder_dim)):
                # calculate FLOPs
                if flops:
                    muls, acs = count_propagation(self.encoder[layer][0], inputs, adj)
                    self.muls += 0
                    self.acs += acs
                # propagation
                h = self.encoder[layer][0]([inputs, adj])
                h_spikes = self.encoder[layer][1](h)
                # calculate FLOPs
                if flops:
                    muls, acs = count_transformation(self.encoder[layer][2], h_spikes)
                    self.muls += 0
                    self.acs += acs
                # transformation
                z = self.encoder[layer][2](h_spikes)
                z_spikes = self.encoder[layer][3](z)
                # skip-connections
                inputs = torch.cat([z_spikes, h_spikes], dim = 1)
            h_g[:, :, t] = z_spikes.clone()
        return h_g

    def decode_graph(self, h_g, act = lambda x: x, flops = False):
        self.q_logit = torch.zeros([h_g.size(0), self.decoder_dim, self.T], device = h_g.device) # (N, C)
        self.z = torch.zeros([h_g.size(0), self.decoder_dim, self.T], device = h_g.device) # (N, C, T)
        y = torch.zeros([h_g.size(0), h_g.size(0), self.T], device = h_g.device) # (N, N, T)
        for t in range(self.T):
            # calculate FLOPs
            if flops:
                muls, acs = count_transformation(self.decoder[0], h_g[:, :, t])
                self.muls += 0
                self.acs += acs
            self.z[:, :, t] = self.decoder(h_g[:, :, t])
            self.q_logit[:, :, t] = self.decoder[1].v
            # calculate FLOPs
            if flops:
                muls, acs = count_inner_product_graph(self.reconstruct, [self.z[:, :, t].clone(), self.z[:, :, t].clone()])
                self.muls += 0
                self.acs += acs
            y[:, :, t] = self.reconstruct([self.z[:, :, t].clone(), self.z[:, :, t].clone()])
        reconstruction = self.readout(y)  # (N, N)
        # calculate FLOPs
        if flops:
            muls, acs = count_readout(self.reconstruct, y)
            self.muls += muls
            self.acs += acs
        return act(reconstruction)

    def loss_function(self, labels, kl_weight = 1., p_prior = 0.1):
        # cross-entropy reconstruction loss
        recons_loss = torch.mean(-labels * torch.log(self.reconstruction + 1e-15) - (1 - labels) * torch.log(1 - self.reconstruction + 1e-15))
        loss = recons_loss
        return loss, recons_loss

class SVGAE(SGAE):
    def __init__(self, encoder_dim, decoder_dim, n_features, inputs_encoding,
                 task = 'link_prediction', T = 10, dropout = 0., tau = 2., threshold = 0.2, reset = None, bn = False):
        super().__init__(encoder_dim, decoder_dim, n_features, inputs_encoding, task, T, dropout, tau, threshold, reset, bn)

        self.input_dim = n_features
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.T = T
        self.inputs_encoding = inputs_encoding
        self.task = task
        # build Encoder
        input_dim = n_features
        encoder = []
        for layer, layer_dim in enumerate(encoder_dim):
            # propagation
            layers = [GraphConvolution()]
            # use Poisson encoding for the first layer
            if layer == 0:
                layers.append(encoding.PoissonEncoder())
            else:
                layers.append(neuron.LIFNode(tau = tau, decay_input = False, v_threshold = threshold, v_reset = reset))
            # transformation
            layers.append(SpikingFC(input_dim, layer_dim, bias = False, dropout = dropout, bn = bn))
            layers.append(neuron.LIFNode(tau = tau, decay_input = False, v_threshold = threshold, v_reset = reset, surrogate_function = surrogate.PiecewiseLeakyReLU(w = 0.5, c = 0.)))
            input_dim += layer_dim
            encoder.append(nn.Sequential(*layers))
        self.encoder = nn.Sequential(*encoder)
        # build Decoder
        layers = [SpikingFC(encoder_dim[-1], decoder_dim, bias = True, dropout = dropout, bn = bn)]
        layers.append(neuron.LIFNode(tau = tau, decay_input = False, v_threshold = threshold, v_reset = reset, surrogate_function = SigmoidSurrogate()))
        self.decoder = nn.Sequential(*layers)
        if task == 'link_prediction':
            self.reconstruct = WeightedInnerProduct_Link(decoder_dim)
        elif task == 'graph_generation':
            self.reconstruct = WeightedInnerProduct_Graph(decoder_dim)
        self.readout = PotentialOutput()

    def loss_function(self, labels, kl_weight = 1., p_prior = 0.1):
        # cross-entropy reconstruction loss
        recons_loss = torch.mean(-labels * torch.log(self.reconstruction + 1e-15) - (1 - labels) * torch.log(1 - self.reconstruction + 1e-15))
        # KL divergence
        kld = kl_bernoulli(p_prior, torch.sigmoid(self.q_logit)) / labels.size(0) / self.T
        loss = recons_loss + kl_weight * kld
        return loss, recons_loss
        