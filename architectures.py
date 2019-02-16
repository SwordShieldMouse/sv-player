from includes import *

def get_conv_dim(h, w, kernel, padding):
    return (h + 2 * padding - kernel + 1, w + 2 * padding - kernel + 1)

def get_pool_dim(h, w, kernel):
    return (h // kernel, w // kernel)

def get_padding(kernel):
    # get the padding dim necessary to ensure that dim of image is not decreasing
    # ensure that kernel is odd number so we don't have to deal with rounding issues
    assert kernel % 2 == 1, "kernel_size isn't odd"
    return (kernel - 1) // 2

class State_Rep(nn.Module):
    # TODO: how do we keep track of motion? do we need a replay buffer? an rnn?
        # important for: dodging projectiles, assessing movement of enemies for aiming
    def __init__(self, c, h, w, attn_heads = 3, gru_layers = 1, final_c = 5):
        super(State_Rep, self).__init__()
        self.c = c
        self.h = h
        self.w = w
        self.attn_heads = attn_heads
        self.final_c = final_c
        self.gru_layers = gru_layers

        # try other conv architectures in future, like densenet
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 5, kernel_size = 3, padding = 0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, self.final_c, kernel_size = 3, padding = 0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        # get the final image shape after the conv layers
        self.final_h = h
        self.final_w = w
        for _ in range(2):
            self.final_h, self.final_w = get_conv_dim(self.final_h, self.final_w, kernel = 3, padding = 0)
            self.final_h, self.final_w = get_pool_dim(self.final_h, self.final_w, kernel = 2)

        # relational nn
        self.relational_nn = Relational_NN(self.final_c, attn_heads)

        # conv_grus
        self.conv_grus = nn.ModuleList()
        for _ in range(gru_layers):
            self.conv_grus.append(Conv_GRU(self.final_c))

    def forward(self, x, hidden):
        # give a batch size since nn.conv2d requires it
        x = torch.unsqueeze(x, 0)
        # Gym gives us inputs in the form (h, w, c), so convert to (c, h, w)

        x = x.permute(0, 3, 1, 2)

        # run the relational network
        relations = self.relational_nn(x)
        assert relations.numel() == self.attn_heads * self.final_c, "size mismatch for relations network"

        # run the GRU
        conv_gru_outs = [self.conv_layers(x)]
        # 1 for the batch size
        # for holding the hidden states of the grus
        #hs = [torch.rand([1, self.final_c, self.final_h, self.final_w]).to(device)]
        for ix in range(self.gru_layers):
            gru_out, hidden = self.conv_grus[ix](conv_gru_outs[ix], hidden)
            conv_gru_outs.append(gru_out)
            #hs.append(gru_h)


        # the last output is what we'll feed into our other models
        conv_gru_out = conv_gru_outs[-1].view(1, -1)
        assert conv_gru_out.numel() == self.final_c * self.final_h * self.final_w, "size mismatch for gru"

        out = torch.cat([relations, conv_gru_out], dim = -1)
        #print(out.shape)
        return out, hidden

    def reset_hidden(self):
        return torch.rand([1, self.final_c, self.final_h, self.final_w]).to(device)

    def get_final_img_shape(self):
        return self.final_c, self.final_h, self.final_w

    def get_out_len(self):
        #conv_len = self.final_channel_size * self.final_h * self.final_w
        attn_len = self.attn_heads * self.final_c
        gru_len = self.final_c * self.final_h * self.final_w
        #return conv_len + attn_len + gru_len
        return attn_len + gru_len

class Conv_GRU(nn.Module):
    # a convolutional GRU that takes an image, performs convolutions, then passes it through a GRU
    def __init__(self, c):
        super(Conv_GRU, self).__init__()
        self.c = c

        self.r_convs = nn.ModuleList()
        self.z_convs = nn.ModuleList()
        self.n_convs = nn.ModuleList()

        for _ in range(2):
            self.r_convs.append(nn.Conv2d(c, c, kernel_size = 3, padding = get_padding(3)))
            self.z_convs.append(nn.Conv2d(c, c, kernel_size = 3, padding = get_padding(3)))
            self.n_convs.append(nn.Conv2d(c, c, kernel_size = 3, padding = get_padding(3)))

        #self.h = torch.rand([c, h, w], requires_grad = True).to(device)

    def forward(self, x, h):
        #print(h.shape)
        #print(x.shape, h.shape)
        r = torch.sigmoid(self.r_convs[0](x) + self.r_convs[1](h))
        z = torch.sigmoid(self.z_convs[0](x) + self.z_convs[1](h))
        n = torch.tanh(self.n_convs[0](x) + r * self.n_convs[1](h))
        #print(z.shape, n.shape, h.shape)
        out = (1 - z) * n + z * h

        return out, out



class Relational_NN(nn.Module):
    # models relations between objects pairs
    # or maybe we should do this: https://arxiv.org/pdf/1711.07971.pdf
        # difference seems to be just in the parametrisation
    def __init__(self, c, attn_heads):
        super(Relational_NN, self).__init__()
        self.c = c
        self.attn_heads = attn_heads

        self.gs = nn.ModuleList()
        for _ in range(attn_heads):
            self.gs.append(nn.Sequential(
                nn.Linear(c, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 32),
                nn.LeakyReLU(),
            ))


    def forward(self, x):
        # input will be a feature map of size c x h x w, c is object size
        # each c x 1 x 1 vector is an object
        x = x.permute(0, 2, 3, 1) # so that we can access the channel vectors easily
        x = x.view(-1, self.c) # rows correspond to a single object vector of length c
        # There are thus h * w rows
        n_rows = x.shape[0]


        ys = []
        for ix, g in enumerate(self.gs):
            ys.append(g(x))
        # perform the interaction
        stacked_ys = torch.stack(ys)
        y = F.softmax(torch.sum(torch.bmm(stacked_ys, stacked_ys.permute(0, 2, 1)), dim = 1), dim = 1)

        # attention output would be just be vector of intensities, where length of the vector is number of objects
        attn = torch.matmul(y, x).view(1, -1)
        return attn


class Value_Fn(nn.Module):
    # conv. network that reads the image input
    # TODO: should implement a notion of objects, perhaps with a relational neural network
    def __init__(self, state_rep):
        super(Value_Fn, self).__init__()

        self.linears = nn.Sequential(
            nn.Linear(state_rep.get_out_len(), 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out = self.linears(x)
        return out


class Policy(nn.Module):
    # TODO: agent seems stuck in one position after first episode
        # seems like it is adhering to unoptimal strategy of waiting for a column to shoot up
        # but does not discover strategy of hiding behind blocks
        # mb need to add some signal to explore if reward stays the same or does not increase?
            # intrinsic reward is self-improvement? how is this different from maximizing return?
    def __init__(self, action_dim, state_rep):
        super(Policy, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(state_rep.get_out_len(), 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_dim),
            nn.LogSoftmax(dim = -1)
        )

    def forward(self, x):
        out = self.linears(x)
        return out

# TODO: try a random search to compare
