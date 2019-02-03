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
    # common class for converting observation image into something meaningful
    # somehow put in a relational neural net

    def __init__(self, c, h, w, attn_heads = 3):
        super(State_Rep, self).__init__()
        self.c = c
        self.attn_heads = 3
        self.final_channel_size = 5

        # try other conv architectures in future, like densenet
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 5, kernel_size = 3, padding = 0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, self.final_channel_size, kernel_size = 3, padding = 0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

        self.final_h = h
        self.final_w = w
        for _ in range(2):
            self.final_h, self.final_w = get_conv_dim(self.final_h, self.final_w, kernel = 3, padding = 0)
            self.final_h, self.final_w = get_pool_dim(self.final_h, self.final_w, kernel = 2)

        self.relational_nn = Relational_NN(self.final_channel_size, attn_heads)

    def forward(self, x):
        # give a batch size since nn.conv2d requires it
        x = torch.unsqueeze(x, 0)
        # Gym gives us inputs in the form (h, w, c), so convert to (c, h, w)
        #print(x.shape)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        relations = self.relational_nn(x)
        #print(relations.shape)
        return torch.cat([x.view(1, -1), relations], dim = -1)

    def get_final_img_shape(self):
        return self.final_channel_size, self.final_h, self.final_w

    def get_out_len(self):
        return self.final_channel_size * self.final_h * self.final_w + self.attn_heads * self.final_channel_size

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

        # probably need multiheaded attention to deal with different kinds of relationships
            # e.g., shields, enemies, projectiles
        # try something more efficient with matrix multiplication
        # each row of x should interact with all the other rows
        # first project to a subspace
        ys = torch.zeros([self.attn_heads, n_rows, 32])
        #print(ys.shape)
        for ix, g in enumerate(self.gs):
            ys[ix] = g(x)
        #y = self.g(x)
        # perform the interaction
        #print(torch.sum(torch.bmm(ys, ys.permute(0, 2, 1)), dim = 1).shape)
        ys = F.softmax(torch.sum(torch.bmm(ys, ys.permute(0, 2, 1)), dim = 1), dim = 1)
        #y = F.softmax(torch.sum(y.mm(torch.t(y)), dim = 0), dim = 0)
        # now we have a n_rows by n_rows matrix; problem is we only have one number for each interaction

        # attention output would be just be vector of intensities, where length of the vector is number of objects
        attn = torch.matmul(ys, x).view(1, -1)
        return attn


class Value_Fn(nn.Module):
    # conv. network that reads the image input
    # TODO: should implement a notion of objects, perhaps with a relational neural network
    def __init__(self, c, h, w, attn_heads = 3):
        super(Value_Fn, self).__init__()

        self.state_rep = State_Rep(c, h, w, attn_heads)

        self.linears = nn.Sequential(
            nn.Linear(self.state_rep.get_out_len(), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.state_rep(x)
        x = self.linears(x)
        return x


class Policy(nn.Module):
    # should policy and value function share representation of the image?
    # maybe yes since this reduces training time and things that are relevant for policy should also be relevant for value function?
    def __init__(self, action_dim, c, h, w, attn_heads = 3):
        super(Policy, self).__init__()
        self.h = h
        self.w = w
        self.c = c

        self.state_rep = State_Rep(c, h, w, attn_heads)

        self.linears = nn.Sequential(
            nn.Linear(self.state_rep.get_out_len(), 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_dim),
            nn.LogSoftmax(dim = -1)
        )

    def forward(self, x):
        x = self.state_rep(x)
        x = self.linears(x)
        return x
