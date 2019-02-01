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

    def __init__(self, c, h, w, obj_out_dim = 16):
        super(State_Rep, self).__init__()
        self.c = c

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

        self.relational_nn = Relational_NN(self.final_channel_size, obj_out_dim)

    def forward(self, x):
        # give a batch size since nn.conv2d requires it
        x = torch.unsqueeze(x, 0)
        # Gym gives us inputs in the form (h, w, c), so convert to (c, h, w)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        relations = self.relational_nn(x)
        return torch.cat([x.view(1, -1), relations]) # output is of size 1 x (h * w + obj_out_dim) if we don't downsample the image

class Relational_NN(nn.Module):
    # models relations between objects pairs
    # or maybe we should do this: https://arxiv.org/pdf/1711.07971.pdf
        # difference seems to be just in the parametrisation
    # what dimennsion should be output be?
    def __init__(self, c, output_size):
        super(Relational_NN, self).__init__()
        self.c = c
        self.g_output_size = 32

        self.f = nn.Sequential(
            nn.Linear(self.g_output_size, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size),
            nn.LeakyReLU()
        )

        self.g = nn.Sequential(
            nn.Linear(c * 2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, self.g_output_size),
            nn.LeakyReLU(),
        )


    def forward(self, x):
        # input will be a feature map of size c x h x w, c is object size
        # each c x 1 x 1 vector is an object
        x = x.permute(0, 2, 3, 1) # so that we can access the channel vectors easily
        x = x.view(-1, self.c) # rows correspond to a single object vector of length c
        # There are thus h * w rows
        n_rows = x.shape[0]
        print(n_rows)

        # how do we do this efficiently? it's basically like a non-local convolution
        # could do something involve a matrix multiplication x x^T that would give us the object pairs feature map
            # use some weight matrix as well?
        # do it non-efficiently first to get a minimum working prototype
        # to make this faster, could  only focus on some relations and not all n choose 2 of them?
        inner_sum = torch.zeros([n_rows * (n_rows - 1) // 2, self.g_output_size], requires_grad = True)
        ix = 0
        for i in range(0, n_rows):
            print("iteration {}".format(i))
            for j in range(i + 1, n_rows):
                #print(inner_sum[ix].shape)
                #print(self.g(torch.cat([x[i, :], x[j, :]])).shape)
                inner_sum[ix] = self.g(torch.cat([x[i, :], x[j, :]]))
                ix += 1
        print("done relational")
        return self.f(torch.sum(inner_sum, 0)) # returns a 1 x output_size vector


class Value_Fn(nn.Module):
    # conv. network that reads the image input
    # TODO: should implement a notion of objects, perhaps with a relational neural network
    def __init__(self, c, h, w, obj_out_dim = 30):
        super(Value_Fn, self).__init__()

        self.state_rep = State_Rep(c, h, w, obj_out_dim)
        self.linears = nn.Sequential(
            nn.Linear(h * w + obj_out_dim, 128),
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
    def __init__(self, action_dim, c, h, w, obj_out_dim = 30):
        super(Policy, self).__init__()
        self.h = h
        self.w = w
        self.obj_out_dim = obj_out_dim
        self.c = c

        self.state_rep = State_Rep(c, h, w, obj_out_dim)
        self.linears = nn.Sequential(
            nn.Linear(h * w + obj_out_dim, 128),
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
