import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import init


class dummyhgru(Function):
    '''
    Taken from: https://github.com/c-rbp/pathfinder_experiments/blob/main/models/hgrucleanSEG.py
    '''
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors

        args = ctx.args
        truncate_iter = args[-1]
        exp_name = args[-2]
        i = args[-3]
        epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=neumann_v_prev,
                                            retain_graph=True, allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)

            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g

            normsv.append(normv.data.item())
            normsg.append(normg.data.item())

        return None, neumann_g, None, None, None, None


class hgruCell(nn.Module):

    """
    Minor changes from:
    https://github.com/c-rbp/pathfinder_experiments/blob/main/models/hgrucleanSEG.py
    """
    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8, grad_method='bptt',
                 activ='softplus'):
        super(hgruCell, self).__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm

        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList(
            [nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True) for i in range(4)])

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)

        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)

        for bn in self.bn:
            init.constant_(bn.weight, 0.1)

        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data
        self.softpl = nn.Softplus()
        self.softpl.register_backward_hook(lambda module, grad_i, grad_o: print(len(grad_i)))
        self.activ = getattr(F, activ)

    def forward(self, input__, prev_state_input, timestep=0):

        input_ = input__
        prev_state2 = prev_state_input

        g1_t = torch.sigmoid((self.u1_gate(prev_state2)))

        c1_t = self.bn[1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))

        next_state1 = self.activ(input_ - self.activ(c1_t * (self.alpha * prev_state2 + self.mu)))  # h1(t)

        g2_t = torch.sigmoid((self.u2_gate(next_state1)))
        c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))

        h2_t = self.activ(self.kappa * next_state1 + self.gamma * c2_t + self.w * next_state1 * c2_t)
        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        prev_state2 = self.activ(prev_state2)

        return prev_state2, g2_t


class hGRU(nn.Module):

    """
    Made several changes from:
    https://github.com/c-rbp/pathfinder_experiments/blob/main/models/hgrucleanSEG.py

    The original hGRU architecture was introduced in:

    Linsley, D., Kim, J., Veerabadran, V., Windolf, C., & Serre, T. (2018). Learning long-range spatial dependencies
    with horizontal gated recurrent units. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi,
    & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 31). Curran Associates,
    Inc. https://proceedings.neurips.cc/paper_files/paper/2018/file/ec8956637a99787bd197eacd77acce5e-Paper.pdf

    """

    def __init__(self, num_channels = 25, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False,
                 grad_method='bptt', activ='softplus', xavier_gain=1.0, n_in=4, n_classes=2):
        super(hGRU, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.activ = activ
        self.xavier_gain = xavier_gain
        self.num_channels = num_channels  # number of channels in hidden states
        self.n_in = n_in   # number of input channels
        self.n_classes = n_classes  # number of classes to predict

        self.conv0 = nn.Conv2d(n_in, self.num_channels, kernel_size=7, padding=3)
        self.unit1 = hgruCell(self.num_channels, self.num_channels, filt_size, activ=self.activ)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(2, eps=1e-03)
        self.conv6 = nn.Conv2d(self.num_channels, self.n_classes, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # For classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.n_classes, self.n_classes)
        init.xavier_normal_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def readout(self, x):
        output = self.conv6(x)
        output = self.bn(output)
        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def forward(self, x, epoch, itr, target=None, criterion=None,
                testmode=False):
        x = self.conv0(x)
        x = torch.pow(x, 2)
        internal_state = torch.zeros_like(x, requires_grad=False)
        internal_state = torch.nn.init.xavier_normal_(internal_state, gain=self.xavier_gain)

        states = []
        if self.grad_method == 'rbp':
            with torch.no_grad():
                for i in range(self.timesteps - 1):
                    if testmode: states.append(internal_state)
                    internal_state, g2t = self.unit1(x, internal_state, timestep=i)
            if testmode: states.append(internal_state)
            state_2nd_last = internal_state.detach().requires_grad_()
            i += 1
            last_state, g2t = self.unit1(x, state_2nd_last, timestep=i)
            internal_state = dummyhgru.apply(state_2nd_last, last_state, epoch, itr, self.exp_name, self.num_iter)
            if testmode: states.append(internal_state)

        elif self.grad_method == 'bptt':
            for i in range(self.timesteps):
                if testmode: states.append(internal_state)
                internal_state, g2t = self.unit1(x, internal_state, timestep=i)
                if i == self.timesteps - 2:
                    state_2nd_last = internal_state
                elif i == self.timesteps - 1:
                    last_state = internal_state
            if testmode: states.append(internal_state)
        # internal_state = torch.tanh(internal_state)
        output = self.readout(internal_state)

        if not testmode:
            loss = criterion(output, target)

        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.9
        double_neg = False
        if self.training and self.jacobian_penalty:
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2
                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = \
                    torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                        create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2
            elif pen_type == 'idloss':
                norm_1_vect = torch.rand_like(last_state).requires_grad_()
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=True, allow_unused=True)[0]
                jv_penalty = (jv_prod - norm_1_vect) ** 2
                jv_penalty = jv_penalty.mean()
                if torch.isnan(jv_penalty).sum() > 0:
                    raise ValueError('Nan encountered in penalty')
        if testmode: return {'output': output, 'states': states}  # [LG] not returning loss when testmode=True cause we won't always have target

        return_dict = {}
        if type(loss) == tuple:
            assert (type(loss[1]) == dict)
            return_dict.update(loss[1])
            loss = loss[0]
        return_dict.update({'output': output, 'jv_penalty': jv_penalty, 'loss': loss})

        return return_dict
