from .hgru import hGRU


def setup_model(**kwargs):
    timesteps = kwargs.get('timesteps')
    jacobian_penalty = kwargs.get('penalty')
    filter_size = kwargs.get('kernel_size', 15)

    if kwargs.get('model') == 'hgru':
        print("Init model hGRU ", kwargs.get('algo'), 'penalty: ', jacobian_penalty, 'steps: ', timesteps)
        model = hGRU(timesteps=timesteps, filt_size=filter_size, num_iter=15, exp_name=kwargs.get('name'),
                     jacobian_penalty=jacobian_penalty,
                     grad_method=kwargs.get('algo'),
                     activ=kwargs.get('activ', 'softplus'),
                     xavier_gain=kwargs.get('xavier_gain', 1.0),
                     num_channels=kwargs.get('n_hidden_channels', 25),
                     n_in=kwargs.get('n_in', 4),
                     n_classes=kwargs.get('n_classes', 2))
    else:
        raise NotImplementedError('Model not implemented')

    print(sum([p.numel() for p in model.parameters() if p.requires_grad]))

    return model
