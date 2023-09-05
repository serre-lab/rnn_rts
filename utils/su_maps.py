import torch
import numpy as np
from .loss import get_edl_vars


def get_su_values(img, model, fixation_x, fixation_y, device, step_size=5, batch_size=32):
    """

    :param img:  image for which to generate a spatial uncertainty map
    :param model:  model to obtain uncertainty values from
    :param fixation_x: x locatoin of fixation dot (column)
    :param fixation_y: y location of fixation dot (row)
    :param device:  which device to use once batches are ready
    :param step_size:  step size when sampling a grid of cue dot locations (px units)
    :param batch_size: how many inputs (i.e., img but with different cue dot locations) to gather in a batch
    :return: G X T tensor containing uncertainty values. G is number of locations in grid, T is number of timesteps
    """

    print(img.shape)
    print(tuple(img.shape))
    c, h, w = tuple(img.shape)

    grid = np.zeros((h, w))
    grid[::step_size, ::step_size] = 1
    cue_y, cue_x = np.where(grid == 1)
    u_all = []

    with torch.no_grad():
        # Creating multiple stimuli from the same outline
        inputs = img[0:3].cpu()  # dropping the dot channel for now
        inputs = torch.unsqueeze(inputs, 0)
        inputs = inputs.repeat(len(cue_y), 1, 1, 1)  # copies of the outline

        # Make the dot channel (1-value indicates dot location)
        dots = torch.zeros((len(cue_y), 1, h, w))
        dots[:, :, fixation_y, fixation_x] = 1   # fixation location is constant for all inputs
        dots[range(len(cue_y)), 0, cue_y, cue_x] = 1  # cue location varies

        # Add the dot channel
        inputs = torch.cat([inputs, dots], dim=1)

        # Split it up in batches
        batches = torch.split(inputs, batch_size)

        # Run batches
        for j, b_imgs in enumerate(batches):
            b_imgs = b_imgs.to(device)

            # Run model
            output_dict = model.forward(b_imgs, 0, 0, testmode=True)  # testmode will include hidden states in output_dict

            # Process outputs
            states = output_dict['states']  # hidden states per time step
            T = len(states)  # number of timesteps (including h0)
            outputs = [model.readout(x) for x in states]  # need to apply the readout still

            num_classes = outputs[0].shape[-1]
            uncertainties = [get_edl_vars(x, num_classes=num_classes)[1].detach().cpu().numpy() for x in
                             outputs]  # compute uncertainties from model outputs
            uncertainties = np.stack(uncertainties, axis=1).squeeze(axis=2)

            u_all.append(uncertainties)

    return np.concatenate(u_all, axis=0), cue_x, cue_y