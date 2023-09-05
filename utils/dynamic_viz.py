import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


def setup_plot(img, u, activations):

    timesteps = u.shape[0]

    # Setting up subplots
    # ====================
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(14*0.75, 4*0.75), gridspec_kw={'wspace': 0.4})
    plt.subplots_adjust(top=0.85, bottom=0.15)
    fig.canvas.draw()

    # Uncertainty
    # ====================================

    axs[1].plot(np.array(range(0, u.shape[0])), u, color="#4f4f4f")
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, timesteps-1)
    axs[1].set_xlabel("Time ($t$)", fontsize=20)
    axs[1].set_ylabel("Uncertainty (" + u"\u03F5" + ")", color='black', fontsize=20)
    axs[1].tick_params(axis='y', labelcolor='black', labelsize=20)
    axs[1].tick_params(axis='x', labelcolor='black', labelsize=20)
    axs[1].set_xticks([0, timesteps-1])
    axs[1].set_xticklabels(["0", "$T$"])
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels(["0", "1"])
    axs[1].xaxis.set_label_coords(.5, -.05)
    axs[1].fill_between(np.array(range(0, u.shape[0])), 0, u, alpha=0.4, color='black')

    (dot1positive,) = axs[1].plot([], [], color='#158daf', marker='o', linestyle='',
                                  label="Predicts 'Yes'", animated=True)
    (dot1negative,) = axs[1].plot([], [], color='#c5d822', marker='d', linestyle='',
                                  label="Predicts 'No'", animated=True)

    lgnd = axs[1].legend(loc='upper left', fontsize=6)
    lgnd.get_frame().set_edgecolor('black')

    at = AnchoredText(r"$\xi_{cRNN}$" + " = {:.2f}".format(np.trapz(u)), prop=dict(size=6), frameon=True,
                      loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axs[1].add_artist(at)


    # Image
    # ====================================
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Input', fontsize=18)

    # Skeleton for hidden states
    # =====================================

    vmin_hidden = np.percentile(activations, 1)
    vmax_hidden = np.percentile(activations, 99)

    viz = axs[2].imshow((activations[0, :, :]), vmin=vmin_hidden, vmax=vmax_hidden, cmap=plt.get_cmap("Reds"),
                        animated=True)
    axs[2].axis('off')
    axs[2].set_title('Activations', fontsize=18)

    return fig, axs, dot1positive, dot1negative, viz

#
# =====================================

# bg = fig.canvas.copy_from_bbox(fig.bbox)
# fig.canvas.blit(fig.bbox)

# Dynamic elements
# =====================================


def drawframe(t, predictions, u, activations, dot1positive, dot1negative, viz):

    #fig.canvas.restore_region(bg)

    # Adding dots over time
    dot1positive.set_xdata(np.array(range(0, u.shape[0]))[:t + 1][predictions[:t + 1] == 1])
    dot1positive.set_ydata(u[:t + 1][predictions[:t + 1] == 1])
    dot1negative.set_xdata(np.array(range(0, u.shape[0]))[:t + 1][predictions[:t + 1] == 0])
    dot1negative.set_ydata(u[:t + 1][predictions[:t + 1] == 0])

    # Update activations panel
    viz.set_data(activations[t, :, :])

    return dot1negative, dot1negative, viz



