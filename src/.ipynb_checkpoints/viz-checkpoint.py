import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

import holoviews as hv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')


fig_root = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'reports',
            'figs'
        )
)

def plot_similarity_matrix(sim_mat, title=None, ax=None, vmin=-1, vmax=1, cmap='bwr'):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    im = ax.matshow(sim_mat, cmap=cmap, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    cbar = fig.colorbar(im, cax=cax,ticks=[vmin,vmax])
    if title is not None:
        ax.set_title(title)
    return ax



def plot_states(states, xlabel="state", ylabel="node", title="", cbar_label="", annot=None, window_step=2.0, ax=None):

    t = states.shape[0] * window_step 
    if ax is None:
        fig, ax = plt.subplots(figsize=(20,5))
    else:
        fig = plt.gcf()
    im = ax.imshow(states.T, extent=[0,t,states.shape[1],0], aspect='auto')
    vmin,vmax = states.min(), states.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax,ticks=[vmin,vmax])
    cbar.set_label(cbar_label)
    cbar.ax.yaxis.set_label_position('left')

    if annot is not None:
        for x in annot:
            ax.axvline(x=x,c='r')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax


def plot_time_series(y, x=None, sp=1.0, unit="s", title=None, ylabel=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    if x is None:
        ax.plot(
                np.arange(y.shape[0]) * sp,
                y,
        )
    else:
        ax.plot(
                x,
                y,
        )


    ax.set_xlabel(f"time [{unit}]")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    return ax



def plot_fcd(FCD, window_step, unit="s", title=None, ax=None, labels=None, colorbar=True, cbar_label='$CC[FC(t_1), FC(t_2)]$', **kwargs):
    """
    FCD:            square FCD matrix
    window_step:    sliding window increment 
    unit:           time unit of the increment
    """

    t = FCD.shape[0] * window_step 
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    im = ax.matshow(FCD, extent=[0,t,t,0], **kwargs)

    divider = make_axes_locatable(ax)


    if labels is not None:
        lax = divider.append_axes('right', size='8%', pad=0.05)
        lax.pcolormesh(labels[:,np.newaxis], cmap="tab20")
        lax.invert_yaxis()
        lax.set_title("state")
        lax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
        )
        lax.set_title=("cluster")
    if colorbar:
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)

    if title is not None:
        ax.set_title(title,y=1.1)
    ax.set_xlabel("time [%s]" % unit)
    ax.set_ylabel("time [%s]" % unit)


def plot_fc(FC, title=None, ax=None, colorbar=True, cbar_label='$FC$', **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    im = ax.matshow(FC, **kwargs)

    divider = make_axes_locatable(ax)
    if colorbar:
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(cbar_label)

    if title is not None:
        ax.set_title(title,y=1.1)
    ax.set_xlabel('roi')
    ax.set_ylabel('roi')


def plot_connectivity(connectivity, title=None, figsize=(6,4)):
    fig, (ax_w, ax_trl) = plt.subplots(nrows=1,ncols=2, figsize=figsize)

    im = ax_w.imshow(connectivity.weights)
    cbar = fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)
    ax_w.set_title('weights')

    im = ax_trl.imshow(connectivity.tract_lengths)
    cbar = fig.colorbar(im, ax=ax_trl, fraction=0.046, pad=0.04)
    ax_trl.set_title('tract lengths')

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

def plot_raw_stacked(
    raw, frames=None, tmin=None, tmax=None, figsize=(30, 20), highlights=None, title=None
):
    ch_names = raw.info["ch_names"]
    f, axs = plt.subplots(len(ch_names), sharex=True, sharey=True, figsize=figsize)
    for ch, chname in enumerate(ch_names):
        data, times = raw[ch]
        axs[ch].plot(times, data.squeeze(), "k", label=ch_names[ch])
        if tmin or tmax:
            axs[ch].set_xlim(left=tmin, right=tmax)
        axs[ch].set_xlabel("t [s]")
        axs[ch].legend()
        axs[ch].spines["right"].set_visible(False)
        axs[ch].spines["top"].set_visible(False)
        axs[ch].yaxis.set_ticks_position("left")
        if frames is not None:
            for t in frames:
                axs[ch].axvline(t, color="r", alpha=0.1)
        if highlights is not None:
            for h0, h1 in highlights:
                axs[ch].axvspan(h0, h1, color="red", alpha=0.1)
        elif len(raw.annotations)>0:
            for ann in raw.annotations:
                axs[ch].axvspan(
                        ann["onset"],
                        ann["onset"] + ann["duration"],
                        color="red",
                        alpha=0.1
                )

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.subplots_adjust(hspace=0)


def plot_mne_raw(raw, filename=None, interval=(None, None), figsize=(48,48), title=None, scale=1.):
    plt.figure(figsize=figsize)

    t_idx = [0,len(raw)]
    assert len(interval)==2
    for i, t in enumerate(interval):
        if t is not None:
            t_idx[i] = raw.time_as_index(interval[i])[0]

    data, time = raw[:,t_idx[0]:t_idx[1]]
    names = raw.info["ch_names"]

    maxrange = 0
    for i in range(data.shape[0]):
        data[i, :] -= np.mean(data[i, :])
        contact_range = np.max(data[i, :]) - np.min(data[i, :])
        maxrange = max(maxrange, contact_range)
    data /= maxrange

    nchannels = data.shape[0]
    for i in range(nchannels):
        plt.plot(time, scale*data[nchannels - i - 1, :] + i, 'k', lw=0.4)
    plt.yticks(np.r_[:nchannels], reversed(["%-9s" % name for name in names]))

    plt.gca().xaxis.set_tick_params(labeltop='on')

    plt.xlabel("t [s]")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


def plot_ts_stack(data,scale=0.9, lw=0.4, title=None, labels=None):
    data = data - np.mean(data, axis=1, keepdims=True)
    maxrange = np.max(np.max(data, axis=1) - np.min(data, axis=1))
    data /= maxrange

    n_nodes = data.shape[1]
    fig, ax = plt.subplots(figsize=(48,0.5*n_nodes))
    for i in range(n_nodes):
        ax.plot(scale*data[:, i] + i, 'k', lw=lw)
    ax.autoscale(enable=True, axis='both', tight=True)
    if title is not None:
        ax.set_title(title)

    if labels is None:
        labels = np.r_[:n_nodes]
    ax.set_yticks(np.r_[:n_nodes])
    ax.set_yticklabels(labels)

def grid_plot(plot_fun, ncols, N, sharex=False, sharey=False, suptitle=None, subf_width=3, subf_height=3):
    """
    Plot given function in a grid. Use `functools.partial` to wrap a data plotting.

    Arguments:
        plot_fun    plotting function, will be given keyword arguments `ax` and `i` 
        N           total number of subplots
        ncols       number of columns (number of rows is calculated)
        sharex
        sharey
    """
    nrows = -(-N//ncols)
    fig, axs = plt.subplots(
            ncols=ncols, 
            nrows=nrows,
            figsize=(
                subf_width*ncols,
                subf_height*nrows
            ), 
            sharex=sharex, 
            sharey=sharey
    )
    for i in range(N):
        ax = axs.flatten()[i]
        plot_fun(i=i,ax=ax)

    if (i+1)%ncols != 0:
        for j in range(i+1, len(axs.flatten())):
            axs.flatten()[j].axis('off')

    if suptitle is not None:
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()
    return fig, axs

def plot_mne_raw_hv(raw, scale=1., cmap='gray'):
    path = hv.Path((raw.times,(raw._data.T + np.arange(raw._data.shape[0])*scale)))

    opts = hv.opts.RGB(width=1024,height=900,tools=['xpan','xwheel_zoom'],xlabel='time [s]')
    
    return datashade(path,normalization='linear',precompute=True,cmap=cmap).opts(opts)


def plot_trajectories(ax, ts, var_names=None):
    """
    ax: matplotlib axes with the phase plane
    ts: time series to plot [n_trajectories,n_time,2]
    var_names: names of the state variables
    """
    if var_names is None:
        var_names = ['x','y']
    for tr in ts:
        p = ax.plot(tr[:,0], tr[:,1])
        color = p[0].get_color()
        for i,var in enumerate(var_names):
            ax_divider = make_axes_locatable(ax)
            ax_ts = ax2_divider.append_axes("top", size="7%", pad="2%")


# %load -n phase_plane_trajectories
def phase_plane_trajectories(pp_partial, trajectories, times=None, s_var=None):
    """
    pp_partial:     phase_plane partial function (only ax parameter will be
                    provided)
    trajectories:   time series to plot [n_time,2,n_trajectories];  
    times:          time indices of the data
    s_var:          state variable names
    """

    n_tr = trajectories.shape[2]
    n_var = pp_partial.keywords['model'].nvar


    fig, axs = plt.subplots(
            nrows=2+n_tr*n_var,
            figsize=(10, 10+n_tr*0.5*n_var),
            gridspec_kw={
                'height_ratios': [10,0.1]+[1]*n_tr*n_var,
            },
    )
    axs[1].set_visible(False) # makes x axis label of the phase plane visible

    for ax in axs.flat[3:]:
        ax.get_shared_x_axes().join(axs[1], ax)
    for ax in axs.flat[2:-1]:
        ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
        ax.xaxis.offsetText.set_visible(False)

    pp_partial(ax=axs[0])

    if s_var is None:
        s_var = pp_partial.keywords['model'].state_variables[:2]

    if times is None:
        times = np.arange(n_tr)

    for i in range(n_tr):
        X = trajectories[:,0,i]
        Y = trajectories[:,1,i]
        p = axs[0].plot(X, Y, '-o', markersize=10, markevery=X.size)
        color = p[0].get_color()
        for j, var in enumerate(s_var):
            ax = axs[2 + i*2 + j]
            ax.plot(times, trajectories[:,j,i], color=color)
            ax.set_ylabel(var)

    fig.subplots_adjust()

    return fig

def plot_connectivity_centroids(conn, colors=None, axs=None):
    dims = [[0,1], [0,2], [1,2]]
    dim_labels = 'xyz'
    if axs is None:
        fig, axs = plt.subplots(ncols=3)
    for (d0,d1), ax in zip(dims, axs.flatten()):
        im = ax.scatter(conn.centres[:,d0],conn.centres[:,d1], c=colors)
        ax.set(
            xlabel=dim_labels[d0],
            ylabel=dim_labels[d1],
            aspect='equal'
        )
    if colors is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = fig.colorbar(im, cax=cax)


def plot_pca(pca_ts, title="PCA", color='k', s=1, ax=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(15,15))
    ax.scatter(pca_ts[:,0],pca_ts[:,1], c=color, s=s)
    ax.set_aspect("equal")
    ax.set_xlabel("1st component")
    ax.set_ylabel("2nd component")
    ax.set_title(title)

    return ax

