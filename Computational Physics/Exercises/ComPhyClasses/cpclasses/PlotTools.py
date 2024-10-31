import os

def format_axis(ax, xlabel = 'x', ylabel = 'y', title = None, grid=True, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_aspect('equal')
    if legend:
        ax.legend()
    ax.grid(grid)
    return ax

def coordinate_axes(ax, lw = 1):
    ax.axhline(0, color = 'black', lw = lw, zorder = 3)
    ax.axvline(0, color = 'black', lw = lw, zorder = 3)
    return ax



def save_plot(fig, filename, **kwargs):

    current_week = os.getcwd().split('\\')[-1]
    save_dir = f'../../Weeks/Figures/{current_week}'
    #Make a directory if it doesn't exist:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
   
    fig.savefig(f'{save_dir}\\{filename}', **kwargs)
    return None