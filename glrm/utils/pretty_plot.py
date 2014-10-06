from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import normal
from numpy import hstack, unique, diag, ones
from numpy.linalg import svd
from numpy.ma import masked_where

COLORS = ['m', 'r', 'y', 'g', 'c', 'b', 'w']

def visualize_recovery(A, At, title, title2, n1, filename, printfig = False):
    vmin = min(A.min(), At.min())
    vmax = max(A.max(), At.max())
    my_dpi = 96
    plt.figure(figsize=(1100/my_dpi, 400/my_dpi), dpi = my_dpi)
    plt.subplot(1,3,1)
    plt.imshow(A, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.title(title)
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.subplot(1,3,2)
    plt.imshow(At, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.title(title2)
    plt.subplot(1,3,3)
    plt.imshow(hstack((abs(At[:,:n1] - A[:,:n1]), (A[:,n1:] != At[:, n1:])*vmax/2.0)),
        interpolation='nearest', vmin = vmin, vmax = vmax)
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.title("misclassified points")
    if printfig: plt.savefig('{0}.eps'.format(filename), bbox_inches = 'tight')
    plt.show()

def visualize_recovery_mixed(A, At, title, title2, n1, k, filename, printfig=False):
    vmin = min(A.min(), At.min())
    vmax = max(A.max(), At.max())
    my_dpi = 96
    plt.figure(figsize=(1400/my_dpi, 300/my_dpi), dpi = my_dpi)
    plt.subplot(1,3,1)
    plt.imshow(A, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.title(title)
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(At, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.title(title2)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(hstack(((At[:,:n1] - A[:,:n1]), A[:,n1:] != At[:, n1:])),
            interpolation='nearest', vmin = -1, vmax = 1)
    plt.colorbar()
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.title("misclassified points")
    if printfig: plt.savefig('{0}.eps'.format(filename), bbox_inches = 'tight')
    plt.show()

def visualize_recovery_missing(A, At, missing, title, title2, n1, k, mag, filename, printfig = False):
    vmin = min(A.min(), At.min())
    vmax = max(A.max(), At.max())
    my_dpi = 96
    plt.figure(figsize=(1400/my_dpi, 250/my_dpi), dpi = my_dpi)
    
    plt.subplot(1,4,1)
    plt.imshow(A, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title(title)
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    
    plt.subplot(1,4,2)
    masked_data = ones(A.shape)
    for ij in missing: masked_data[ij] = 0
    masked_data = masked_where(masked_data > 0.5, masked_data)
    plt.imshow(A, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.imshow(masked_data, cmap = cm.binary, interpolation = "nearest")
    plt.title("remove entries")
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    
    plt.subplot(1,4,3)
    plt.imshow(At, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.title(title2)
    plt.colorbar()
    
    plt.subplot(1,4,4)
    plt.imshow(hstack(((At[:,:n1] - A[:,:n1]), A[:,n1:] != At[:, n1:])),
            interpolation='nearest', vmin = -mag, vmax = mag)
    plt.colorbar()
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    plt.title("error")
    if printfig: plt.savefig('{0}.eps'.format(filename), bbox_inches = 'tight')
    plt.show()

def draw_latent_space(X, Y, names, num_numerical, color_labels, both = True, printfig = False):
    # assuming k = 3, plot profiles in 3D space
    m,n = X.shape[0], Y.shape[1]
    Ymarkers = num_numerical*['*'] + (Y.shape[1]-num_numerical)*['o']

    # svd in case X, Y have more than 3 cols, rows
    X, Y = X[:,:-1], Y[:-1, :]
    u, s, v = svd(X, full_matrices = False)
    X = X.dot(v.T[:,:3])
    u, s, v = svd(Y.T, full_matrices = False)
    Y = Y.T.dot(v.T[:,:3]).T

    # colors to be plotted
    if len(unique(color_labels)) < 7:
        color_labels = list(color_labels)
        labels = unique(color_labels)
        for i in range(len(color_labels)):
            if int(color_labels[i]) == -1: 
                color_labels[i] = 'k'
            else: color_labels[i] = COLORS[int(color_labels[i])]
    else: # grayscale
        color_labels = color_labels - color_labels.min()
        color_labels = color_labels/color_labels.max()
        color_labels = [str(i) for i in color_labels]

    fig = plt.figure(1)
    if both:
        ax = fig.add_subplot(121, projection='3d')
        for i, c in enumerate(color_labels):
            ax.scatter(X[i,0], X[i,1], X[i, 2], marker = 'o', color = c)

        ax = fig.add_subplot(122, projection = '3d')
    else: ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y[0,:num_numerical], Y[1,:num_numerical], Y[2,:num_numerical],
            marker = 'o')
    ax.scatter(Y[0,num_numerical:], Y[1,num_numerical:], Y[2,num_numerical:],
            marker = 'o')
    plt.tick_params(\
            axis="both",
            which="both",
            left = "off",
            right = "off",
            top = "off",
            labelleft ="off",
            labelbottom="off")
    mag = 0.0025*max(Y.flatten())
    for j in range(n): ax.text(Y[0,j] + mag*normal(), Y[1,j] + mag*normal(),
            Y[2,j] + mag*normal(), names[j])
    plt.show()
