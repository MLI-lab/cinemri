
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import torch
import PIL.Image
from matplotlib.patches import Ellipse


def renderMovie(imgs, N_repetitions=1, path="/root/movie.mp4", vmax=-1, interval=50):
    """
    Create an `.mp4` file from an array of images.
    """
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(111)
    ax.axis("off")
    
    if vmax == -1:
        vmax = np.max(np.abs(imgs[0]))

    ims = []
    for c in range(N_repetitions):
        for (i,f) in enumerate(imgs):
            im = ax.imshow(np.abs(f), cmap="gray", vmax=vmax, animated=True)
            if i == 0 and c == 0:
                ax.imshow(np.abs(f), cmap="gray", vmax=vmax)  # show an initial one first
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
    ani.save(path)

def renderMovieColor(imgs, N_repetitions=1, path="/root/movie.mp4", vmin=-1, vmax=-1, interval=50):
    """
    Create an `.mp4` file from an array of images.
    """
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(111)
    ax.axis("off")
    
    if vmax == -1:
        vmax = np.max(imgs[0])

    if vmin == -1:
        vmin = np.min(imgs[0])

    ims = []
    for c in range(N_repetitions):
        for (i,f) in enumerate(imgs):
            im = ax.imshow(f, vmin=vmin, vmax=vmax, animated=True)
            if i == 0 and c == 0:
                ax.imshow(f, vmin=vmin, vmax=vmax)  # show an initial one first
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
    ani.save(path)

def render_img_cross_section_movie(imgs, x=140, y_range=[50,350], angle=50, vmax=-1, path="/root/movie.mp4", interval=50):
    height, width = imgs[0].shape

    y_range = np.array(y_range)

    if vmax == -1:
        vmax = np.max(np.abs(imgs[0]))

    c, s = np.cos(angle * math.pi / 180), np.sin(angle * math.pi / 180)
    R_inv = np.matrix(((c, -s), (s, c)))

    # subtract -> rotate -> add, for rotation about the center of the image
    pos = np.matrix(np.stack((np.array([x, x]) - width/2, y_range - height/2)))
    pos_rotated = R_inv @ pos
    x_rotated = pos_rotated[0,:] + width/2
    y_rotated = pos_rotated[1,:] + height/2

    frames = [] # for storing the generated images
    fig = plt.figure(figsize=(7,7))
    for img in imgs:
        a = plt.imshow(np.abs(img), vmax=vmax, cmap="gray", animated=True)
        plt.plot(np.array(x_rotated)[0], np.array(y_rotated)[0], c="red")
        frames.append([a])

    ani = animation.ArtistAnimation(fig, frames, interval=interval, blit=True)
    ani.save(path)


def plot_sensemaps(smaps): # smaps must have shape (Nx, Ny, Nc)
    """
    Plot the magnitude of all sensemaps as images.
    Input shape: (Nx, Ny, Nc)
    """
    (Nx, Ny, Nc) = smaps.shape
    w = math.ceil(math.sqrt(Nc))
    fig, axes = plt.subplots(w, math.ceil(Nc/w),figsize = (16,16))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.3)
    for i in range(Nc):
        axes[i//w,i%w].set_title("Coil "+ str(i))
        axes[i//w,i%w].imshow(np.abs(smaps[:,:,i].T))


# source for verification: http://www.cs.utah.edu/~tch/CS6640F2020/resources/How%20to%20draw%20a%20covariance%20error%20ellipse.pdf
def sigma_ellipse(mu, cov):
    w, t = np.linalg.eig(cov)
    if w[1] <= w[0]:
        angle = -math.atan2(t[0,1], t[0,0])
    else:
        angle = -math.atan2(t[1,1], t[1,0])

    ellipse = Ellipse(mu,
        width=math.sqrt(np.max(w))*2,
        height=math.sqrt(np.min(w))*2,
        angle=angle*180/math.pi,
        edgecolor='r',
        facecolor='none',
        alpha=0.3)

    return ellipse