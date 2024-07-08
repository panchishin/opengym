
from matplotlib import pyplot as plt


def plot_res(values, title=''):   
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    # clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    
    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


def plot_many(values, title='', titles=None):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    # clear_output(wait=True)
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,5))
    f.suptitle(title)
    if titles is None:
        for value in values:
            ax.plot(value)
    else:
        for value, label in zip(values, titles):
            ax.plot(value, label=label)
    ax.axhline(195, c='red',ls='--', label='goal')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.legend()
    
    plt.show()


def plot_mean_std(values_mean, values_std, title, titles):
    
    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    
    for value, label in zip(values_mean, titles):
        ax[0].plot(value, label=label)
        ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward Mean')
    ax[0].legend()

    for value, label in zip(values_std, titles):
        ax[1].plot(value, label=label)
        ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Reward STD')
    ax[1].legend()

    plt.show()

def scatter_plot(x, y, title, xlabel, ylabel):
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    plot_mean_std([[200,400,100],[100,200,300,400,500,400,300,200,100]], [[5,80,20],[10,20,30,40,50,40,30,20,10]],"test", ["a","b"])