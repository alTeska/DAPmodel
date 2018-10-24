import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Plotting means and std
def plot_mean_std(means, stds, name=''):
    """Function takes means and distributions and generates 3 subplots visualizaing the input separately and combined"""
    idx = np.arange(0, len(means))

    mean_std, axes = plt.subplots(3, 1, figsize=(16,8))
    axes[0].plot(idx, stds, marker='*')
    axes[1].plot(idx, means, marker='o', linestyle='dashed')
    axes[2].errorbar(idx, means, stds, linestyle='dashed', marker='o')

    plt.suptitle('means and std of ' + name)

    return mean_std, axes


# Simulate the distribution
def simulate_data_distr(distr, model, stats, n_samples=1000):
    """Simulates data for a given model and summary statistics with given number of samples"""
    posterior_corr = []

    for i in np.arange(0, n_samples):
        parameters = distr.gen()[0]  # draw samples of parameters ; flatten

        data = model.gen_single(parameters)  # generate data
        data_stats = stats.calc([data])     # summary statistics of the data

        posterior_corr.append(data_stats)

    return posterior_corr


# Sampling from distributions and plotting it
def get_list_column(lst, n):
    """Returns n-th column of the list and as a separate list"""
    lst_n = [item[0][n] for item in lst]
    return lst_n


def split_list(inp):
    """Returns n x m list as m x n list"""
    out = []
    l_inp = np.shape(inp)[-1]

    for n in np.arange(0, l_inp):
        out.append(get_list_column(inp, n))

    return out


def sample_distributions(prior, posterior, M, S, ii=1000):
    """Returns samples from prior and posterior distributions"""
    prior_sampl = simulate_data_distr(prior, M, S, n_samples=ii)
    prior_sampl = split_list(prior_sampl)

    posterior_sampl = simulate_data_distr(posterior, M, S, n_samples=ii)
    posterior_sampl = split_list(posterior_sampl)
    
    return prior_sampl, posterior_sampl


def plot_distributions(prior, posterior):
    """Returns a plot of prior and posterior distributions of the parameters (sampled)"""
    l = len(prior)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    prior_df = pd.DataFrame(prior)
    posterior_df = pd.DataFrame(posterior)

    prior_df.dropna(inplace=True)
    posterior_df.dropna(inplace=True)

    print(prior_df)
    # g = sns.FacetGrid(prior_df, height=6, aspect=2., hue='variable')
    g = sns.FacetGrid(prior_df, height=6, aspect=2.)
    print(g)
    # g.map(plt.plot, 'value')
    #
    # g_post = sns.FacetGrid(posterior_df, height=6, aspect=2., hue='variable')
    # g_post.map(plt.plot, 'value')

    # for i in np.arange(0, l):
    #     ax[0].hist(prior[i], bins=int(np.sqrt(len(prior[i])))
    #                , label='param '+str(i+1))
    #     ax[1].hist(posterior[i], bins=int(np.sqrt(len(posterior[i])))
    #                , label='param '+str(i+1))
    #
    # ax[0].set_title('Prior distribution')
    # ax[1].set_title('Posterior distribution')
    # ax[0].legend()
    # ax[1].legend()

    return g
    # return g, g_post
    # pass


def plot_distributions_cross(prior, posterior, params=None):
    """Returns a plot of comparison between prior and posterior distributions of the parameters"""
    if params:
        l = int(params/2)
        fig, ax = plt.subplots(2, l, figsize=(16, 16))

        for i in np.arange(0, l):
            ax[0, i].hist(prior[i], bins=int(np.sqrt(len(prior[i])))
                    , label='prior')
            ax[0, i].hist(posterior[i], bins=int(np.sqrt(len(posterior[i])))
                    , label='post')

            ax[0, i].set_title(str(i+1) + ' parameter distr')
            ax[0, i].legend()

        for i in np.arange(0, l):
            ax[1, i].hist(prior[i+l], bins=int(np.sqrt(len(prior[i+4])))
            , label='prior')
            ax[1, i].hist(posterior[i+l], bins=int(np.sqrt(len(posterior[i+4])))
            , label='post')

            ax[1, i].set_title(str(i+l+1) + ' parameter distr')
            ax[1, i].legend()
    else:
        l = len(prior)
        fig, ax = plt.subplots(1, l, figsize=(16, 8))
        for i in np.arange(0, l):
            ax[i].hist(prior[i], bins=int(np.sqrt(len(prior[i])))
                    , label='prior')
            ax[i].hist(posterior[i], bins=int(np.sqrt(len(posterior[i])))
                    , label='post')

            ax[i].set_title(str(i+1) + ' parameter distr')
            ax[i].legend()

    return fig, ax


# Generate Dataframes from Logs
def column_names(n_col, key):
    """Creates a list with names of the columns"""
    col_names = [0] * n_col

    for n in np.arange(0, n_col):
        col_names[n] = key+str(n)

    return col_names


def merge_data(input_data, data, col_names):
    """Returns merged dataframes from multiple rounds of an experiment"""
    dd = pd.DataFrame(data=input_data, columns=col_names)
    data = pd.concat([data, dd], axis=0, sort=False, ignore_index=True)
    return data


def generate_dataframe(logs, key, melted=False):
    """Returns dataframe based on SNPE logs for a given key(observable)"""

    if not type(key)  is str:  raise TypeError("key must be a string")
    if not type(logs) is list: raise TypeError("logs must be a list")
    if not any(c for c in logs[0].keys() if key == c):
        raise ValueError("key not found in observable list")

    data = pd.DataFrame()

    n_rounds = len(logs)
    lenght_type = len(np.shape(logs[0][key]))

    # loss only
    if lenght_type == 1:
        data_temp = logs[0][key]
        for ii in np.arange(1, n_rounds):
            dd = logs[ii][key]
            data_temp = np.concatenate((data_temp, dd))

        data = pd.DataFrame(data_temp, columns=['loss'])

    # all of the biases
    elif lenght_type == 2:
        n_hiddens = np.shape(logs[0][key])[1]
        col_names = column_names(n_hiddens, key)

        for ii in np.arange(0, n_rounds):
            data = merge_data(logs[ii][key][:, :], data, col_names)

    elif lenght_type == 3:
        # hidden layers
        if 'h' in key:
            n_hiddens = np.shape(logs[0][key])[2]
            col_names = column_names(n_hiddens, key)

            for ii in np.arange(0, n_rounds):
                data = merge_data(logs[ii][key][:, 0, :], data, col_names)

        # weights only
        # BUG: size does not agree
        elif 'weights' in key:
            n_hiddens = np.shape(logs[0][key])[1]
            n_components = np.shape(logs[0][key])[2]

            keys_names = column_names(n_components, key)

            for ii in np.arange(0, n_rounds):
                for i in np.arange(0, n_hiddens):
                    keys_names_temp = [s + str(i) for s in keys_names]
                    data_temp = merge_data(logs[ii][key][:, i, :], data_temp,
                                           col_names)

                data = pd.concat([data, data_temp], axis=0, sort=False, ignore_index=True)

        # means and precision
        else:
            n_hiddens = np.shape(logs[0][key])[1]
            col_names = column_names(n_hiddens, key)

            for ii in np.arange(0,n_rounds):
                data = merge_data(logs[ii][key][:, :, 0], data, col_names)

    else:
        print ('unsolved, longer then 3')

    # reshape(melt) the dataframe for plotting
    if melted:
        data = data.melt()

    return data


def dataframe_to_plot(df, key, melted=False):
    """Generates plot from data frame with chosen key"""
    if melted:
        g = sns.FacetGrid(df, height=6, aspect=2., hue='variable')
        g.map(plt.plot, 'value')
        plt.legend()

    else:
        g = sns.FacetGrid(df, height=6, aspect=2.)
        g.map(plt.plot, key)

    plt.title(key)

    return g


def logs_to_plot(logs, key, melted=False):
    """Generates plot of given key from SNPE logs"""
    df = generate_dataframe(logs, key, melted=melted)
    g = dataframe_to_plot(df, key, melted=melted)

    return g
