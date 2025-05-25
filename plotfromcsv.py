import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

dim = 3
folder = 'gaussian_mixture/'
folder2 = 'asymmetric_gaussian_mixture/'
name = 'gaussian_mixture'
name2 = 'asymmetric_gaussian_mixture'
label = 'Symmetric GM'
label2 = 'Asymmetric GM'
marker = ''
plotregret = True
plotmoderr = True
plotrejection = True
computetime = True
plottraj = False
comparison = True
plotlogscale = False
plotdividebyt = False

if plotregret:
    regret = np.loadtxt(f'{folder}{dim}D-{name}-regret{marker}.csv', delimiter=',')
    print(f'Number of Regret Simulations:    {regret.shape[1]}')
    plt.figure(figsize=(4, 3))
    x_vals = np.arange(1, len(regret) + 1)
    y = np.mean(regret, axis=1)
    if plotlogscale:
        y = y / (np.sqrt(x_vals))

        # Confidence interval calculation
        t_crit = stats.t.ppf([0.025, 0.975], regret.shape[1] - 1)
        sem = stats.sem(regret / np.expand_dims(np.sqrt(x_vals), axis=1), axis=1)
        lower = y + t_crit[0] * sem
        upper = y + t_crit[1] * sem

        plt.fill_between(x_vals, lower, upper, alpha=0.2)
    elif plotdividebyt:
        x_vals = x_vals[100:]
        y = y[100:]
        y = y / x_vals

        # Confidence interval calculation
        t_crit = stats.t.ppf([0.025, 0.975], regret.shape[1] - 1)
        sem = stats.sem(regret[100:] / np.expand_dims(x_vals, axis=1), axis=1)
        lower = y + t_crit[0] * sem
        upper = y + t_crit[1] * sem

        plt.fill_between(x_vals, lower, upper, alpha=0.2)
    else:
        # Confidence interval calculation
        t_crit = stats.t.ppf([0.025, 0.975], regret.shape[1] - 1)
        sem = stats.sem(regret, axis=1)
        lower = y + t_crit[0] * sem
        upper = y + t_crit[1] * sem

        plt.fill_between(x_vals, lower, upper, alpha=0.2)

    plt.plot(x_vals, y, '-', label=f'{label} {dim}D')

    if comparison:
        regret2 = np.loadtxt(f'{folder2}{dim}D-{name2}-regret{marker}.csv', delimiter=',')
        print(f'Number of Regret Simulations:    {regret2.shape[1]}')
        y = np.mean(regret2, axis=1)
        if plotlogscale:
            y = y / (np.sqrt(x_vals))

            # Confidence interval calculation
            t_crit = stats.t.ppf([0.025, 0.975], regret.shape[1] - 1)
            sem = stats.sem(regret / np.expand_dims(np.sqrt(x_vals), axis=1), axis=1)
            lower = y + t_crit[0] * sem
            upper = y + t_crit[1] * sem

            plt.fill_between(x_vals, lower, upper, alpha=0.2)
            plt.ylabel(r'$R(T)/\sqrt{T}$', fontsize=16)
            plt.xscale('log', base=10)
        elif plotdividebyt:
            y = y[100:]
            y = y / x_vals

            # Confidence interval calculation
            t_crit = stats.t.ppf([0.025, 0.975], regret.shape[1] - 1)
            sem = stats.sem(regret[100:] / np.expand_dims(x_vals, axis=1), axis=1)
            lower = y + t_crit[0] * sem
            upper = y + t_crit[1] * sem

            plt.fill_between(x_vals, lower, upper, alpha=0.2)
            plt.ylabel(r'$R(T)/T$', fontsize=16)
            plt.suptitle(f'{dim}D System', y=0.95, fontsize=16)
        else:
            # Confidence interval calculation
            t_crit = stats.t.ppf([0.025, 0.975], regret2.shape[1] - 1)
            sem = stats.sem(regret2, axis=1)
            lower = y + t_crit[0] * sem
            upper = y + t_crit[1] * sem

            plt.fill_between(x_vals, lower, upper, alpha=0.2)
            plt.ylabel('Cumulative Regret', fontsize=16)

        plt.plot(x_vals, y, '-', label=f'{label2} {dim}D')

    plt.xlabel(r'$T$', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.show()

if plotmoderr:
    moderr = np.loadtxt(f'{folder}{dim}D-{name}-moderr{marker}.csv', delimiter=',')
    print(f'Number of Moderr Simulations:    {moderr.shape[1]}')

    plt.figure(figsize=(4, 3))
    x_vals = np.arange(1, len(moderr) + 1)
    y = np.mean(moderr, axis=1)

    # Confidence interval calculation
    t_crit = stats.t.ppf([0.025, 0.975], moderr.shape[1] - 1)
    sem = stats.sem(moderr, axis=1)
    lower = y + t_crit[0] * sem
    upper = y + t_crit[1] * sem

    plt.fill_between(x_vals, lower, upper, alpha=0.2)
    plt.plot(x_vals, y, '-', label=f'{label} {dim}D')

    if comparison:
        moderr2 = np.loadtxt(f'{folder2}{dim}D-{name2}-moderr{marker}.csv', delimiter=',')
        print(f'Number of Moderr Simulations:    {moderr2.shape[1]}')
        y = np.mean(moderr2, axis=1)

        # Confidence interval calculation
        t_crit = stats.t.ppf([0.025, 0.975], moderr2.shape[1] - 1)
        sem = stats.sem(moderr2, axis = 1)
        lower = y + t_crit[0] * sem
        upper = y + t_crit[1] * sem

        plt.fill_between(x_vals, lower, upper, alpha=0.2)
        plt.plot(x_vals, y, '-', label=f'{label2} {dim}D')

    plt.xlabel(r'$k$', fontsize=16)
    plt.ylabel('System Parameter Error', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.show()

if plotrejection:
    rejection = np.loadtxt(f'{folder}{dim}D-{name}-rejection{marker}.csv', delimiter=',')
    print(f'Number of Rejection Simulations: {rejection.shape[1]}')

    plt.figure(figsize=(4, 3))
    x_vals = np.arange(1, len(rejection) + 1)
    y = np.sum(rejection, axis=1) / (np.sum(rejection, axis=1) + rejection.shape[1])

    plt.plot(x_vals, y, '-', label=f'{label} {dim}D')

    if comparison:
        rejection2 = np.loadtxt(f'{folder2}{dim}D-{name2}-rejection{marker}.csv', delimiter=',')
        print(f'Number of Rejection Simulations: {rejection2.shape[1]}')
        y = np.sum(rejection2, axis=1) / (np.sum(rejection2, axis=1) + rejection2.shape[1])

        plt.plot(x_vals, y, '-', label=f'{label2} {dim}D')

    plt.xlabel(r'$k$', fontsize=16)
    plt.ylabel('Sample Rejection Rate', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.show()

if plottraj:
    traj = np.empty((100, regret.shape[0] - 1, dim * 3))
    for i in range(100):
        traj[i] = np.loadtxt(f'{folder}{dim}D_traj/{dim}D-gaussian_mixture-traj{i}.csv', delimiter=',')

    plt.figure(figsize=(4, 3))
    x_vals = np.arange(1, traj.shape[1] + 1)
    y = np.mean(traj[:, :, 0], axis = 0)

    # Confidence interval calculation
    t_crit = stats.t.ppf([0.025, 0.975], traj.shape[0] - 1)
    sem = stats.sem(traj[:, :, 0], axis = 0)
    lower = y + t_crit[0] * sem
    upper = y + t_crit[1] * sem

    plt.fill_between(x_vals, lower, upper, alpha=0.2)
    plt.plot(x_vals, y, '-', label=f'Gaussian mixture {dim}D')
    plt.xlabel(r'$T$', fontsize=16)
    plt.ylabel('Trajectory', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.show()

if computetime:
    time = np.loadtxt(f'{folder}{dim}D-{name}-time{marker}.csv', delimiter=',')
    print(f'Execution Time for {len(time)} Simulations: mean = {np.mean(time):5.4f}, std = {np.std(time):5.4f}')
    if comparison:
        time2 = np.loadtxt(f'{folder2}{dim}D-{name2}-time{marker}.csv', delimiter=',')
        print(f'Execution Time for {len(time2)} Simulations: mean = {np.mean(time2):5.4f}, std = {np.std(time2):5.4f}')