import os
import matplotlib
import matplotlib.pyplot as plt
from .utils import myplot, multiplot, singleplot, mnplot


def plot_figure(args, plot_iters, Regrets1, Regrets2, KLs1, KLs2, PD_history, Sub1_1, Sub1_2, Sub1_3, Sub2_1, Sub2_2, Sub2_3):
    # Plot 01
    matplotlib.rcParams.update({"font.size": 24})
    plt.rcParams["figure.figsize"] = [40, 16]
    fig, ax = plt.subplots(2, 2)
    myax = ax[0, 0]
    myplot(myax, plot_iters, Regrets1, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Average Random Adversarial Regret for Radar")
    myax.set_ylabel(r"Average Regret")

    myax = ax[0, 1]
    myplot(myax, plot_iters, Regrets2, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Average Random Adversarial Regret for Jammer")
    myax.set_ylabel(r"Average Regret")

    myax = ax[1, 0]
    myplot(myax, plot_iters, KLs1, "log", "log", 1e-4, 1e1)
    myax.title.set_text("KL to Nash (Radar)")
    myax.set_ylabel(r"$KL(x^*, \bar{x}_T)$")

    myax = ax[1, 1]
    myplot(myax, plot_iters, KLs2, "log", "log", 1e-4, 1e1)
    myax.title.set_text("KL to Nash (Jammer)")
    myax.set_ylabel(r"$KL(y^*, \bar{y}_T)$")
    if args.env == "radar":
        plot_path = os.path.join("figures/", args.env+args.dim + "_" + args.alg1 + "_VS_" + args.alg2 + "_" +str(int(args.iter)) + "_regret.png")
    else:
        plot_path = os.path.join("figures/", args.env + "_" + args.alg1 + "_VS_" + args.alg2 + "_" + str(int(args.iter)) + "_regret.png")
    plt.savefig(plot_path)
    plt.close()

    plt.rcParams["figure.figsize"] = [40, 8]
    fig, ax = plt.subplots(1, 3)
    myax = ax[0]
    myplot(myax, plot_iters, PD_history, "log", "linear", 0, 1)
    myax.title.set_text("Probability of detection for radar")
    myax.set_ylabel("Probability")

    myax = ax[1]
    multiplot(myax, plot_iters, Sub1_1, 'blue', '$f_0$', "log", "log", 1e-4, 1e1)
    multiplot(myax, plot_iters, Sub1_2, 'red', '$f_1$', "log", "log", 1e-4, 1e1)
    multiplot(myax, plot_iters, Sub1_3, 'green', '$f_2$', "log", "log", 1e-4, 1e1)
    myax.legend(loc='best')
    myax.grid(True)
    myax.title.set_text("Subpulse-selection policy for radar")
    myax.set_ylabel("Probability")
    myax.set_ylim(0, 1)
    myax.set_xlim(left=1)
    myax.set_xscale("log")

    myax = ax[2]
    multiplot(myax, plot_iters, Sub2_1, 'blue', '$f_0$', "log", "log", 1e-4, 1e1)
    multiplot(myax, plot_iters, Sub2_2, 'red', '$f_1$', "log", "log", 1e-4, 1e1)
    multiplot(myax, plot_iters, Sub2_3, 'green', '$f_2$', "log", "log", 1e-4, 1e1)
    myax.legend(loc='best')
    myax.grid(True)
    myax.title.set_text("Subpulse-selection policy for jammer")
    myax.set_ylabel("Probability")
    myax.set_ylim(0, 1)
    myax.set_xlim(left=1)
    myax.set_xscale("log")
    # save figure
    if args.env == "radar":
        plot_path = os.path.join("figures/", args.env+args.dim + "_" + args.alg1 + "_VS_" + args.alg2 + "_" +str(int(args.iter)) + "_pd.png")
    else:
        plot_path = os.path.join("figures/", args.env + "_" + args.alg1 + "_VS_" + args.alg2 + "_" + str(int(args.iter)) + "_pd.png")
    plt.savefig(plot_path)
    plt.close()


def plot_figure_single(plot_iters, Regret1, E_Regret1, KL1, Dual_Gap, plot_path):
    matplotlib.rcParams.update({"font.size": 24})
    plt.rcParams["figure.figsize"] = [40, 16]
    fig, ax = plt.subplots(2, 2)
    myax = ax[0, 0]
    singleplot(myax, plot_iters, Regret1, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Average Random Adversarial Regret")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[0, 1]
    singleplot(myax, plot_iters, E_Regret1, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Average Expected Adversarial Regret")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 0]
    singleplot(myax, plot_iters, KL1, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"KL to Nash")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 1]
    singleplot(myax, plot_iters, Dual_Gap, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Duality Gap")
    # myax.set_ylabel(r"Average Regret")

    plt.savefig(plot_path)
    plt.close()

def plot_figure_mseed(plot_iters, Regret1, E_Regret1, KL1, Dual_Gap, plot_path):
    matplotlib.rcParams.update({"font.size": 24})
    plt.rcParams["figure.figsize"] = [40, 16]
    fig, ax = plt.subplots(2, 2)
    myax = ax[0, 0]
    myplot(myax, plot_iters, Regret1, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Average Random Adversarial Regret")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[0, 1]
    myplot(myax, plot_iters, E_Regret1, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Average Expected Adversarial Regret")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 0]
    myplot(myax, plot_iters, KL1, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"KL to Nash")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 1]
    myplot(myax, plot_iters, Dual_Gap, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Duality Gap")
    # myax.set_ylabel(r"Average Regret")

    plt.savefig(plot_path)
    plt.close()

def plot_figure_compare(plot_iters, alg1_name, alg2_name, Regret1, E_Regret1, KL1, Dual_Gap1, Regret2, E_Regret2, KL2, Dual_Gap2, plot_path):
    matplotlib.rcParams.update({"font.size": 24})
    plt.rcParams["figure.figsize"] = [40, 16]
    fig, ax = plt.subplots(2, 2)
    myax = ax[0, 0]
    multiplot(myax, plot_iters, Regret1, 'b', alg1_name, "log", "log", 1e-4, 1e1)
    multiplot(myax, plot_iters, Regret2, 'r', alg2_name, "log", "log", 1e-4, 1e1)
    myax.title.set_text(r"Average Random Adversarial Regret")
    myax.legend(loc="best")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[0, 1]
    multiplot(myax, plot_iters, E_Regret1, 'b', alg1_name, "log", "log", 1e-4, 1e1)
    multiplot(myax, plot_iters, E_Regret2, 'r', alg2_name, "log", "log", 1e-4, 1e1)
    myax.legend(loc="best")
    myax.title.set_text(r"Average Expected Adversarial Regret")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 0]
    multiplot(myax, plot_iters, KL1, 'b', alg1_name, "log", "log", 1e-5, 1e1)
    multiplot(myax, plot_iters, KL2, 'r', alg2_name, "log", "log", 1e-5, 1e1)
    myax.legend(loc="best")
    myax.title.set_text(r"KL to Nash")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 1]
    multiplot(myax, plot_iters, Dual_Gap1, 'b', alg1_name, "log", "log", 1e-4, 1e1)
    multiplot(myax, plot_iters, Dual_Gap2, 'r', alg2_name, "log", "log", 1e-4, 1e1)
    myax.legend(loc="best")
    myax.title.set_text(r"Duality Gap")
    # myax.set_ylabel(r"Average Regret")

    plt.savefig(plot_path)
    plt.close()

def plot_figure_dim(mn, Regret1, E_Regret1, KL1, Dual_Gap, plot_path):
    matplotlib.rcParams.update({"font.size": 30})
    plt.rcParams["figure.figsize"] = [40, 50]
    fig, ax = plt.subplots(2, 2)
    myax = ax[0, 0]
    mnplot(myax, mn, Regret1, "log", "log", 1e-4, 1e-1)
    myax.title.set_text(r"Average Random Adversarial Regret")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[0, 1]
    mnplot(myax, mn, E_Regret1, "log", "log", 1e-4, 1e-1)
    myax.title.set_text(r"Average Expected Adversarial Regret")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 0]
    mnplot(myax, mn, KL1, "log", "log", 1e-4, 1e-1)
    myax.title.set_text(r"KL to Nash")
    # myax.set_ylabel(r"Average Regret")

    myax = ax[1, 1]
    mnplot(myax, mn, Dual_Gap, "log", "log", 1e-4, 1e-1)
    myax.title.set_text(r"Duality Gap")
    # myax.set_ylabel(r"Average Regret")

    plt.savefig(plot_path)
    plt.close()
