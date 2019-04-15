from matplotlib import pyplot as plt


def plotting(data, mode="training"):
    """

    data: list containing log likelihoods, aer
    mode: "training" or "validation"

    """
    plt.plot(range(len(data)), data)
    if mode == "training":
        plt.xlabel("iteration")
        plt.ylabel("log likelihood")
    elif mode == "validation":
        plt.xlabel("iteration")
        plt.ylabel("aer")
