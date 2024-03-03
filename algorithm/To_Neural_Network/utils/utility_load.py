import numpy as np


def load_utility(game, para, a0, a1, h):
    # if game.seed == 1:
    #     # No noise for counter example
    #     u0 = game.u(h, 0)
    #     u1 = -u0

    if para.algo == "EWF":
        if para.game == "RadarEnv01":
            u0 = game._u0_matrix[:, a1]
            u1 = -game._u0_matrix[a0]
            u0 = u0 - 1
        else:
            u0 = game._u0_matrix[:, a1]
            u1 = -game._u0_matrix[a0]
            # u0  = (u0 - 1) / 2
            # u1  = (u1 - 1) / 2

    elif para.algo == "RMF" or para.algo == "RMPF":
        if para.game == "RadarEnv01":
            u0 = game._u0_matrix[:, a1]
            u1 = -game._u0_matrix[a0,:]
            u0 = u0 - 1
        else:
            u0 = game._u0_matrix[:, a1]
            u1 = -game._u0_matrix[a0]
            # u0 = (u0 - 1) / 2
            # u1 = (u1 - 1) / 2

    elif para.algo == "OTSF" or para.algo == "OCBF":
        u0 = game._u0_matrix[:, a1]
        u1 = -game._u0_matrix[a0]
        noise_vec = np.random.normal(0, np.sqrt(0.1), u0.shape)
        u0 = u0 + noise_vec
    else:
        # bernoulli setting: [0, 1]
        u0 = game.u(h, 0)
        ## u0_mean = u0
        ## u0 = np.random.binomial(1, u0_mean)
        ## u1 = -u0
        # u0_mean = (u0 + 1) / 2
        # u0 = np.random.binomial(1, u0_mean)
        # u0 = 2 * u0 - 1
        # u1 = -u0
        noise = np.random.normal(0, np.sqrt(0.1))
        u0 = u0 + noise
        # u0 = np.clip(u0, -1, 1)
        u1 = -u0

        if para.algo == "Exp3Auto":
            u0 = (u0 - 1) / 2
            u1 = (u1 - 1) / 2

    return u0, u1