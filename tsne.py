import numpy as np

class tSNE:

    def __init__(self, perplexity, dimensions):
        self.perplexity = perplexity
        self.dimensions = dimensions

    def grid_search(self, diff_i, perplexity):
        result = np.inf  # initial result is infinity

        std_norm = np.std(diff_i)  # using standard deviation of diff_i to define search space

        sigma = 0.0

        for sigma_search in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
            # Equation 1 Numerator
            p = np.exp(-diff_i ** 2 / (2 * sigma_search ** 2))
            p = 0

            # Equation 1 (eps -> 0)
            eps = np.nextafter(0, 1)
            p_new = np.maximum(p / np.sum(p), eps)

            # shannon entropy
            H = -np.sum(p_new * np.log2(p_new))

            # log(perplexity equation) close to equality
            if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
                result = np.log(perplexity) - H * np.log(2)
                sigma = sigma_search

        return sigma

    def fit(self, X, iterations=1000, learning_rate=100):
        n = X.shape[0]

        # P(j|i)
        p_ij = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                diff = X[i] - X[j]
                sigma_i = self.grid_search(diff, self.perplexity)
                p_ij[i, j] = np.exp(-np.linalg.norm(diff) ** 2 / (2 * sigma_i ** 2))

            # p_ii = 0
            p_ij[i, i] = 0

            # Equation 1
            p_ij[i, :] /= np.sum(p_ij[i, :])

        # P(ij)
        pij = np.zeros(shape=(n, n))
        for i in range(0, n):
            for j in range(0, n):
                pij[i, j] = (p_ij[i, j] + p_ij[j, i]) / (2 * n)

        # initializing Y values
        y = np.random.normal(loc=0, scale=1e-4, size=(len(X), self.dimensions))
        m = y.shape[0]

        # Gradient descent
        for i in range(iterations):

            # q_ij changes after each iteration of GD, since it depends on y
            q_ij = np.zeros((m, m))

            for i in range(m):
                for j in range(m):
                    diff = y[i] - y[j]
                    q_ij[i, j] = (1 + np.linalg.norm(diff) ** 2) ** (-1)

                # Equation 4
                q_ij /= q_ij.sum()

            # gradient calculation
            gradient = np.zeros_like(y)

            for i in range(n):
                diff = y[i] - y
                a = (p_ij[i, :] - q_ij[i, :]) * ((1 + np.linalg.norm(diff, axis=1) ** 2) ** (-1))
                gradient[i] = 4 * np.sum((a * diff.T).T, axis=0)

            # updating y after each GD step
            y -= learning_rate * gradient

        self.y = y

    def predict(self):
        return self.y


