

#TO DO: Add a class for Kernel Regression
#Make it nicer to use kernels with different hyperparameters instead of choosing them outside.

class Regression():

    def __init__(self, xs, ys, x_range = [0, 10]):
        self.betas = None
        self.x_range = x_range
        self.xs = xs
        self.ys = ys

    def regression(self, new = False):
        if self.betas is None or new:
            identity = np.eye(self.order + 1)
            identity[0, 0] = 0
            self.betas = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.ys
        return self.betas

    def plot(self, ax, x_samples,  color = 'C1', solid_line = True, alpha = 1):
        if solid_line:
            xs = np.linspace(*self.x_range, 100)
            plot_space = np.array([self.predict(x) for x in xs])            
            ax.plot(xs, plot_space, color = color, alpha = alpha)
        plot_samples = np.array([self.predict(x) for x in x_samples])
        ax.scatter(x_samples, plot_samples, color = color, label = self.type + ' Regression', alpha = alpha)
        return ax

    def _build_design_matrix(self, x, order = 10):
        X = np.array([x**(i) for i in range(order + 1)]).T
        return X


class LinearRegression(Regression):

    def __init__(self, xs, ys, **kwargs):
        super().__init__(xs, ys, **kwargs)
        self.order = 1
        self.X = np.array([np.ones(len(xs)), xs]).T
        self.type = 'Linear'


    def predict(self, x):
        if self.betas is None:
            self.regression()
        return self.betas[0] + self.betas[1] * x


    

class QuadraticRegression(Regression):

    def __init__(self, xs, ys, **kwargs):
        super().__init__(xs, ys, **kwargs)
        self.order = 2
        self.X = np.array([np.ones(len(xs)), xs, xs**2]).T
        self.type = 'Quadratic'


    def predict(self, x):
        if self.betas is None:
            self.regression()
        return self.betas[0] + self.betas[1] * x + self.betas[2] * x**2



class PolynomialRegression(Regression):

    def __init__(self, xs, ys, order = 3, **kwargs):
        super().__init__(xs, ys, **kwargs)
        self.order = order
        self.X = np.array([xs**(i) for i in range(order + 1)]).T
        if order == 1:
            self.type = 'Linear'
        elif order == 2:
            self.type = 'Quadratic'
        elif order == 3:
            self.type = 'Cubic'
        else:
            self.type = f'{order}th Order'


    def predict(self, x):
        if self.betas is None:
            self.regression()
        return sum([beta * x**i for i, beta in enumerate(self.betas)])



class RidgeRegression(Regression):

    def __init__(self, xs, ys, order = 3, lmb = 10, x_range = [0, 10], **kwargs):
        self.xs = xs.astype(float)
        self.ys = ys
        self.lmb = lmb
        self.order = order
        self.betas = None
        self.x_range = x_range

        #Construct the design matrix
        self.X = self._build_design_matrix(xs, order)
       
        if order == 1:
            self.type = 'Linear Ridge'
        elif order == 2:
            self.type = 'Quadratic Ridge'
        elif order == 3:
            self.type = 'Cubic Ridge'
        else:
            self.type = f'{order}th Order Ridge'


    def regression(self, new = False):
        if self.betas is None or new:
            identity = np.eye(self.order + 1)
            #identity[0, 0] = 0
            first_term = np.linalg.inv((self.X.T @ self.X) + self.lmb * identity)
            second_term = self.X.T @ self.ys
            self.betas = first_term @ second_term
           
        return self.betas

    def predict(self, x):
        if self.betas is None:
            self.regression()
        return np.sum(beta * x**i for i, beta in enumerate(self.betas))

    def plot(self, ax, x_samples,  color = 'C1', solid_line = False, alpha = 1):
        if solid_line:
            xs = np.linspace(*self.x_range, 100)
            ax.plot(xs, self.predict(xs), color = color, alpha = alpha)
        ax.scatter(x_samples, self.predict(x_samples), color = color, label = self.type + rf' Regression $\lambda = ${self.lmb:.2g}', alpha = alpha)
        return ax

    



class RidgeRegression2:
    def __init__(self, xs, ys, degree=3, lmb=1.0, scale=False):
        """
        Initializes the RidgeRegression class.

        :param xs: Training data input (1D array).
        :param ys: Training data output (1D array).
        :param degree: The degree of the polynomial for the feature transformation.
        :param lmb: The regularization parameter (lambda) for ridge regression.
        :param scale: Whether to apply feature scaling (standardization).
        """
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.degree = degree
        self.lmb = lmb
        self.scale = scale
        self.betas = None

        # Build the design matrix
        self.X = self._build_design_matrix(self.xs)

        # Apply scaling if required
        if self.scale:
            self.mean = np.mean(self.X[:, 1:], axis=0)
            self.std = np.std(self.X[:, 1:], axis=0)
            self.X[:, 1:] = (self.X[:, 1:] - self.mean) / self.std
        
        # Fit the model (compute betas)
        self.regression()

    def _build_design_matrix(self, x):
        """
        Builds the design matrix for polynomial regression.
        :param x: Input data (1D array).
        :return: Design matrix (2D array).
        """
        return np.array([x**i for i in range(self.degree + 1)])

    def regression(self):
        """
        Performs ridge regression to compute the coefficients (betas).
        """
        identity = np.eye(self.X.shape[1])
        #identity[0, 0] = 0  # Do not regularize the bias term

        # Perform ridge regression to find betas
        self.betas = np.linalg.inv(self.X.T @ self.X + self.lmb * identity) @ self.X.T @ self.ys

    def predict(self, x):
        """
        Makes predictions using the learned model.

        :param x: New input data (1D array or scalar).
        :return: Predicted values.
        """
        if np.isscalar(x):
            x = np.array([x])

        # Build the design matrix for new data
        X_new = self._build_design_matrix(x)

        

        # Return the predictions
        return X_new @ self.betas

    def plot(self, x_samples, ax=None):
        """
        Plots the original data points along with the fitted regression curve.

        :param x_samples: The points to plot the regression curve.
        :param ax: Optional axis object for plotting.
        :return: The axis object.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()

        # Plot original data
        ax.scatter(self.xs, self.ys, color='blue', label='Data Points')

        # Plot the fitted curve
        xs = np.linspace(min(self.xs), max(self.xs), 100)
        ax.plot(xs, self.predict(xs), color='red', label=f'Ridge Regression (lambda={self.lmb})')

        ax.legend()
        return ax


class KernelRegression(Regression):


    def __init__(self, xs, ys, *args ,bandwidth = 1, degree = 2,kernel = '', lmb = 1, reg_type = '', custom_kernel = None, **kwargs):
        self.xs = xs
        self.ys = ys
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.lmb = lmb
        self.alphas = None
        self.K = None
        self.type = reg_type
        self.degree = degree
        self.custom_kernel = custom_kernel
    
        super().__init__(xs, ys, *args, **kwargs)


    def poly_kernel(self, xi, xj):
        return (1 + xi * xj)**self.degree

    def gauss_kernel(self, xi, xj):
        return np.exp(-0.5 * (xi - xj)**2 / self.bandwidth**2)

    def kernel_func(self, xi, xj):
        if self.custom_kernel is not None:
            return self.custom_kernel(xi, xj)
        
        elif self.kernel == 'poly':
            return self.poly_kernel(xi, xj)
        elif self.kernel == 'gauss':
            return self.gauss_kernel(xi, xj)
        else:
            raise ValueError('Invalid kernel type. Choose poly or gauss')

    def kernel_type(self, type):
        self.kernel = type
        return
    
    def kernel_matrix(self):
        N = len(self.xs)
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K[j, i] = self.kernel_func(self.xs[i], self.xs[j])
        self.K = K
        return self.K

    def _kernel_vector(self, x_star, xs):
        N = len(xs)
        K_vec = np.zeros(N)
        for i in range(N):
            K_vec[i] = self.kernel_func(xs[i], x_star)
        return K_vec
        

    def regression(self):
        if self.alphas is None:
            self.K = self.kernel_matrix()
            self.alphas = np.linalg.inv(self.K + self.lmb * np.eye(len(self.xs))) @ self.ys
        return self.alphas

    def predict(self, x):
        if self.alphas is None:
            self.regression()
        K_vec = self._kernel_vector(x_star = x, xs = self.xs)
        return np.dot(K_vec,self.alphas)



class DiscreteRBF():
    
    def __init__(self, color='C1',xwidth=0.01,method='midpoint', sigma_kernel = 0.1):
        self.color = color
        self.xwidth = xwidth
        self.sigma_kernel = sigma_kernel
        self.sigma_phi = self.sigma_kernel / np.sqrt(2)
        self.bin_edges = np.arange(-12,12.01,self.xwidth)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) /2
        self.method = method
        self.A = 1/np.sqrt(self.xwidth)*np.sqrt(np.sqrt(2*np.pi)*self.sigma_kernel) 
    
    def _gauss_integral(self,x):
        return (1+erf(x/np.sqrt(2)/self.sigma_phi))/2

    def descriptor(self,pos):
        vals = 1/np.sqrt(2*np.pi)/self.sigma_phi*\
                np.exp(-(self.bin_centers-pos)**2/2/self.sigma_phi**2)*self.xwidth
        return vals*self.A
    
    def draw(self,pos,ax,color=None):
        if color is None:
            color = self.color
        vector = self.descriptor(pos)
        
        ax.bar(self.bin_centers,vector,width=0.8 * self.xwidth)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel(r'$x_i$')
        ax.set_ylabel(r'$\phi_i(x)$')
