import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from sympy import symbols, solve, Eq


class Box():
    def __init__(self,y, dydx, times, x0, markersize = 10, wall = 30):
        self.y = y

        if dydx == None:
            dydx = self.dydx_finite_difference
        self.dydx = dydx
        self.times = times  
        self.x0 = x0
        self.xs = None
        self.ts = None
        self.ys = None
        self.wall = wall
        self.events = None
        
        
        self.drawing_elements = {}
        self.orientations = None
        self.markersize = 10
        self.g = 9.81


    def dydx_finite_difference(self, x, delta_x = 0.001):
        f_x = self.y(x)
        f_x_dx = self.y(x + delta_x)
        return (f_x_dx - f_x)/delta_x

    def dxdt(self, t,x):
        y_curr = self.y(x)
        y_init = self.y(self.x0)
        if y_init - y_curr < 0:    
            return 0
        dxdt = np.sqrt(2*self.g * (y_init - y_curr))* 1/np.sqrt(1 + self.dydx(x)**2)

        
        return dxdt

    def event_hit(self, t, x):
        return np.array(x) - self.wall
    
    
    def _solve_ode(self):

        t_init = self.times[0]
        t_end = self.times[-1]
        t_evals = self.times
        event_hit = self.event_hit

        
    
        
        sol = solve_ivp(
            self.dxdt,
            t_span=(t_init, t_end),
            y0=[self.x0 + 1e-3],
            t_eval=t_evals,
            events= event_hit)
        ts = sol.t
        xs = sol.y[0]
        ys = [self.y(x) for x in xs]
        if self.events != None:
            event_hit = sol.t_events[0]
        rotations = [np.arctan(self.dydx(x)) for x in xs]
        return ts, xs, ys, event_hit, rotations
    

    def draw(self, ax, t):

        t_idx = np.argmin(np.abs(self.times - t))
        if t_idx == len(self.times):
            t_idx -= 1
        if self.drawing_elements == {}:
            
            if (t < self.times[0]) or (t > self.times[-1]):
                return None
            self.ts, self.xs, self.ys, self.events, self.orientations = self._solve_ode()
            self.drawing_elements[ax] = ax.plot(self.xs, self.ys, marker = (4,0,45), markersize = self.markersize)[0]
        
        rotation_rad = self.orientations[t_idx] - np.pi/2
        rotation_deg =  45 + np.degrees(rotation_rad)
        
        adjusted_x = self.xs[t_idx]# - self.markersize/14 * np.cos(rotation_rad)
        adjusted_y = self.ys[t_idx]# - self.markersize/14 * np.sin(rotation_rad)
        self.drawing_elements[ax].set_data(adjusted_x, adjusted_y)
        self.drawing_elements[ax].set_marker((4,0,rotation_deg))

        return self.drawing_elements[ax]



class Brachistochrone(Box):

    def __init__(self, times, x0 = 0, markersize = 10, x_limits = [0, 30], y_limits = [22.5, 0], wall = 30):


        phi_1 = fsolve(lambda phi: (np.cos(phi) - 1)/(phi - np.sin(phi)) - (y_limits[1] - y_limits[0])/(x_limits[1] - x_limits[0]), 0.5)[0]
        self.a_brach = x_limits[1]/(phi_1 - np.sin(phi_1))
        self.y_0_brach = y_limits[1] - self.a_brach * (1 + np.cos(phi_1))
        self.y = self.y_brach
        self.dydx = self.dydx_brach
        super().__init__(y = self.y_brach, dydx = self.dydx_brach, times = times, x0 = x0, markersize = markersize, wall = wall)


   

    def dydx_brach(self, x, delta_x = 0.001):
        f_x = self.y_brach(x)
        f_x_dx = self.y_brach(x + delta_x)
        return (f_x_dx - f_x)/delta_x

    def y_brach(self, x):
        phi_x = fsolve(lambda phi: self.a_brach * (phi - np.sin(phi)) - x, 0)[0]
        
        return self.y_0_brach + self.a_brach * (1 + np.cos(phi_x))



class Linear(Box):

    def __init__(self, times, x0, markersize = 10, x_limits = [0, 30], y_limits = [22.5, 0], wall = 30):
        
        self.slope = (y_limits[1] - y_limits[0])/(x_limits[1] - x_limits[0])
        self.y_0_linear = y_limits[0]
        self.y = self.y_linear
        self.dydx = self.dydx_linear
        super().__init__(y = self.y, dydx =self.dydx, times = times, x0 = x0, markersize = markersize, wall = wall)

    def y_linear(self, x):
        return self.y_0_linear + self.slope * x


    def dydx_linear(self, x):
        return self.slope


class Parabolic(Box):

    def __init__(self, times, x0, markersize = 10, x_limits = [0, 30], y_limits = [22.5, 0], roots = [10, 30], wall = 30):
        a, b = symbols('a b')
        sol = solve([Eq(a*roots[0]**2 + b*roots[0] + y_limits[0], 0), Eq(a*roots[1]**2 + b*roots[1] + y_limits[0], 0)], (a, b))
        self.a, self.b = float(sol[a]), float(sol[b])      
        self.c = y_limits[0]
        self.y = self.y_parabola
        self.dydx = self.dydx_parabola
        super().__init__(y = self.y, dydx = self.dydx, times = times, x0 = x0, markersize = markersize, wall = wall)

    def y_parabola(self, x):
        return self.a * x**2 + self.b * x + self.c

    def dydx_parabola(self, x):
        return 2 * self.a * x + self.b