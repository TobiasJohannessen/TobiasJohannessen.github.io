class Box():
    def __init__(self, y, dydx, times, x0, markersize = 10):
        self.y = y
        self.dydx = dydx
        self.times = times  
        self.x0 = x0
        self.xs = None
        self.ts = None
        self.ys = None
        self.events = None
        self.drawing_elements = {}
        self.orientations = None
        self.markersize = 10
    
    def dydx_brach(self, x, delta_x = 0.001):
        f_x = y_brach(x)
        f_x_dx = y_brach(x + delta_x)
        return (f_x_dx - f_x)/delta_x
    
    def _solve_ode(self, event = None):
        y_init = self.y(self.x0)
        x_init = self.x0
        t_init = self.times[0]
        t_end = self.times[-1]
        t_evals = self.times
        event_hit = None

        def dxdt(t,x):
            y_curr = self.y(x)
            if y_init - y_curr < 0:
                return 0
            dxdt = np.sqrt(2*g * (y_init + 0.001 - y_curr))* 1/np.sqrt(1 + dydx_brach(x)**2)

            return dxdt
        
        sol = solve_ivp(
            dxdt,
            t_span=(t_init, t_end),
            y0=[x_init],
            t_eval=t_evals,
            events=event)
        ts = sol.t
        xs = sol.y[0]
        ys = [y_brach(x) for x in xs]
        if event != None:
            event_hit = sol.t_events[0]
        rotations = [np.arctan(dydx_brach(x)) for x in xs]
        return ts, xs, ys, event_hit, rotations
    

    def draw(self, ax, t):

        t_idx = np.argmin(np.abs(self.times - t))
        if t_idx == len(self.times):
            t_idx -= 1
        if self.drawing_elements == {}:
            
            if (t < self.times[0]) or (t > self.times[-1]):
                return None
            self.ts, self.xs, self.ys, self.events, self.orientations = self._solve_ode(self.events)
            self.drawing_elements[ax] = ax.plot(self.xs, self.ys, marker = (4,0,45), markersize = self.markersize)[0]
        
        rotation_rad = self.orientations[t_idx] - np.pi/2
        rotation_deg =  45 + np.degrees(rotation_rad)
        
        adjusted_x = self.xs[t_idx] - self.markersize/14 * np.cos(rotation_rad)
        adjusted_y = self.ys[t_idx] - self.markersize/14 * np.sin(rotation_rad)
        self.drawing_elements[ax].set_data(adjusted_x, adjusted_y)
        self.drawing_elements[ax].set_marker((4,0,rotation_deg))

        return self.drawing_elements[ax]