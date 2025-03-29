import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize, Bounds
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
import torch.nn.functional as F

class Inhomogenous_Poisson:
    def __init__(self, kernel, T_max, M):
        self.T_max   = T_max
        self.kernel  = kernel
        self.M       = M
    
    def simulate(self):
        P = []
        t = 0
        while t < self.T_max:
            Exp = np.random.exponential(scale = 1.0 / self.M)
            t  += Exp
            if t < self.T_max and np.random.uniform() <= self.kernel(t)/self.M:
                P.append(t)
        return P
    
    def visu_event(self, nb_times, nb_points):
        times = np.linspace(0, self.T_max, nb_points)
        dt = times[1]-times[0]
        N_t = np.zeros(len(times))
        for i in range (nb_times):
            ts = self.simulate()
            N_t += np.array([np.sum(ts <= t) for t in times])
        N_t = N_t / nb_times
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = times, y = N_t, mode='markers', name="Average Number of events", marker=dict(size = 0.85, color = 'darkred')))
        fig.add_trace(go.Scatter(x = times, y = np.cumsum(self.kernel(times)*dt), mode='lines', name="Integral of the intensity function", line=dict(width = 0.85, color = 'black')))
        fig.update_layout(
                title="Average Number of events",
                xaxis_title="Time",
                yaxis_title="Average Number of events",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()


class Hawkes_process:
    def __init__(self, T_max, M_max = 1, lambda_0 = 1, kernel = lambda x: np.exp(-x), kernel_type = 'Exp', alpha_exp = 1, beta_exp = 1, c_pl = 1, p_pl = 1, k_pl = 1):
        self.T_max    = T_max
        
        self.lambda_0 = lambda_0
        self.M_max    = M_max
        
        # Exponential Kernel
        if kernel_type == 'Exp':
            self.kernel = lambda x: alpha_exp*np.exp(-beta_exp*x)
        self.alpha = alpha_exp
        self.beta  = beta_exp
        
        # Power Law Kernel
        if kernel_type == 'Power_Law':
            self.kernel = lambda x: k_pl/(c_pl+x)**p_pl
        self.c = c_pl
        self.p = p_pl
        self.k = k_pl
        self.kernel = kernel
        
    def simulate_by_thinning(self, eps):
        P = [np.random.exponential(scale = 1.0/self.lambda_0)]
        t = P[0]
        def intensity(x, past_events):
            return self.lambda_0 + sum(self.kernel(x - ti) for ti in past_events)
        while t < self.T_max:
            M = intensity(t+eps, P)
            Exp = np.random.exponential(scale = 1.0 / M)
            t  += Exp
            if t < self.T_max and np.random.uniform() <= intensity(t, P)/M:
                P.append(t)
        return P
    
    def simulate_by_thinning_exponential(self):
        t0 = np.random.exponential(scale=1.0/self.lambda_0)
        P = [t0]
        t = t0
        contrib = 0.0

        while t < self.T_max:
            M = self.lambda_0 + self.alpha * (contrib + 1)
            dt = np.random.exponential(scale=1.0 / M)
            t_new = t + dt
            if t_new > self.T_max:
                break
            contrib *= np.exp(-self.beta * dt)
            intensity_t = self.lambda_0 + self.alpha * (contrib + 1)
            if np.random.uniform() <= intensity_t / M:
                contrib += 1
                P.append(t_new)
            t = t_new
        return P
    
    def simulate_by_thinning_power_law(self, eps = 0):
        P = [np.random.exponential(scale = 1.0/self.lambda_0)]
        t = P[0]
        def intensity(x, past_events):
            return self.lambda_0 + sum(self.kernel(x - ti) for ti in past_events)
        while t < self.T_max:
            M = intensity(t+eps, P)
            Exp = np.random.exponential(scale = 1.0 / M)
            t  += Exp
            if t < self.T_max and np.random.uniform() <= intensity(t, P)/M:
                P.append(t)
        return P

    def simulate_by_cluster(self, n_ite):
        nb_parents = np.random.poisson(self.lambda_0 * self.T_max)
        parents    = np.sort(np.random.uniform(0, self.T_max, size=nb_parents))
        all_events = list(parents)
        current_generation = parents
        for gen in range(n_ite):
            new_generation = []
            for t_p in current_generation:
                if t_p >= self.T_max:
                    continue
                T_remain = self.T_max - t_p
                process = Inhomogenous_Poisson(self.kernel, T_remain, self.M_max)
                children_rel = process.simulate()
                children_abs = [t_p + cr for cr in children_rel]
                new_generation.extend(children_abs)
            if len(new_generation) == 0:
                break
            all_events.extend(new_generation)
            current_generation = new_generation
        all_events.sort()
        return all_events

    def visu_intensity(self, nb_points, method):
        eps    = 1e-14
        n_ite  = 40
        if method == 'Thinning':
            events = self.simulate_by_thinning(eps)
        if method == 'Cluster':
            events = self.simulate_by_cluster(n_ite)
        if method == 'Exp':
            events = self.simulate_by_thinning_exponential()
        if method == "Power_Law":
            print('(Power_law)')
            events = self.simulate_by_thinning_power_law()
        print(len(events))
        times  = [0, events[0] - eps]
        
        def intensity(x, past_events):
            return self.lambda_0 + sum(self.kernel(x - ti) for ti in past_events)
        intens = [intensity(0, []), intensity(times[1],[])]
        
        for i in range (len(events)-1):
            t       = np.linspace(events[i], events[i+1]-eps, nb_points).tolist()
            inte    = [intensity(t[k], events[:i+1]) for k in range (len(t))]
            intens += inte
            times  += t
            
        times.append(events[-1])
        t       = np.linspace(events[-1], self.T_max-eps, nb_points).tolist()
        inte    = [intensity(t[k], events) for k in range (len(t))]
        intens += inte
        times  += t
        
        fig     = go.Figure()
        fig.add_trace(go.Scatter(x = times, y = intens, mode='lines', name="Intensity", line=dict(width = 0.85, color = 'darkblue')))
        fig.update_layout(
                title=f"Hawkes Process simulation via {method} kernel",
                xaxis_title="Time",
                yaxis_title="Intensity Function",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()
        
    def visu_time(self, method):
        if method == 'Exp':
            times = self.simulate_by_thinning_exponential()
        if method == 'Thining':
            times = self.simulate_by_thinning()
        if method == 'Power_Law':
            times = self.simulate_by_thinning_power_law()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=np.arange(len(times)), mode='lines', name="Number of events", line=dict(width = 0.85, color = 'darkred')))
        fig.update_layout(
                title="Number of events",
                xaxis_title="Time",
                yaxis_title="Number of events",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()
        
    def exponential_estimation(self, reg_factor = 0):
        '''
        Estimation via ML for exponential kernels alpha * exp(-beta * t)
        '''
        times = self.simulate_by_thinning_exponential()
        T_org = times[-1]
        
        times = np.array(times)
        times = times/T_org
        T = times[-1]
        
        def log_likelihood(lambda_, beta_, alpha_):
            A = [0.0]
            for i in range(1, len(times)):
                dt = times[i] - times[i-1]
                A.append(np.exp(-beta_ * dt) * (A[-1] + 1))
            A = np.array(A)
            term1 = np.sum(np.log(lambda_ + alpha_ * A))
            term2 = lambda_ * T + (alpha_ / beta_) * np.sum(1 - np.exp(-beta_ * (T - np.array(times))))
            reg_term = reg_factor * alpha_ ** 2
            return term1 - term2 - reg_term
        
        def gradient_log_likelihood(lambda_, beta_, alpha_):
            n = len(times)
            A = [0.0]
            B = [0.0]
            for i in range(1, n):
                dt = times[i] - times[i-1]
                Ai = np.exp(-beta_ * dt) * (A[-1] + 1)
                A.append(Ai)
                Bi = np.exp(-beta_ * dt) * (B[-1] - dt * (A[-2] + 1))
                B.append(Bi)
            A = np.array(A)
            B = np.array(B)
            dL_dlambda = np.sum(1.0 / (lambda_ + alpha_ * A)) - T
            dL_dalpha = np.sum(A / (lambda_ + alpha_ * A)) - (1.0 / beta_) * np.sum(1 - np.exp(-beta_ * (T - times))) - 2 * reg_factor * alpha_
            grad_term1_beta = np.sum(alpha_ * B / (lambda_ + alpha_ * A))
            sum_term = np.sum(1 - np.exp(-beta_ * (T - times)))
            sum_term2 = np.sum((T - times) * np.exp(-beta_ * (T - times)))
            grad_term2_beta = -alpha_ / (beta_ ** 2) * sum_term + (alpha_ / beta_) * sum_term2
            dL_dbeta = grad_term1_beta - grad_term2_beta
            return np.array([dL_dlambda, dL_dbeta, dL_dalpha])


        def objective(params):
            lambda_, beta_, alpha_ = params
            return -log_likelihood(lambda_, beta_, alpha_)
        
        def grad_objective(params):
            lambda_, beta_, alpha_ = params
            return -gradient_log_likelihood(lambda_, beta_, alpha_)
        
        n_events = len(times)
        
        observed_intensity = n_events / T
        alpha_init = 0.5
        lambda_init = observed_intensity * (1 - alpha_init)
        beta_init = 1.0
        x0 = np.array([lambda_init, beta_init, alpha_init])
        bounds = Bounds([1e-9, 1e-9, 1e-9], [np.inf, np.inf, np.inf])
        result = minimize(objective, x0, jac=grad_objective, method='L-BFGS-B', bounds=bounds, options={"disp": False, "ftol": 1e-10, "gtol": 1e-8})
        lambda_opt, beta_opt, alpha_opt = result.x
        f_opt = -result.fun
        lambda_opt = lambda_opt/T_org
        beta_opt = beta_opt/ T_org
        alpha_opt = alpha_opt/ T_org
        print("Optimal lambda   =", lambda_opt)
        print("Optimal beta     =", beta_opt)
        print("Optimal alpha    =", alpha_opt)
        print("Maximum log-like =", f_opt)
        
        def kernel_estimate(x):

            return alpha_opt * np.exp(-beta_opt * x)
        
        x = np.linspace(0, 1, 10000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = x, 
            y = kernel_estimate(x),
            mode = 'markers',
            name = "Estimated Kernel",
            marker = dict(size = 0.85, color = 'darkred')
        ))
        fig.add_trace(go.Scatter(
            x = x,
            y = self.kernel(x),
            mode = 'lines',
            name = "Real Kernel",
            line = dict(width = 0.85, color = 'black')
        ))
        fig.update_layout(
            title = "Estimated vs real kernel function",
            xaxis_title = "Time",
            yaxis_title = "Kernel Function",
            plot_bgcolor = '#D3D3D3',
            paper_bgcolor = '#D3D3D3',
            xaxis = dict(showgrid=True, gridcolor='#808080'),
            yaxis = dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
        
        return alpha_opt, beta_opt, lambda_opt

    def fit_exponential(self, times, lambda_init = 1, reg_factor  = 0):
        '''
        Estimation via ML for exponential kernels alpha * exp(-beta * t)
        '''
        T_org = times[-1]
        
        times = np.array(times)
        times = times/T_org
        T = times[-1]
        def log_likelihood(lambda_, beta_, alpha_):
            A = [0.0]
            for i in range(1, len(times)):
                dt = times[i] - times[i-1]
                A.append(np.exp(-beta_ * dt) * (A[-1] + 1))
            A = np.array(A)
            term1 = np.sum(np.log(lambda_ + alpha_ * A))
            term2 = lambda_ * T + (alpha_ / beta_) * np.sum(1 - np.exp(-beta_ * (T - np.array(times))))
            reg_term = reg_factor * alpha_ ** 2
            return term1 - term2 - reg_term

        def gradient_log_likelihood(lambda_, beta_, alpha_):
            n = len(times)
            A = [0.0]
            B = [0.0]
            for i in range(1, n):
                dt = times[i] - times[i-1]
                Ai = np.exp(-beta_ * dt) * (A[-1] + 1)
                A.append(Ai)
                Bi = np.exp(-beta_ * dt) * (B[-1] - dt * (A[-2] + 1))
                B.append(Bi)
            A = np.array(A)
            B = np.array(B)
            dL_dlambda = np.sum(1.0 / (lambda_ + alpha_ * A)) - T
            dL_dalpha = np.sum(A / (lambda_ + alpha_ * A)) - (1.0 / beta_) * np.sum(1 - np.exp(-beta_ * (T - times))) - 2 * reg_factor * alpha_
            grad_term1_beta = np.sum(alpha_ * B / (lambda_ + alpha_ * A))
            sum_term = np.sum(1 - np.exp(-beta_ * (T - times)))
            sum_term2 = np.sum((T - times) * np.exp(-beta_ * (T - times)))
            grad_term2_beta = -alpha_ / (beta_ ** 2) * sum_term + (alpha_ / beta_) * sum_term2
            dL_dbeta = grad_term1_beta - grad_term2_beta
            return np.array([dL_dlambda, dL_dbeta, dL_dalpha])


        def objective(params):
            lambda_, beta_, alpha_ = params
            return -log_likelihood(lambda_, beta_, alpha_)
        
        def grad_objective(params):
            lambda_, beta_, alpha_ = params
            return -gradient_log_likelihood(lambda_, beta_, alpha_)
        
        n_events = len(times)
        
        observed_intensity = n_events / T
        alpha_init = 0.5
        lambda_init = observed_intensity * (1 - alpha_init)
        beta_init = 1.0
        x0 = np.array([lambda_init, beta_init, alpha_init])
        bounds = Bounds([1e-9, 1e-9, 1e-9], [1e9, 1 - 1e-9, 1 - 1e-9])
        result = minimize(objective, x0, jac=grad_objective, method='L-BFGS-B', bounds=bounds, options={"disp": False, "ftol": 1e-15, "gtol": 1e-15})
        lambda_opt, beta_opt, alpha_opt = result.x
        
        return alpha_opt/T_org, beta_opt/T_org, lambda_opt/T_org

    def power_law_estimation(self, reg_term = 0, c_u = 1):
        c_ = c_u
        '''
        Estimation via ML for power law kernels with c<10 and 1<p<5
        '''
        eps   = 1e-9
        times = self.simulate_by_thinning_power_law()
        T = times[-1]
        print('Generated', len(times), 'points for estimation')
        
        times = np.array(times)
        
        def kern(t, c, p):
            return 1/(c+t)**p
        
        def func_aux(t, p, c):
            return c**(1-p) - (T-t+c)**(1-p)

        def log_likelihood(lambda_, k_, p_):
            A = []
            for i in range (len(times)):
                times_inf = times[times < times[i]]
                s = k_ * np.sum(kern(times[i]-times_inf,c_,p_))
                A.append(s)
            return np.sum(np.log(lambda_+np.array(A))) - (lambda_*T + k_/(p_-1)*np.sum(func_aux(times,p_,c_))) + reg_term * (1-((p_ - 1) * (c_ ** (p_ - 1)) - k_))**2
        
        # def log_likelihood(lambda_, k_, c_, p_):
        #     N = len(times)
        #     diff = times.reshape(-1, 1) - times.reshape(1, -1)
        #     mask = np.tril(np.ones((N, N), dtype=bool), k=-1)
        #     K = np.where(mask, kern(diff, c_, p_), 0)
        #     S = k_ * np.sum(K, axis=1)
        #     log_term = np.sum(np.log(lambda_ + S))
        #     integral_term = lambda_*T + k_/(p_-1)*np.sum(func_aux(times, p_, c_))
        def log_likelihood(lambda_, k_, p_):
            diff_matrix = np.subtract.outer(times, times)
            mask = np.tril(np.ones(diff_matrix.shape), k=-1).astype(bool)
            diffs = diff_matrix[mask]
            
            kernel_vals = 1.0 / (c_ + diffs)**p_
            
            S = np.zeros_like(times)
            idx = 0
            for i in range(len(times)):
                n = i
                if n > 0:
                    S[i] = np.sum(kernel_vals[idx: idx+n])
                    idx += n
            
            term1 = np.sum(np.log(lambda_ + k_ * S))
            term2 = lambda_ * T + k_/(p_-1)*np.sum(c_**(1-p_) - (T - times + c_)**(1-p_))
            
            reg = reg_term * (1 - ((p_ - 1) * c_**(p_ - 1) - k_))**2
            return term1 - term2 + reg
        
        def grad_log_likelihood(lambda_, k_, p_):
            N = len(times)
            grad_lambda = 0.0
            grad_k = 0.0
            grad_p = 0.0
            for i in range(N):
                past = times[times < times[i]]
                if past.size == 0:
                    S_i = 0.0
                    dS_dp = 0.0
                else:
                    diff = times[i] - past
                    denom = c_ + diff
                    terms = 1/denom**p_
                    S_i = np.sum(terms)
                    dterms_dp = -np.log(denom) * terms
                    dS_dp = np.sum(dterms_dp)
                intensity_i = lambda_ + k_ * S_i
                grad_lambda += 1/intensity_i
                grad_k += S_i/intensity_i
                grad_p += k_ * dS_dp/intensity_i

            grad_lambda -= T
            G = c_**(1-p_) - (T - times + c_)**(1-p_)
            sum_G = np.sum(G)
            grad_k -= sum_G/(p_ - 1)
            term1 = k_ * sum_G/(p_ - 1)**2
            dG_dp = -np.log(c_)*c_**(1-p_) + np.log(T - times + c_)*(T - times + c_)**(1-p_)
            term2 = - k_/(p_ - 1) * np.sum(dG_dp)
            grad_p += term1 + term2
            Q = (p_ - 1) * (c_ ** (p_ - 1)) - k_
            dQ_dk = -1.0
            dQ_dp = c_**(p_ - 1) * (1 + (p_ - 1) * np.log(c_))
            grad_lambda_reg = 0.0
            grad_k_reg = -2 * reg_term * (1 - Q) * dQ_dk
            grad_p_reg = -2 * reg_term * (1 - Q) * dQ_dp
            grad_lambda += grad_lambda_reg
            grad_k += grad_k_reg
            grad_p += grad_p_reg
            return np.array([grad_lambda, grad_k, grad_p])

        
        def objective(params):
            lambda_, k_, p_ = params
            return -log_likelihood(lambda_, k_, p_)
        
        def grad_objective(params):
            lambda_, k_, p_ = params
            return -grad_log_likelihood(lambda_, k_, p_)
        
        def estimate_k_init(times, m, p_init, lambda_init, c_):
            T_max = times[-1]
            segment_duration = T_max / m
            counts = []
            for i in range(m):
                t_start = i * segment_duration
                t_end = (i + 1) * segment_duration
                count = np.sum((times >= t_start) & (times < t_end))
                counts.append(count)
            counts = np.array(counts)
            N_est = np.mean(counts)
            observed_intensity = N_est / segment_duration
            
            k_init = (1 - lambda_init / observed_intensity) * (p_init - 1) / (c_ ** p_init)
            
            
            return k_init
        process_init = Hawkes_process(T)
        n_events = len(times)
        observed_intensity = n_events / T
        _ , p_init , lambda_init = process_init.fit_exponential(times)
        
        
        # window_length = 40
        # N_simulations = 50000
        window_length = 35
        N_simulations = 10000
        epochs = 70
        lr = 1e-3
        print("Training the NN...")
        self.fit_power_law_DL(length=window_length, N_simulations=N_simulations, epochs=epochs, lr=lr, c_ = c_u, lambda_ = lambda_init)
        # n_events = len(times)
        point_init = self.predict_parameters(times)
        # observed_intensity = n_events / T
        k_init = point_init[0]
        p_init = min(max(point_init[1],1 + 0.05),2)
        if p_init == 2 or p_init == 1:
            print('Fichtre')
            p_init = 1.5
        
        if k_init<1e-8:
            k_init = 0.1
        

        # #p_init = 1.5 #1 + k_init/(1-observed_intensity/lambda_init)
        # k_init = np.abs((1-observed_intensity/lambda_init)*(p_init-1))
        k_init = estimate_k_init(times, 50, p_init, lambda_init, c_)
        print(k_init)
        print(lambda_init)
        
        if p_init == 2 or p_init == 1:
            print('Fichtre')
            p_init = 1.5
        print(p_init)
        # lambda_init = observed_intensity * (1 - k_init/(p_init-1)*c_**(p_init-1))
        x0 = np.array([lambda_init, k_init, p_init])
        bounds = Bounds([1e-9, 1e-9, 1 + 0.05], [1e9, 1, 2])
        
        # result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={"disp": False, "ftol": 1e-15,  "gtol": 1e-15})
        # lambda_opt, k_opt, p_opt = result.x

        lambda_opt, k_opt, p_opt = lambda_init, k_init, p_init
        f_opt = 'ouiiiii' # -result.fun
        print("Optimal lambda   =", lambda_opt)
        print("Optimal k        =", k_opt)
        print("c choisi         =", c_)
        print("Optimal power    =", p_opt)
        print("Maximum log-like =", f_opt)
        
        def kernel_estimate(x):
            return k_opt/(c_+x)**p_opt
        
        x = np.linspace(0, 1, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x = x, 
            y = kernel_estimate(x),
            mode = 'markers',
            name = "Estimated Kernel",
            marker = dict(size = 0.85, color = 'darkred')
        ))
        fig.add_trace(go.Scatter(
            x = x, 
            y = self.kernel(x),
            mode = 'lines',
            name = "Real Kernel",
            line = dict(width = 0.85, color = 'black')
        ))
        fig.update_layout(
            title = "Estimated vs real kernel function",
            xaxis_title = "Time",
            yaxis_title = "Kernel Function",
            plot_bgcolor = '#D3D3D3',
            paper_bgcolor = '#D3D3D3',
            xaxis = dict(showgrid=True, gridcolor='#808080'),
            yaxis = dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
        
        return lambda_opt, k_opt, c_u, p_opt


    def simulate_hawkes_power_law(self, length, lambda_true, k_true, c_true, p_true):
        """
        Simule un processus Hawkes à noyau power law par thinning jusqu'à obtenir au moins
        'length' événements (ou jusqu'à T_max). Les paramètres sont pris comme arguments.
        """
        T_max = self.T_max
        events = []
        t = 0.0
        # On fixe une borne supérieure approximative pour l'intensité.
        lambda_bar = lambda_true + k_true / (c_true**p_true)
        while t < T_max and len(events) < length:
            # Génération de l'intervalle d'attente avec une intensité majorante
            u = np.random.rand()
            t = t - np.log(u) / lambda_bar
            D = np.random.rand()
            # Calcul de l'intensité réelle au temps t
            if len(events) == 0:
                intensity = lambda_true
            else:
                # On considère uniquement les événements antérieurs à t
                past_events = np.array(events)[np.array(events) < t]
                if past_events.size > 0:
                    diffs = t - past_events
                    intensity = lambda_true + k_true * np.sum(1.0 / (c_true + diffs)**p_true)
                else:
                    intensity = lambda_true
            if D <= intensity / lambda_bar:
                events.append(t)
        events = np.array(events)
        # Si on n'a pas obtenu assez d'événements, on pad avec T_max
        if events.size < length:
            events = np.pad(events, (0, length - events.size), 'constant', constant_values=T_max)
        else:
            events = events[:length]
        return events

    class ParameterEstimator(nn.Module):
        def __init__(self, input_length, output_size=2, hidden_channels=64, dropout_prob=0.2):
            """
            input_length: longueur de la séquence (nombre d'événements dans la fenêtre)
            output_size: nombre de paramètres à prédire (ici k et p)
            hidden_channels: nombre de canaux utilisés dans les convolutions
            """
            super().__init__()
            # La séquence d'entrée est de taille (batch, 1, input_length)
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm1d(hidden_channels)
            self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm1d(hidden_channels)
            self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
            self.bn3   = nn.BatchNorm1d(hidden_channels)
            # Pour réduire la dimension spatiale, on utilise un global average pooling
            self.fc    = nn.Linear(hidden_channels, output_size)
            self.dropout = nn.Dropout(dropout_prob)
            
        def forward(self, x):
            # x est de taille (batch, input_length)
            x = x.unsqueeze(1)  # passe à (batch, 1, input_length)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Global average pooling sur la dimension temporelle
            x = torch.mean(x, dim=2)  # résultat de taille (batch, hidden_channels)
            out = self.fc(x)  # résultat de taille (batch, output_size)
            return out

    def fit_power_law_DL(self, length, N_simulations=1000, epochs=100, lr=1e-3, c_ = 1, lambda_ = 100):
        """
        Simule N_simulations processus Hawkes power law avec paramètres aléatoires en parallèle.
        Chaque simulation génère une séquence d'événements de longueur 'length' (normalisés par T_max).
        Le réseau de neurones est entraîné pour estimer [lambda, k, c, p].
        """
        def simulate_one(_):
            # Choix aléatoire des paramètres dans des intervalles choisis arbitrairement
            lambda_true = lambda_
            k_true      = np.random.uniform(0.001, 0.1)
            c_true      = c_
            p_true      = np.random.uniform(1.1, 2.0)
            # Simulation du processus (on utilise uniquement les 'length' premiers événements)
            times = self.simulate_hawkes_power_law(length, lambda_true, k_true, c_true, p_true)
            times = np.diff(np.concatenate(([0], times)))
            # Normalisation par T_max pour faciliter l'apprentissage
            times_norm = times
            return times_norm, [k_true, p_true]
        print('Data Simulated')
        # Exécution parallèle des simulations
        results = Parallel(n_jobs=-1)(delayed(simulate_one)(i) for i in range(N_simulations))
        X, Y = zip(*results)
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)

        # Passage aux tenseurs torch
        X_tensor = torch.tensor(X)
        Y_tensor = torch.tensor(Y)

        input_size = length
        model = self.ParameterEstimator(input_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Boucle d'entraînement
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, Y_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

        # Sauvegarde du modèle entraîné et de la taille de la fenêtre
        self.model = model
        self.window_length = length

    def predict_parameters(self, event_times):
        """
        Pour une nouvelle série d'événements (tableau event_times), 
        utilise une approche glissante (de taille window_length) pour obtenir plusieurs estimations 
        via le réseau de neurones, puis retourne la moyenne de ces prédictions.
        """
        if not hasattr(self, 'model'):
            raise ValueError("Le modèle n'est pas entraîné. Appelez d'abord fit_power_law_DL.")
        model = self.model
        length = self.window_length
        event_times = np.array(event_times)
        event_times = np.diff(np.concatenate(([0], event_times)))
        preds = []
        # Pour chaque fenêtre glissante de 'length' événements
        for i in range(len(event_times) - length + 1):
            window = event_times[i:i+length]
            # Normalisation par T_max
            window_norm = window / self.T_max
            window_tensor = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                pred = model(window_tensor).numpy().flatten()
            preds.append(pred)
        preds = np.array(preds)
        # Moyenne des prédictions pour chaque paramètre
        avg_params = preds.mean(axis=0)
        print('lambda_true, k_true, p_true, ',avg_params)
        return avg_params


