import pandas as pd
import plotly.graph_objects as go
import numpy as np
import Random_event_simulation as rand_event
import Hawkes as hk
import warnings
warnings.filterwarnings("ignore")


from Hawkes.model import baseline_pconst
import numpy as np


def patched_l(self, t):
    para = self.para
    # Vérifier que l'intervalle est défini
    if not hasattr(self, 'itv'):
        raise AttributeError("baseline_pconst n'est pas correctement configuré. Veuillez appeler set_itv() avant de simuler.")
    # Si aucune donnée n'a été fournie, on crée une donnée par défaut
    if not hasattr(self, 'Data'):
        # On initialise avec un tableau contenant le début de l'intervalle
        self.Data = {'T': np.array([self.itv[0]])}
    # Si l'attribut pconst n'existe pas, appeler prep_fit pour l'initialiser
    if not hasattr(self, 'pconst'):
        self.prep_fit()
    coef = para['mu']
    return self.pconst.set_coef(coef).get_y_at(t)

from Hawkes.model import baseline_plinear, baseline_loglinear
import numpy as np

def patched_l_plinear(self, t):
    para = self.para
    # Vérifier que l'intervalle est défini
    if not hasattr(self, 'itv'):
        raise AttributeError("baseline_plinear n'est pas correctement configuré. Veuillez appeler set_itv() avant de simuler.")
    # Si aucune donnée n'a été fournie, on crée une donnée par défaut
    if not hasattr(self, 'Data'):
        self.Data = {'T': np.array([self.itv[0]])}
    # Si l'attribut plinear n'existe pas, on appelle prep_fit pour l'initialiser
    if not hasattr(self, 'plinear'):
        self.prep_fit()
    coef = para['mu']
    return self.plinear.set_coef(coef).get_y_at(t)

def patched_l_loglinear(self, t):
    para = self.para
    # Vérifier que l'intervalle est défini
    if not hasattr(self, 'itv'):
        raise AttributeError("baseline_loglinear n'est pas correctement configuré. Veuillez appeler set_itv() avant de simuler.")
    # Si aucune donnée n'a été fournie, on crée une donnée par défaut
    if not hasattr(self, 'Data'):
        self.Data = {'T': np.array([self.itv[0]])}
    # Si l'attribut loglinear n'existe pas, on appelle prep_fit pour l'initialiser
    if not hasattr(self, 'loglinear'):
        self.prep_fit()
    coef = para['mu']
    return self.loglinear.set_coef(coef).get_y_at(t)

# Appliquer le monkey patch sur les classes concernées
baseline_plinear.l = patched_l_plinear
baseline_loglinear.l = patched_l_loglinear




# Appliquer le monkey patch sur la classe baseline_pconst
baseline_pconst.l = patched_l

from Hawkes.model import baseline_loglinear
import numpy as np

def patched_l_loglinear(self, t):
    para = self.para
    # Vérifier que l'intervalle est défini
    if not hasattr(self, 'itv'):
        raise AttributeError("baseline_loglinear n'est pas correctement configuré. Veuillez appeler set_itv() avant de simuler.")
    # Si aucune donnée n'a été fournie, on crée une donnée par défaut
    if not hasattr(self, 'Data'):
        self.Data = {'T': np.array([self.itv[0]])}
    # Si l'attribut loglinear n'existe pas, on appelle prep_fit pour l'initialiser
    if not hasattr(self, 'loglinear'):
        self.prep_fit()
    coef = para['mu']
    # Si t est un scalaire, le transformer en tableau
    scalar_input = np.isscalar(t)
    if scalar_input:
        x_val = np.array([t])
    else:
        x_val = t
    # Calculer l'intensité
    result = self.loglinear.set_coef(coef).get_y_at(x_val)
    # Si l'entrée était scalaire, renvoyer la première valeur du résultat
    if scalar_input:
        return result[0]
    else:
        return result

# Appliquer le monkey patch sur la classe baseline_loglinear
baseline_loglinear.l = patched_l_loglinear



from Hawkes.model import baseline_pconst
import numpy as np

def patched_l(self, t):
    para = self.para
    # Vérifier que l'intervalle est défini
    if not hasattr(self, 'itv'):
        raise AttributeError("baseline_pconst n'est pas correctement configuré. Veuillez appeler set_itv() avant de simuler.")
    # Si aucune donnée n'a été fournie, on crée une donnée par défaut
    if not hasattr(self, 'Data'):
        # On initialise avec un tableau contenant le début de l'intervalle
        self.Data = {'T': np.array([self.itv[0]])}
    # Si l'attribut pconst n'existe pas, appeler prep_fit pour l'initialiser
    if not hasattr(self, 'pconst'):
        self.prep_fit()
    coef = para['mu']
    return self.pconst.set_coef(coef).get_y_at(t)

from Hawkes.model import baseline_plinear, baseline_loglinear
import numpy as np

def patched_l_plinear(self, t):
    para = self.para
    # Vérifier que l'intervalle est défini
    if not hasattr(self, 'itv'):
        raise AttributeError("baseline_plinear n'est pas correctement configuré. Veuillez appeler set_itv() avant de simuler.")
    # Si aucune donnée n'a été fournie, on crée une donnée par défaut
    if not hasattr(self, 'Data'):
        self.Data = {'T': np.array([self.itv[0]])}
    # Si l'attribut plinear n'existe pas, on appelle prep_fit pour l'initialiser
    if not hasattr(self, 'plinear'):
        self.prep_fit()
    coef = para['mu']
    return self.plinear.set_coef(coef).get_y_at(t)

def patched_l_loglinear(self, t):
    para = self.para
    # Vérifier que l'intervalle est défini
    if not hasattr(self, 'itv'):
        raise AttributeError("baseline_loglinear n'est pas correctement configuré. Veuillez appeler set_itv() avant de simuler.")
    # Si aucune donnée n'a été fournie, on crée une donnée par défaut
    if not hasattr(self, 'Data'):
        self.Data = {'T': np.array([self.itv[0]])}
    # Si l'attribut loglinear n'existe pas, on appelle prep_fit pour l'initialiser
    if not hasattr(self, 'loglinear'):
        self.prep_fit()
    coef = para['mu']
    return self.loglinear.set_coef(coef).get_y_at(t)

# Appliquer le monkey patch sur les classes concernées
baseline_plinear.l = patched_l_plinear
baseline_loglinear.l = patched_l_loglinear




# Appliquer le monkey patch sur la classe baseline_pconst
baseline_pconst.l = patched_l



class data_analysis:
    def __init__(self, df):
        self.df_ = df
        tick = np.min(np.abs(self.df_['bid_px_00']-self.df_['ask_px_00']))
        self.tick =  np.round(tick, 4)
        self.df_ = self.df_[np.abs(self.df_['price'].diff(1))<30*tick]
        self.data_hawkes()
        
        
    def stats_(self):
        tick = np.min(np.abs(self.df_['bid_px_00']-self.df_['ask_px_00']))
        print(' -- Dataset statistics -- \n')
        self.tick =  np.round(tick, 4)
        print('Tick                                  :', np.round(tick, 4))
        bid_ask = np.mean(np.abs(self.df_['bid_px_00']-self.df_['ask_px_00']))
        print('Average Bid-ask                       :', bid_ask)
        print('Number of Jumps of more than just one :', self.p_u)
        print('Number of Jumps of less than just one :', self.p_d)
        print('Average jump size (up)                :', self.average_jump_u/np.round(tick, 4))
        print('Average jump size (down)              :', self.average_jump_d/np.round(tick, 4))
        print('Number of up jumps                    :',len(self.increases))
        print('Number of down jumps                  :',len(self.decreases))
        print('\n -- Graphs -- ')
        jump_sizes = np.array(sorted(self.jump_counts_down.keys()))
        counts = np.array([self.jump_counts_down[size] for size in jump_sizes])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=jump_sizes,
            y=counts,
            name="Down",
            marker_color='darkred'
        ))
        jump_sizes = np.array(sorted(self.jump_counts_up.keys()))
        counts = np.array([self.jump_counts_up[size] for size in jump_sizes])
        fig.add_trace(go.Bar(
            x=jump_sizes,
            y=counts,
            name="Up",
            marker_color='darkblue'
        ))
        fig.update_layout(
            title="Size Jump Histogram",
            xaxis_title="Jump Size",
            yaxis_title="Number of Jumps",
            plot_bgcolor='#D3D3D3',
            paper_bgcolor='#D3D3D3',
            xaxis=dict(showgrid=True, gridcolor='#808080'),
            yaxis=dict(showgrid=True, gridcolor='#808080')
        )
        fig.show()
        self.visu_price()
        # self.visu_arrival()
        # self.visu_time()
    

    def visu_price(self):
        df = self.df_
        df    = df[df['action'] == 'T']
        time  = pd.to_datetime(df['ts_event'])
        price = df['price']
        fig   = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=price, mode='lines', name="Stock-Price", line=dict(width = 0.85, color = 'darkred')))
        fig.update_layout(
                title="Price_GOOGL",
                xaxis_title="Time",
                yaxis_title="Price",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()

    def visu_time(self):
        df = self.df_
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        t_0            = df['ts_event'].iloc[0]
        df             = df.copy()
        df['ts_event'] = (df['ts_event'] - t_0).dt.total_seconds()
        df             = df[df['action'] == 'T']
        time           = df['ts_event'].to_numpy()
        df['diff']     = df['ts_event'].diff()
        dt             = df['diff']
        print(len(time[1:]), 'points considered')
        fig            = go.Figure()
        fig.add_trace(go.Scatter(x=time[1:], y=dt[1:], mode='lines', name="Intensity", line=dict(width = 0.85, color = 'darkblue')))
        fig.update_layout(
                title="Intensity",
                xaxis_title="Time",
                yaxis_title="intensity",
                plot_bgcolor='#D3D3D3',
                paper_bgcolor='#D3D3D3',
                xaxis=dict(showgrid=True, gridcolor='#808080'),
                yaxis=dict(showgrid=True, gridcolor='#808080')
            )
        fig.show()
        fig            = go.Figure()
        fig.add_trace(go.Scatter(x=time[1:], y=np.arange(len(dt[1:])), mode='lines', name="Number of events", line=dict(width = 0.85, color = 'darkred')))
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

    def data_hawkes(self):
        tick = np.min(np.abs(self.df_['bid_px_00']-self.df_['ask_px_00']))
        df = self.df_
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        df = df.sort_values('ts_event').reset_index(drop=True)
        #df = df[df['action'] =='T']
        df["mid"] = (df["bid_px_00"]+df["ask_px_00"])/2
        df = df[df['mid'].diff(1)!=0]
        t0 = df['ts_event'].iloc[0]
        df['time_sec'] = (df['ts_event'] - t0).dt.total_seconds()

        increases = []
        decreases = []

        self.p_u = 0
        self.p_d = 0
        
        self.average_jump_u = 0
        self.average_jump_d = 0
        
        precision = 6

        self.jump_counts_up   = {}
        self.jump_counts_down = {}
        self.jump_size_up = {}
        self.jump_size_down = {}
        
        tick = np.min(np.abs(self.df_['bid_px_00']-self.df_['ask_px_00']))
        
        prev_price = df['price'].iloc[0]
        for idx, row in df.iloc[1:].iterrows():
            current_price = row['price']
            timestamp = row['time_sec']
            jump = current_price - prev_price
            jump_key = round(jump, precision)
            if current_price > prev_price:
                self.average_jump_u += current_price-prev_price
                if current_price-prev_price>0.015:
                    self.p_u +=1
                increases.append(timestamp)
                if jump_key in self.jump_size_up:
                    self.jump_size_up[jump_key].append(row['size'])
                else:
                    self.jump_size_up[jump_key] = [row['size']]
                if jump_key in self.jump_counts_up:
                    self.jump_counts_up[jump_key] += 1
                else:
                    self.jump_counts_up[jump_key] = 1
            elif current_price < prev_price:
                self.average_jump_d += current_price-prev_price
                if current_price-prev_price<0.015:
                    self.p_d +=1
                decreases.append(timestamp)
                if jump_key in self.jump_size_down:
                    self.jump_size_down[jump_key].append(row['size'])
                else:
                    self.jump_size_down[jump_key] = [row['size']]
                if jump_key in self.jump_counts_down:
                    self.jump_counts_down[jump_key] += 1
                else:
                    self.jump_counts_down[jump_key] = 1
            prev_price = current_price
        self.increases = increases
        self.decreases = decreases
        self.average_jump_u /= len(increases)
        self.average_jump_d /= len(decreases)
        transform_key = lambda k: int(k / tick)

        new_jump_counts_down = {}
        for k, v in self.jump_counts_down.items():
            new_key = transform_key(k)
            new_jump_counts_down[new_key] = v
        
        new_jump_counts_up = {}
        for k, v in self.jump_counts_up.items():
            new_key = transform_key(k)
            new_jump_counts_up[new_key] = v
            
        new_jump_size_down = {}
        for k, v in self.jump_size_down.items():
            new_key = transform_key(k)
            new_jump_size_down[new_key] = v
        
        new_jump_size_up = {}
        for k, v in self.jump_size_up.items():
            new_key = transform_key(k)
            new_jump_size_up[new_key] = v

        self.jump_counts_down = new_jump_counts_down
        self.jump_counts_up = new_jump_counts_up
        self.jump_size_down = new_jump_size_down
        self.jump_size_up = new_jump_size_up
        self.df = df
        self.df.to_csv("test__")
    
    def visu_arrival(self):
        fig = go.Figure()
            # fig.add_trace(go.Scatter(x=times, y=np.arange(len(dt[1:])), mode='lines', name="Real number of events", line=dict(width = 0.85, color = 'black')))
        fig.add_trace(go.Scatter(x=self.increases, y=np.arange(len(self.increases)), mode='lines', name="N+", line=dict(width = 0.85, color = 'darkgreen')))
        fig.add_trace(go.Scatter(x=self.decreases, y=np.arange(len(self.decreases)), mode='lines', name="N-", line=dict(width = 0.85, color = 'darkblue')))
        fig.update_layout(
                    title="Number of events N+ and N-",
                    xaxis_title="Time",
                    yaxis_title="Number of events",
                    plot_bgcolor='#D3D3D3',
                    paper_bgcolor='#D3D3D3',
                    xaxis=dict(showgrid=True, gridcolor='#808080'),
                    yaxis=dict(showgrid=True, gridcolor='#808080')
                )
        fig.show()
        
    def fit_exp_(self, visu = False):
        process = rand_event.Hawkes_process(self.increases[-1])
        alpha_opt, beta_opt, lambda_opt = process.fit_exponential(self.increases)

        if visu:
            print("Estimation Done !!!")
            print("alpha_opt, beta_opt, lambda_opt = ", alpha_opt, beta_opt, lambda_opt)
            process_2 = rand_event.Hawkes_process(self.increases[-1], alpha_exp = alpha_opt, beta_exp = beta_opt, lambda_0 = lambda_opt)
            events_sim = np.array(process_2.simulate_by_thinning_exponential())
            print(f"Simulated {len(events_sim)} events")
            print(f"{len(self.increases)} real events")
            fig = go.Figure()
            # fig.add_trace(go.Scatter(x=times, y=np.arange(len(dt[1:])), mode='lines', name="Real number of events", line=dict(width = 0.85, color = 'black')))
            fig.add_trace(go.Scatter(x=events_sim, y=np.arange(len(events_sim)), mode='lines', name="Simulated number of events", line=dict(width = 0.85, color = 'darkgreen')))
            fig.add_trace(go.Scatter(x=self.increases, y=np.arange(len(self.increases)), mode='lines', name="Real number of events", line=dict(width = 0.85, color = 'darkred')))
            fig.update_layout(
                    title="Number of events (Real vs Estimated)",
                    xaxis_title="Time",
                    yaxis_title="Number of events",
                    plot_bgcolor='#D3D3D3',
                    paper_bgcolor='#D3D3D3',
                    xaxis=dict(showgrid=True, gridcolor='#808080'),
                    yaxis=dict(showgrid=True, gridcolor='#808080')
                )
            fig.show()
            x = np.linspace(0,1,100)
            def kernel(x):
                return alpha_opt*np.exp(-beta_opt*x)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=kernel(x), mode='lines', name="Estimated kernel", line=dict(width = 0.85, color = 'darkred')))
            fig.update_layout(
                    title="Estimated Kernel",
                    xaxis_title="Time",
                    yaxis_title="Kernel",
                    plot_bgcolor='#D3D3D3',
                    paper_bgcolor='#D3D3D3',
                    xaxis=dict(showgrid=True, gridcolor='#808080'),
                    yaxis=dict(showgrid=True, gridcolor='#808080')
                )
            fig.show()

        return alpha_opt, beta_opt, lambda_opt
    
    def fit(self, method, visu = False):
        if method == 'Exp_univariate':
            process = rand_event.Hawkes_process(self.increases[-1])
            alpha_opt_plus, beta_opt_plus, lambda_opt_plus = process.fit_exponential(self.increases)
            process = rand_event.Hawkes_process(self.decreases[-1])
            alpha_opt_minus, beta_opt_minus, lambda_opt_minus = process.fit_exponential(self.decreases)
            if visu:
                print('Fitting done.')
                df   = self.df_
                df   = df[df['action'] == 'T']
                df   = self.df_.sort_values('ts_event').reset_index(drop=True)
                t0   = df['ts_event'].iloc[0]
                df['time_sec'] = (df['ts_event'] - t0).dt.total_seconds()
                _, _ = self.simulate_exp_univariate(df['time_sec'].to_numpy()[-1], alpha_opt_plus, beta_opt_plus, lambda_opt_plus, alpha_opt_minus, beta_opt_minus, lambda_opt_minus, self.df_['price'].iloc[0], tick = np.round(np.min(np.abs(self.df_['bid_px_00']-self.df_['ask_px_00'])),4), visu = True, show_real_p=True)
            return alpha_opt_plus, beta_opt_plus, lambda_opt_plus, alpha_opt_minus, beta_opt_minus, lambda_opt_minus
        if method == 'Exp_multivariate':
            return 'A coder'
        if method == 'Power_law_univariate':
            return 'A coder'
        if method == 'Power_law_multivariate':
            return 'A coder'
        else:
            return "Method not available, choose between 'Exp_univariate', 'Exp_multivariate', 'Power_law_univariate', 'Power_law_multivariate'."
        
    def simulate_exp_univariate(self, T, alpha_opt_plus, beta_opt_plus, lambda_opt_plus, alpha_opt_minus, beta_opt_minus, lambda_opt_minus, p_0, tick = 0.01, visu = False, show_real_p = False):
        # process_plus  = rand_event.Hawkes_process(T, kernel_type = 'Exp', alpha_exp = alpha_opt_plus, beta_exp = beta_opt_plus, lambda_0 = lambda_opt_plus)
        # process_minus = rand_event.Hawkes_process(T, kernel_type = 'Exp', alpha_exp = alpha_opt_minus, beta_exp = beta_opt_minus, lambda_0 = lambda_opt_minus)
        # event_plus    = process_plus.simulate_by_thinning_exponential()
        # event_minus   = process_minus.simulate_by_thinning_exponential()
        
        
        itv = [0, self.increases[-1]]
        h2 = hk.estimator().set_kernel('pow').set_baseline('loglinear',num_basis=6)
        h2.fit(self.increases,itv)
        para = h2.para
        h1 = hk.simulator().set_kernel('pow').set_baseline('loglinear',num_basis=6).set_parameter(para)
        event_plus = h1.simulate(itv)
        print(h2.para)
        
        itv = [0, self.decreases[-1]]
        h2 = hk.estimator().set_kernel('pow').set_baseline('loglinear',num_basis=6)
        h2.fit(self.decreases,itv)
        para = h2.para
        h1 = hk.simulator().set_kernel('pow').set_baseline('loglinear',num_basis=6).set_parameter(para)
        event_minus = h1.simulate(itv)
        print(h2.para)
        
        keys_down = np.array(list(self.jump_counts_down.keys()))
        values_down = np.array(list(self.jump_counts_down.values()))
        probabilities_down = values_down / values_down.sum()
        keys_up = np.array(list(self.jump_counts_up.keys()))
        values_up = np.array(list(self.jump_counts_up.values()))
        probabilities_up = values_up / values_up.sum()
        
        up   = np.random.choice(keys_up, p=probabilities_up, size = len(event_plus))
        down = np.random.choice(keys_down, p=probabilities_down, size = len(event_minus))
        up_down     = np.concatenate([up, down])
        events_time = np.concatenate([event_plus, event_minus])
        
        ind = np.argsort(events_time)
        events_time = events_time[ind]

        price = p_0 + tick*np.cumsum(up_down[ind])
        if visu:
            fig = go.Figure()
            if show_real_p:
                df   = self.df_
                df   = df[df['action'] == 'T'].copy()
                t0   = df.loc[:, 'ts_event'].iloc[0]
                df.loc[:, 'time_sec'] = (df.loc[:, 'ts_event'] - t0).dt.total_seconds()
                pri  = df['price']
                fig  = go.Figure()
                fig.add_trace(go.Scatter(x=df['time_sec'], y=pri, mode='lines', name="Real Price", line=dict(width = 0.85, color = 'darkred')))
            fig.add_trace(go.Scatter(x=events_time, y=price, mode='lines', name="Simulated Price", line=dict(width = 0.85, color = 'darkblue')))
            fig.update_layout(
                    title="Simulated data",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    plot_bgcolor='#D3D3D3',
                    paper_bgcolor='#D3D3D3',
                    xaxis=dict(showgrid=True, gridcolor='#808080'),
                    yaxis=dict(showgrid=True, gridcolor='#808080')
                )
            fig.show()
            fig  = go.Figure()
            fig.add_trace(go.Scatter(x=np.sort(np.concatenate([self.increases,self.decreases])), y=np.arange(len(np.concatenate([self.increases,self.decreases]))), mode='lines', name="Real data", line=dict(width = 0.85, color = 'darkred')))
            fig.add_trace(go.Scatter(x=events_time, y=np.arange(len(events_time)), mode='lines', name="Simulated data", line=dict(width = 0.85, color = 'darkblue')))
            fig.update_layout(
                    title="Simulated data",
                    xaxis_title="Time",
                    yaxis_title="N(t)",
                    plot_bgcolor='#D3D3D3',
                    paper_bgcolor='#D3D3D3',
                    xaxis=dict(showgrid=True, gridcolor='#808080'),
                    yaxis=dict(showgrid=True, gridcolor='#808080')
                )
            fig.show()
        return events_time, price
    
    
    
    def fit_2(self):
        itv = [0, self.increases[-1]]
        h2 = hk.estimator().set_kernel('exp').set_baseline('const')
        h2.fit(self.increases,itv)
        print(h2.para)