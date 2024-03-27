import torch as th
from functools import cached_property
from utils.bernstein import bernstein_coeff_order10_new


class InitializerModule(th.nn.Module):
    def __init__(self, mean, std, in_dim, out_dim, mlp_shape):
        super(InitializerModule, self).__init__()
        self.mean = mean
        self.std = std
        shapes = [in_dim] + mlp_shape + [out_dim]
        self.net = th.nn.Sequential(
        *[
            th.nn.Sequential(
                th.nn.Linear(in_fea, out_fea),
                th.nn.BatchNorm1d(out_fea),
                th.nn.ReLU()
            )
            for i, (in_fea, out_fea) in enumerate(th.tensor(shapes).unfold(0, 2, 1))
        ] + [
            th.nn.Linear(shapes[-1], shapes[-1])
        ])
            
    def forward(self, primal_sol_1, observations):
        observations = (observations - self.mean) / self.std
        x = th.cat([primal_sol_1, observations], dim=-1)           
        out = self.net(x)
        return out


class DOptimizer(th.nn.Module):
    def __init__(self, initializer_model, num_batch, device):
        super(DOptimizer, self).__init__()
        self.device = device
        self.qp1_train = False 
        self.initializer_model = initializer_model

        self.r1 = 0.0
        self.r2 = 2.5
        self.r3 = -2.5
        self.dist_centre = th.tensor([self.r1, self.r2, self.r3], device=self.device)

        self.num_circles = 3
        self.margin = 0.6

        self.v_max = 30.0
        self.v_min = 0.1
        self.v_des = 20.0 
        self.a_max = 18.0
        self.a_centr = 1.5
        self.num_obs = 10
        self.num_batch = num_batch
        self.steer_max = 1.2
        self.kappa_max = 0.895
        self.wheel_base = 2.875
        self.a_obs = 4.5
        self.b_obs = 3.5

        self.t_fin = 15
        self.num = 100
        self.t = self.t_fin/self.num
        # self.ellite_num = 150
        self.ellite_num = num_batch if num_batch <= 150 else 150
        self.ellite_num_projection = 150
        self.ellite_num_projection = self.ellite_num_projection if num_batch >= self.ellite_num_projection else num_batch

        self.tot_time = th.linspace(0, self.t_fin, self.num, device=self.device)
        tot_time_copy = self.tot_time.view(self.num, 1)

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_new(
            10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy, device=self.device
        )
        self.nvar = self.P.size(dim=1)

        self.A_eq_x = th.vstack([self.P[0], self.Pdot[0], self.Pddot[0]])
        self.A_eq_y = th.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1]])

        self.A_vel = self.Pdot
        self.A_acc = self.Pddot
        self.A_projection = th.eye(self.nvar, device=self.device)

        self.A_y_centerline = self.P
        self.A_obs = th.tile(self.P, (self.num_obs * self.num_circles, 1))
        self.A_lane = th.vstack([self.P, - self.P])

        self.rho_nonhol = 1.0
        self.rho_ineq = 1
        self.rho_obs = 1.0
        self.rho_projection = 1.0
        self.rho_goal = 1.0
        self.rho_lane = 1.0
        self.rho_long = 1.0

        self.rho_v = 1
        self.rho_offset = 1

        self.weight_smoothness_x = 1
        self.weight_smoothness_y = 1

        self.maxiter = 100
        self.maxiter_cem = 1

        self.k_p_v = 2
        self.k_d_v = 2.0 * th.sqrt(th.tensor(self.k_p_v, device=device))

        self.k_p = 2
        self.k_d = 2.0 * th.sqrt(th.tensor(self.k_p))

        self.P1 = self.P[0:25, :]
        self.P2 = self.P[25:50, :]
        self.P3 = self.P[50:75, :]
        self.P4 = self.P[75:100, :]

        self.Pdot1 = self.Pdot[0:25, :]
        self.Pdot2 = self.Pdot[25:50, :]
        self.Pdot3 = self.Pdot[50:75, :]
        self.Pdot4 = self.Pdot[75:100, :]

        self.Pddot1 = self.Pddot[0:25, :]
        self.Pddot2 = self.Pddot[25:50, :]
        self.Pddot3 = self.Pddot[50:75, :]
        self.Pddot4 = self.Pddot[75:100, :]

        self.num_partial = 25

        generator = th.Generator(device=self.device)
        self.key = generator.initial_seed()

        self.gamma = 0.9
        self.gamma_obs = 0.98
        self.gamma_obs_long = 0.9

        # upper lane bound
        self.P_ub_1 = self.P[1:self.num, :]
        self.P_ub_0 = self.P[0:self.num-1, :]
        self.A_ub = self.P_ub_1 + (self.gamma-1) * self.P_ub_0

        # lower lane bound
        self.P_lb_1 = - self.P[1:self.num, :]
        self.P_lb_0 = self.P[0:self.num-1, :]
        self.A_lb = self.P_lb_1 + (1 - self.gamma) * self.P_lb_0

        # vstack A_ub and A_lb
        self.A_lane_bound = th.vstack((self.A_ub, self.A_lb))
        self.d_separate = 5.0

        self.cost_smoothness = 1 * (self.Pddot.T @ self.Pddot)

        self.num_mean_update = 8
        self.t_target = (self.num_mean_update-1) * self.t
        
    def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init):
        x_init_vec = x_init[:, None]
        y_init_vec = y_init[:, None]

        vx_init_vec = vx_init[:, None]
        vy_init_vec = vy_init[:, None]

        ax_init_vec = ax_init[:, None]
        ay_init_vec = ay_init[:, None]

        b_eq_x = th.hstack([x_init_vec, vx_init_vec, ax_init_vec])
        b_eq_y = th.hstack([y_init_vec, vy_init_vec, ay_init_vec, th.zeros((self.num_batch, 1), device=self.device)])

        return b_eq_x, b_eq_y
    
    @cached_property
    def compute_mat_inv_layer_1(self):

        A_vd_1 = self.Pddot1 - (self.k_p_v * self.Pdot1)
        A_vd_2 = self.Pddot2 - (self.k_p_v * self.Pdot2)
        A_vd_3 = self.Pddot3 - (self.k_p_v * self.Pdot3)
        A_vd_4 = self.Pddot4 - (self.k_p_v * self.Pdot4)

        A_pd_1 = self.Pddot1 - (self.k_p * self.P1) - (self.k_d * self.Pdot1)
        A_pd_2 = self.Pddot2 - (self.k_p * self.P2) - (self.k_d * self.Pdot2)
        A_pd_3 = self.Pddot3 - (self.k_p * self.P3) - (self.k_d * self.Pdot3)
        A_pd_4 = self.Pddot4 - (self.k_p * self.P4) - (self.k_d * self.Pdot4)

        cost_x = self.cost_smoothness \
                 + (self.rho_v * th.mm(A_vd_1.T, A_vd_1)) \
                 + (self.rho_v * th.mm(A_vd_2.T, A_vd_2)) \
                 + (self.rho_v * th.mm(A_vd_3.T, A_vd_3)) \
                 + (self.rho_v * th.mm(A_vd_4.T, A_vd_4))

        cost_y = self.cost_smoothness \
                 + (self.rho_offset * th.mm(A_pd_1.T, A_pd_1))\
                 + (self.rho_offset * th.mm(A_pd_2.T, A_pd_2))\
                 + (self.rho_offset * th.mm(A_pd_3.T, A_pd_3))\
                 + (self.rho_offset * th.mm(A_pd_4.T, A_pd_4))

        cost_mat_x = th.vstack([
            th.hstack([cost_x, self.A_eq_x.T]),
            th.hstack([self.A_eq_x, th.zeros((self.A_eq_x.size(dim=0), self.A_eq_x.size(dim=0)), device=self.device)])
        ])
        cost_mat_y = th.vstack([
            th.hstack([cost_y, self.A_eq_y.T]),
            th.hstack([self.A_eq_y, th.zeros((self.A_eq_y.size(dim=0), self.A_eq_y.size(dim=0)), device=self.device)])
        ])

        cost_mat_inv_x = th.linalg.inv(cost_mat_x)
        cost_mat_inv_y = th.linalg.inv(cost_mat_y)

        return cost_mat_inv_x, cost_mat_inv_y
    
    @cached_property
    def compute_mat_inv_layer_2(self):

        cost_x = self.rho_projection * th.mm(self.A_projection.T, self.A_projection) \
                 + self.rho_ineq * th.mm(self.A_acc.T, self.A_acc) \
                 + self.rho_ineq * th.mm(self.A_vel.T, self.A_vel) \
                 + self.rho_obs * th.mm(self.A_obs.T, self.A_obs) \
                 + 0.2 * th.eye(self.nvar, device=self.device)

        cost_y = self.rho_projection * th.mm(self.A_projection.T, self.A_projection) \
                 + self.rho_ineq * th.mm(self.A_acc.T, self.A_acc) \
                 + self.rho_ineq * th.mm(self.A_vel.T, self.A_vel) \
                 + self.rho_obs * th.mm(self.A_obs.T, self.A_obs) \
                 + self.rho_lane * th.mm(self.A_lane_bound.T, self.A_lane_bound) \
                 + 0.2 * th.eye(self.nvar, device=self.device)

        cost_mat_x = th.vstack([
            th.hstack([cost_x, self.A_eq_x.T]),
            th.hstack([self.A_eq_x, th.zeros((self.A_eq_x.size(dim=0), self.A_eq_x.size(dim=0)), device=self.device)])
        ])

        cost_mat_y = th.vstack([
            th.hstack([cost_y, self.A_eq_y.T]),
            th.hstack([self.A_eq_y, th.zeros((self.A_eq_y.size(dim=0), self.A_eq_y.size(dim=0)), device=self.device)])
        ])

        cost_mat_inv_x = th.linalg.inv(cost_mat_x)
        cost_mat_inv_y = th.linalg.inv(cost_mat_y)

        return cost_mat_inv_x, cost_mat_inv_y
    
    def qp_layer_1(self, initial_state, neural_output_batch, cost_mat_inv_x, cost_mat_inv_y):
            
        x_init_vec, y_init_vec, vx_init_vec, vy_init_vec, ax_init_vec, ay_init_vec = initial_state[:, :, None]

        b_eq_x = th.hstack([x_init_vec, vx_init_vec, ax_init_vec])
        b_eq_y = th.hstack([y_init_vec, vy_init_vec, ay_init_vec, th.zeros((self.num_batch, 1), device=self.device)])

        v_des_1 = neural_output_batch[:, 0]
        v_des_2 = neural_output_batch[:, 1]
        v_des_3 = neural_output_batch[:, 2]
        v_des_4 = neural_output_batch[:, 3]

        y_des_1 = neural_output_batch[:, 4]
        y_des_2 = neural_output_batch[:, 5]
        y_des_3 = neural_output_batch[:, 6]
        y_des_4 = neural_output_batch[:, 7]

        A_pd_1 = self.Pddot1 - (self.k_p * self.P1) - (self.k_d * self.Pdot1)
        b_pd_1 = - self.k_p * th.ones((self.num_batch, self.num_partial), device=self.device) * y_des_1[:, None]

        A_pd_2 = self.Pddot2 - (self.k_p * self.P2) - (self.k_d * self.Pdot2)
        b_pd_2 = - self.k_p * th.ones((self.num_batch, self.num_partial), device=self.device) * y_des_2[:, None]

        A_pd_3 = self.Pddot3 - (self.k_p * self.P3) - (self.k_d * self.Pdot3)
        b_pd_3 = - self.k_p * th.ones((self.num_batch, self.num_partial), device=self.device) * y_des_3[:, None]

        A_pd_4 = self.Pddot4 - (self.k_p * self.P4) - (self.k_d * self.Pdot4)
        b_pd_4 = - self.k_p * th.ones((self.num_batch, self.num_partial), device=self.device) * y_des_4[:, None]

        A_vd_1 = self.Pddot1 - self.k_p_v * self.Pdot1
        b_vd_1 = - self.k_p_v * th.ones((self.num_batch, self.num_partial), device=self.device) * v_des_1[:, None]

        A_vd_2 = self.Pddot2 - (self.k_p_v * self.Pdot2)
        b_vd_2 = - self.k_p_v * th.ones((self.num_batch, self.num_partial), device=self.device) * v_des_2[:, None]

        A_vd_3 = self.Pddot3 - (self.k_p_v * self.Pdot3)
        b_vd_3 = - self.k_p_v * th.ones((self.num_batch, self.num_partial), device=self.device) * v_des_3[:, None]

        A_vd_4 = self.Pddot4 - (self.k_p_v * self.Pdot4)
        b_vd_4 = - self.k_p_v * th.ones((self.num_batch, self.num_partial), device=self.device) * v_des_4[:, None]

        lincost_x = - self.rho_v * th.mm(A_vd_1.T, b_vd_1.T).T \
                    - self.rho_v * th.mm(A_vd_2.T, b_vd_2.T).T \
                    - self.rho_v * th.mm(A_vd_3.T, b_vd_3.T).T \
                    - self.rho_v * th.mm(A_vd_4.T, b_vd_4.T).T

        lincost_y = - self.rho_offset * th.mm(A_pd_1.T, b_pd_1.T).T \
                    - self.rho_offset * th.mm(A_pd_2.T, b_pd_2.T).T \
                    - self.rho_offset * th.mm(A_pd_3.T, b_pd_3.T).T \
                    - self.rho_offset * th.mm(A_pd_4.T, b_pd_4.T).T
        
        sol_x = th.mm(cost_mat_inv_x, th.hstack([-lincost_x, b_eq_x]).T).T
        sol_y = th.mm(cost_mat_inv_y, th.hstack([-lincost_y, b_eq_y]).T).T

        primal_sol_x = sol_x[:, 0:self.nvar]
        primal_sol_y = sol_y[:, 0:self.nvar]

        primal_sol_level_1 = th.hstack([primal_sol_x, primal_sol_y])

        return primal_sol_level_1
    
    def qp_layer_2(
            self, cost_mat_inv_x_layer_2,  cost_mat_inv_y_layer_2, b_eq_x, b_eq_y, lamda_x, lamda_y,
            x_obs_traj, y_obs_traj, primal_sol_level_1, y_lane_lb, y_lane_ub,
            alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, s_lane
    ):

        b_lane_lb = - self.gamma * y_lane_lb[:, None] @ th.ones((1, self.num-1), device=self.device)
        b_lane_ub = self.gamma * y_lane_ub[:, None] @ th.ones((1, self.num-1), device=self.device)

        b_lane_bound = th.hstack([b_lane_ub, b_lane_lb])
        b_lane_aug = b_lane_bound - s_lane

        b_ax_ineq = d_a * th.cos(alpha_a)
        b_ay_ineq = d_a * th.sin(alpha_a)

        b_vx_ineq = d_v * th.cos(alpha_v)
        b_vy_ineq = d_v * th.sin(alpha_v)

        c_x_bar = primal_sol_level_1[:, 0:self.nvar]
        c_y_bar = primal_sol_level_1[:, self.nvar:2 * self.nvar]

        temp_x_obs = d_obs * th.cos(alpha_obs) * self.a_obs
        b_obs_x = x_obs_traj + temp_x_obs

        temp_y_obs = d_obs * th.sin(alpha_obs) * self.b_obs
        b_obs_y = y_obs_traj + temp_y_obs

        lincost_x = - lamda_x - self.rho_projection * th.mm(self.A_projection.T, c_x_bar.T).T \
                    - self.rho_ineq * th.mm(self.A_acc.T, b_ax_ineq.T).T \
                    - self.rho_ineq * th.mm(self.A_vel.T, b_vx_ineq.T).T \
                    - self.rho_obs * th.mm(self.A_obs.T, b_obs_x.T).T \

        lincost_y = - lamda_y - self.rho_projection * th.mm(self.A_projection.T, c_y_bar.T).T \
                    - self.rho_ineq * th.mm(self.A_acc.T, b_ay_ineq.T).T \
                    - self.rho_ineq * th.mm(self.A_vel.T, b_vy_ineq.T).T \
                    - self.rho_obs * th.mm(self.A_obs.T, b_obs_y.T).T \
                    - self.rho_lane * th.mm(self.A_lane_bound.T, b_lane_aug.T).T

        sol_x = th.mm(cost_mat_inv_x_layer_2, th.hstack([-lincost_x, b_eq_x]).T).T
        sol_y = th.mm(cost_mat_inv_y_layer_2, th.hstack([-lincost_y, b_eq_y]).T).T

        primal_sol_x = sol_x[:, 0:self.nvar]
        primal_sol_y = sol_y[:, 0:self.nvar]

        primal_sol_level_2 = th.hstack([primal_sol_x, primal_sol_y])

        return primal_sol_level_2
    
    def comp_d_obs_prev(self, d_obs_prev):
        d_obs_batch = th.reshape(d_obs_prev,(self.num_batch,self.num_circles*self.num_obs,self.num))
        d_obs_batch_modified = th.dstack((th.ones((self.num_batch, self.num_circles*self.num_obs, 1), device=self.device), d_obs_batch[:, :, 0:self.num-1]))
        return d_obs_batch_modified.reshape(self.num_batch, -1)

    def compute_alph_d(
            self, primal_sol_level_2, x_obs_traj, y_obs_traj, y_lane_lb, y_lane_ub,
            lamda_x, lamda_y, d_obs_prev
    ):

        primal_sol_x = primal_sol_level_2[:, 0:self.nvar]
        primal_sol_y = primal_sol_level_2[:, self.nvar:2*self.nvar]

        x = th.mm(self.P, primal_sol_x.T).T
        xdot = th.mm(self.Pdot, primal_sol_x.T).T
        xddot = th.mm(self.Pddot, primal_sol_x.T).T

        y = th.mm(self.P, primal_sol_y.T).T
        ydot = th.mm(self.Pdot, primal_sol_y.T).T
        yddot = th.mm(self.Pddot, primal_sol_y.T).T

        x_extend = th.tile(x, (1, self.num_obs * self.num_circles))
        y_extend = th.tile(y, (1, self.num_obs * self.num_circles))

        wc_alpha = (x_extend - x_obs_traj)
        ws_alpha = (y_extend - y_obs_traj)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num * self.num_obs * self.num_circles)
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num * self.num_obs * self.num_circles)

        alpha_obs = th.arctan2(ws_alpha*self.a_obs, wc_alpha*self.b_obs)

        c1_d = 1.0 * self.rho_obs * (self.a_obs**2 * th.cos(alpha_obs)**2 + self.b_obs**2 * th.sin(alpha_obs)**2)
        c2_d = 1.0 * self.rho_obs * (self.a_obs*wc_alpha * th.cos(alpha_obs) + self.b_obs*ws_alpha * th.sin(alpha_obs) )

        d_temp = c2_d/c1_d
        # d_obs = th.maximum(
        #     th.ones((self.num_batch, self.num * self.num_obs * self.num_circles), device=self.device),
        #     d_temp
        # )
        d_obs_prev = self.comp_d_obs_prev(d_obs_prev)
        d_obs = th.maximum(
            th.ones((self.num_batch, self.num * self.num_obs * self.num_circles), device=self.device) + (1 - self.gamma_obs)*(d_obs_prev - 1),
            d_temp
        )    

        wc_alpha_vx = xdot
        ws_alpha_vy = ydot
        alpha_v = th.arctan2(ws_alpha_vy, wc_alpha_vx)
        # alpha_v = alpha_v.clamp(-th.pi/4, th.pi/4)

        c1_d_v = 1.0 * self.rho_ineq * (th.cos(alpha_v)**2 + th.sin(alpha_v)**2 )
        c2_d_v = 1.0 * self.rho_ineq * (wc_alpha_vx * th.cos(alpha_v) + ws_alpha_vy*th.sin(alpha_v))

        d_temp_v = c2_d_v/c1_d_v
        d_v = th.clip(d_temp_v, self.v_min, self.v_max)

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot
        alpha_a = th.arctan2(ws_alpha_ay, wc_alpha_ax)

        c1_d_a = 1.0 * self.rho_ineq * (th.cos(alpha_a)**2 + th.sin(alpha_a)**2)
        c2_d_a = 1.0 * self.rho_ineq * (wc_alpha_ax * th.cos(alpha_a) + ws_alpha_ay*th.sin(alpha_a))

        d_temp_a = c2_d_a/c1_d_a
        d_a = th.clip(
            d_temp_a,
            th.zeros((self.num_batch, self.num), device=self.device),
            th.tensor(self.a_max, device=self.device)
        )

        res_ax_vec = xddot-d_a * th.cos(alpha_a)
        res_ay_vec = yddot-d_a * th.sin(alpha_a)

        res_vx_vec = xdot-d_v * th.cos(alpha_v)
        res_vy_vec = ydot-d_v * th.sin(alpha_v)

        res_x_obs_vec = wc_alpha - self.a_obs * d_obs * th.cos(alpha_obs)
        res_y_obs_vec = ws_alpha - self.b_obs * d_obs * th.sin(alpha_obs)

        res_vel_vec = th.hstack([res_vx_vec,  res_vy_vec])
        res_acc_vec = th.hstack([res_ax_vec,  res_ay_vec])
        res_obs_vec = th.hstack([res_x_obs_vec, res_y_obs_vec])

        b_lane_lb = - self.gamma * y_lane_lb[:, None] @ th.ones((1, self.num-1), device=self.device)  # N x 99
        b_lane_ub = self.gamma * y_lane_ub[:, None] @ th.ones((1, self.num-1), device=self.device)    # N x 99
        b_lane_bound = th.hstack([b_lane_ub, b_lane_lb])                                               # N x 198

        s_lane = th.maximum(
            th.zeros((self.num_batch, 2*(self.num-1)), device=self.device),
            - th.mm(self.A_lane_bound, primal_sol_y.T).T + b_lane_bound
        )

        res_lane_vec = th.mm(self.A_lane_bound, primal_sol_y.T).T - b_lane_bound + s_lane
        # res_long_vec = th.mm(self.A_barrier_long, primal_sol_x.T).T - b_long_bound + s_long

        obs_constraint = -(wc_alpha/self.a_obs)**2-(ws_alpha/self.a_obs)**2+th.ones(( self.num_batch, self.num  * self.num_obs * self.num_circles), device = self.device)
        obs_penalty = th.linalg.norm(th.maximum( th.ones(( self.num_batch, self.num * self.num_obs * self.num_circles  ), device=self.device), obs_constraint   ), dim = 1)

        # res_norm_batch = th.linalg.norm(res_obs_vec, dim=1) \
        #                  + th.linalg.norm(res_acc_vec, dim=1) \
        #                  + th.linalg.norm(res_vel_vec, dim=1) \
        #                  + th.linalg.norm(res_lane_vec, dim=1)
        
        res_norm_batch = obs_penalty \
                         + th.linalg.norm(res_acc_vec, dim=1) \
                         + th.linalg.norm(res_vel_vec, dim=1) \
                         + th.linalg.norm(res_lane_vec, dim=1)
        

        lamda_x = lamda_x - self.rho_ineq * th.mm(self.A_acc.T, res_ax_vec.T).T \
                  - self.rho_ineq * th.mm(self.A_vel.T, res_vx_vec.T).T \
                  - self.rho_obs * th.mm(self.A_obs.T, res_x_obs_vec.T).T

        lamda_y = lamda_y - self.rho_ineq * th.mm(self.A_acc.T, res_ay_vec.T).T \
                  - self.rho_ineq * th.mm(self.A_vel.T, res_vy_vec.T).T \
                  - self.rho_obs * th.mm(self.A_obs.T, res_y_obs_vec.T).T \
                  - self.rho_lane * th.mm(self.A_lane_bound.T, res_lane_vec.T).T

        return alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, s_lane, res_norm_batch
    
    def custom_forward_q2(self, primal_sol_level_1, observations, initial_state, y_lane_lb, y_lane_ub, x_obs, y_obs, vx_obs, vy_obs):
            
        x_obs_traj, y_obs_traj = self.compute_obs_trajectories(x_obs, y_obs, vx_obs, vy_obs)
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state.mT
        
        b_eq_x, b_eq_y = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)

        cost_mat_inv_x_layer_2, cost_mat_inv_y_layer_2 = self.compute_mat_inv_layer_2
        
        initializer_output = self.initializer_model(primal_sol_level_1, observations)
        primal_sol_level_bar, lamda_x, lamda_y = initializer_output.split([2*self.nvar, self.nvar, self.nvar], -1)
        # primal_sol_level_bar = primal_sol_level_1 + primal_sol_level_bar

        d_obs = th.ones((self.num_batch, self.num*self.num_obs*self.num_circles), device=self.device)
        (alpha_obs, d_obs, alpha_a, d_a,
            lamda_x, lamda_y, alpha_v, d_v,
            s_lane, res_norm_batch
        ) = self.compute_alph_d(
             primal_sol_level_bar, x_obs_traj, y_obs_traj, y_lane_lb, y_lane_ub, lamda_x, lamda_y, d_obs
        )
        
        accumulated_res = 0
        for i in range(self.maxiter):
            primal_sol_level_2 = self.qp_layer_2(
                    cost_mat_inv_x_layer_2,  cost_mat_inv_y_layer_2, b_eq_x, b_eq_y,
                    lamda_x, lamda_y, x_obs_traj, y_obs_traj, primal_sol_level_bar, y_lane_lb, y_lane_ub,
                    alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, s_lane
            )
            (alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y,
                alpha_v, d_v, s_lane, res_norm_batch
            ) = self.compute_alph_d(
                primal_sol_level_2, x_obs_traj, y_obs_traj, y_lane_lb, y_lane_ub,
                lamda_x, lamda_y, d_obs
            )
            accumulated_res += res_norm_batch
        
        accumulated_res = accumulated_res/self.maxiter
        
        return primal_sol_level_2, accumulated_res, res_norm_batch
    
    def custom_forward(self, observations, initial_state, y_lane_lb, y_lane_ub, neural_output_batch, x_obs, y_obs, vx_obs, vy_obs):

        x_obs_traj, y_obs_traj = self.compute_obs_trajectories(x_obs, y_obs, vx_obs, vy_obs)
        x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state.mT

        b_eq_x, b_eq_y = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)

        cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_layer_1

        cost_mat_inv_x_layer_2, cost_mat_inv_y_layer_2 = self.compute_mat_inv_layer_2

        primal_sol_level_1 = self.qp_layer_1(initial_state.mT, neural_output_batch, cost_mat_inv_x, cost_mat_inv_y)
        if self.qp1_train:
            return primal_sol_level_1, primal_sol_level_1, 0, 0

        initializer_output = self.initializer_model(primal_sol_level_1, observations)
        primal_sol_level_bar, lamda_x, lamda_y = initializer_output.split([2*self.nvar, self.nvar, self.nvar], -1)
        # primal_sol_level_bar = primal_sol_level_1 + primal_sol_level_bar

        d_obs = th.ones((self.num_batch, self.num*self.num_obs*self.num_circles), device=self.device)
        (alpha_obs, d_obs, alpha_a, d_a,
                lamda_x, lamda_y, alpha_v, d_v,
                s_lane, res_norm_batch
         ) = self.compute_alph_d(
                 primal_sol_level_bar, x_obs_traj, y_obs_traj, y_lane_lb, y_lane_ub, lamda_x, lamda_y, d_obs
         )

        accumulated_res = 0
        for i in range(self.maxiter):
            primal_sol_level_2 = self.qp_layer_2(
                cost_mat_inv_x_layer_2,  cost_mat_inv_y_layer_2, b_eq_x, b_eq_y,
                lamda_x, lamda_y, x_obs_traj, y_obs_traj, primal_sol_level_bar, y_lane_lb, y_lane_ub,
                alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, s_lane
            )
            (alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y,
                    alpha_v, d_v, s_lane, res_norm_batch
             ) = self.compute_alph_d(
                 primal_sol_level_2, x_obs_traj, y_obs_traj, y_lane_lb, y_lane_ub,
                 lamda_x, lamda_y, d_obs
             )
            accumulated_res += res_norm_batch

        accumulated_res = accumulated_res/self.maxiter

        return primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch
    
    def compute_controls(self, c_x_best, c_y_best):

        xdot_best = self.Pdot @ c_x_best    
        ydot_best = self.Pdot @ c_y_best    

        xddot_best = self.Pddot @ c_x_best  
        yddot_best = self.Pddot @ c_y_best  

        curvature_best = (yddot_best * xdot_best - ydot_best * xddot_best) / ((xdot_best**2 + ydot_best**2)**1.5)  # N x 100
        steer_best = th.arctan(curvature_best*self.wheel_base)                                                     # N x 100

        v_best = th.sqrt(xdot_best ** 2 + ydot_best ** 2)  
        a_best = th.diff(v_best, dim=-1) / self.t          

        return v_best, steer_best
    
    def compute_obs_trajectories(self, x_obs, y_obs, vx_obs, vy_obs):

        x_obs_inp_trans = x_obs.reshape(self.num_batch, 1, self.num_obs)    
        y_obs_inp_trans = y_obs.reshape(self.num_batch, 1, self.num_obs)    

        vx_obs_inp_trans = vx_obs.reshape(self.num_batch, 1, self.num_obs)  
        vy_obs_inp_trans = vy_obs.reshape(self.num_batch, 1, self.num_obs) 

        x_obs_traj = x_obs_inp_trans + vx_obs_inp_trans * self.tot_time[:, None]
        y_obs_traj = y_obs_inp_trans + vy_obs_inp_trans * self.tot_time[:, None]

        x_obs_traj = x_obs_traj.permute(0, 2, 1)  
        y_obs_traj = y_obs_traj.permute(0, 2, 1)  

        vx_obs_traj = th.tile(vx_obs_inp_trans, (self.num, 1))  
        vy_obs_traj = th.tile(vy_obs_inp_trans, (self.num, 1))  

        vx_obs_traj = vx_obs_traj.permute(0, 2, 1)  
        vy_obs_traj = vy_obs_traj.permute(0, 2, 1)  

        psi_obs_traj = th.arctan2(vy_obs_traj, vx_obs_traj)  

        x_obs_circles, y_obs_circles = self.split(x_obs_traj, y_obs_traj, psi_obs_traj)

        return x_obs_circles, y_obs_circles

    def split(self, x_obs_traj, y_obs_traj, psi_obs_traj): 
        dist_centre = th.tile(self.dist_centre, (self.num, 1)).T  
        dist_centre = th.tile(dist_centre, (self.num_obs, 1)).reshape(self.num_obs, self.num_circles, -1)  

        psi_obs_traj = th.tile(psi_obs_traj, (self.num_circles, 1)).reshape(self.num_batch, self.num_obs, self.num_circles, -1)
        x_obs_traj = th.tile(x_obs_traj, (self.num_circles,)).reshape(self.num_batch, self.num_obs, self.num_circles, -1)
        y_obs_traj = th.tile(y_obs_traj, (self.num_circles,)).reshape(self.num_batch, self.num_obs, self.num_circles, -1)

        x_temp = dist_centre*th.cos(psi_obs_traj)
        y_temp = dist_centre*th.sin(psi_obs_traj)

        x_obs_circles = x_obs_traj + x_temp
        y_obs_circles = y_obs_traj + y_temp

        x_obs_circles = x_obs_circles.reshape(self.num_batch, self.num_circles*self.num_obs*self.num)
        y_obs_circles = y_obs_circles.reshape(self.num_batch, self.num_circles*self.num_obs*self.num)

        return x_obs_circles, y_obs_circles
    
    def compute_cost(self, primal_sol_level_2, res_norm_batch):

        primal_sol_x = primal_sol_level_2[:, 0:self.nvar]
        primal_sol_y = primal_sol_level_2[:, self.nvar:2*self.nvar]

        x = th.mm(self.P, primal_sol_x.T).T
        xdot = th.mm(self.Pdot, primal_sol_x.T).T
        xddot = th.mm(self.Pddot, primal_sol_x.T).T

        y = th.mm(self.P, primal_sol_y.T).T
        ydot = th.mm(self.Pdot, primal_sol_y.T).T
        yddot = th.mm(self.Pddot, primal_sol_y.T).T

        curvature = (yddot*xdot - xddot*ydot) / ((xdot**2 + ydot**2)**1.5)
        steer = th.arctan(curvature * self.wheel_base)

        idx_ellites = th.argsort(res_norm_batch)

        x_ellites = x[idx_ellites[0:self.ellite_num]]
        y_ellites = y[idx_ellites[0:self.ellite_num]]

        xdot_ellites = xdot[idx_ellites[0:self.ellite_num]]
        ydot_ellites = ydot[idx_ellites[0:self.ellite_num]]

        xddot_ellites = xddot[idx_ellites[0:self.ellite_num]]
        yddot_ellites = yddot[idx_ellites[0:self.ellite_num]]

        steer_ellites = steer[idx_ellites[0:self.ellite_num]]
        res_ellites = res_norm_batch[idx_ellites[0:self.ellite_num]]

        cost_steering = th.linalg.norm(steer_ellites, dim=1)
        steering_vel = th.diff(steer_ellites, dim=1)
        cost_steering_vel = th.linalg.norm(steering_vel, dim=1)

        heading_angle = th.atan2(ydot_ellites, xdot_ellites)
        heading_penalty = th.linalg.norm(
            th.maximum(
                th.zeros((self.ellite_num, self.num), device=self.device),
                th.abs(heading_angle)-10*th.pi/180
            ),
            dim=1
        )
        centerline_cost = th.linalg.norm(y_ellites, dim=1)
        v = th.sqrt(xdot_ellites**2 + ydot_ellites**2)

        filter = th.exp(-2*res_ellites)
        vel_pen = th.linalg.norm(x_ellites - self.v_des*self.t_fin, dim=1)
            
        # vel = th.sqrt(xdot_ellites ** 2 + ydot_ellites ** 2)
        # vel_pen = th.linalg.norm(vel - self.v_des)

        # smoothness_pen = th.linalg.norm(yddot_ellites, dim=1) + th.linalg.norm(xddot_ellites, dim=1)
            
        # cost_batch = 1*res_ellites + 1e-3*vel_pen\
        #          + 0.01*cost_steering + 0.01*cost_steering_vel + 0.01*heading_penalty
            
        cost_batch = 1*res_ellites + 1e-3*filter*vel_pen\
                     + 0.01*cost_steering + 0.01*cost_steering_vel + 0.01*heading_penalty

        return cost_batch, primal_sol_level_2[idx_ellites[0:self.ellite_num]]
    
    def __call__(self, obs, neural_output_batch):
        ub, lb = obs[:, 0], obs[:, 1]   
        vx, vy = obs[:, 2], obs[:, 3]    
        v_init = th.sqrt(vx**2 + vy**2) 

        x = th.zeros_like(v_init) 
        y = th.zeros_like(v_init)  
        ax = th.zeros_like(v_init)  
        ay = th.zeros_like(v_init) 

        initial_state = th.vstack([x, y, vx, vy, ax, ay]).T  

        x_obs_temp = obs[:, 5::5]  
        y_obs_temp = obs[:, 6::5]  
        vx_obs_temp = obs[:, 7::5] 
        vy_obs_temp = obs[:, 8::5]  

        x_obs, y_obs, vx_obs, vy_obs = x_obs_temp, y_obs_temp, vx_obs_temp, vy_obs_temp  # same sizes as before
        primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch = self.custom_forward(
            obs, initial_state, lb, ub, neural_output_batch,
            x_obs, y_obs, vx_obs, vy_obs
        )  

        primal_sol_x = primal_sol_level_2[:, 0:self.nvar]
        primal_sol_y = primal_sol_level_2[:, self.nvar:2*self.nvar]
        
        x = (self.P @ primal_sol_x.T).T
        y = (self.P @ primal_sol_y.T).T
        trajs = th.hstack([x, y])

        return trajs, primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch, v_init
    
    def forward_q2(self, obs, primal_sol_level_1):
        ub, lb = obs[:, 0], obs[:, 1]   
        vx, vy = obs[:, 2], obs[:, 3]    
        v_init = th.sqrt(vx**2 + vy**2) 

        x = th.zeros_like(v_init) 
        y = th.zeros_like(v_init)  
        ax = th.zeros_like(v_init)  
        ay = th.zeros_like(v_init) 

        initial_state = th.vstack([x, y, vx, vy, ax, ay]).T  

        x_obs_temp = obs[:, 5::5]  
        y_obs_temp = obs[:, 6::5]  
        vx_obs_temp = obs[:, 7::5] 
        vy_obs_temp = obs[:, 8::5]  

        x_obs, y_obs, vx_obs, vy_obs = x_obs_temp, y_obs_temp, vx_obs_temp, vy_obs_temp  # same sizes as before
        primal_sol_level_2, accumulated_res, res_norm_batch = self.custom_forward_q2(
                primal_sol_level_1, obs, initial_state, lb, ub,
                x_obs, y_obs, vx_obs, vy_obs
        )  

        primal_sol_x = primal_sol_level_2[:, 0:self.nvar]
        primal_sol_y = primal_sol_level_2[:, self.nvar:2*self.nvar]

        x = (self.P @ primal_sol_x.T).T
        y = (self.P @ primal_sol_y.T).T
        trajs = th.hstack([x, y])

        return trajs, primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch, v_init
    
    def inference(self, obs, neural_output_batch):
        _, primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch, v_init = self(obs, neural_output_batch)
        v_init = v_init[0]
        
        primal_sol_x = primal_sol_level_2[:, 0:self.nvar]
        primal_sol_y = primal_sol_level_2[:, self.nvar:2*self.nvar]

        x = (self.P @ primal_sol_x.T).T
        y = (self.P @ primal_sol_y.T).T
        all_trajs = th.stack([x, y], dim=-1)

        cost_ellite_batch, primal_sol_level_2_ellites = self.compute_cost(primal_sol_level_2, res_norm_batch)
        idx_min = th.argmin(cost_ellite_batch)

        primal_sol_level_2 = primal_sol_level_2_ellites
        primal_sol_x = primal_sol_level_2[:, 0:self.nvar][idx_min]
        primal_sol_y = primal_sol_level_2[:, self.nvar:2*self.nvar][idx_min]

        primal_sol_level_2 = th.hstack([primal_sol_x, primal_sol_y])

        primal_sol_x = primal_sol_level_2[0:self.nvar]
        primal_sol_y = primal_sol_level_2[self.nvar:2*self.nvar]

        x = self.P @ primal_sol_x
        y = self.P @ primal_sol_y

        v_best, steer_best = self.compute_controls(primal_sol_x, primal_sol_y)

        v_control = th.mean(v_best[0:self.num_mean_update])
        steer_control = th.mean(steer_best[0:self.num_mean_update])

        a_control = (v_control-v_init)/self.t_target
        control = th.hstack([a_control, steer_control])
        opt_traj = th.stack([x, y], dim=-1)

        return control, all_trajs, opt_traj
        
