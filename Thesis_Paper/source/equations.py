"""Display equations. The same LaTeX string is rendered to PNG (matplotlib mathtext) for the
Word files and exported verbatim to latex_support/equations.tex."""

EQUATIONS: dict[str, str] = {
    # ---------------- Chapter 2
    "eq_lora": r"h = W_0\,x + \Delta W x = W_0\,x + \frac{\alpha}{r}\,B A\,x,\qquad B\in\mathbb{R}^{O\times r},\; A\in\mathbb{R}^{r\times I},\; r \ll \min(O, I)",
    "eq_rl_objective": r"J(\theta) = \mathbb{E}_{x\sim\mathcal{D},\; y\sim\pi_\theta(\cdot\,|\,x)}\left[\, r(x, y)\,\right] - \beta\,\mathbb{E}_{x\sim\mathcal{D}}\left[\mathrm{KL}\left(\pi_\theta(\cdot\,|\,x)\,\|\,\pi_{\mathrm{ref}}(\cdot\,|\,x)\right)\right]",
    "eq_ewc": r"\mathcal{L}(\theta) = \mathcal{L}_{t}(\theta) + \sum_{k} \frac{\lambda}{2}\,F_k\left(\theta_k - \theta^{*}_{t-1,k}\right)^2",
    "eq_si": r"\omega_k = -\sum_{s} g_k^{(s)}\,\Delta\theta_k^{(s)},\qquad \Omega_k = \frac{\omega_k}{\left(\theta_k^{\mathrm{end}} - \theta_k^{\mathrm{start}}\right)^2 + \xi}",
    # ---------------- Chapter 4: formulation
    "eq_stream": r"\mathcal{S} = \left(\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_T\right),\qquad \mathcal{D}_t = \mathcal{D}_t^{\mathrm{train}} \cup \mathcal{D}_t^{\mathrm{val}} \cup \mathcal{D}_t^{\mathrm{anchor}} \cup \mathcal{D}_t^{\mathrm{test}}\ \ (\mathrm{disjoint})",
    "eq_policy": r"\pi_\theta(y\,|\,x) = \prod_{j=1}^{|y|} p_{W_0,\theta}\left(y_j \,|\, x, y_{<j}\right),\qquad \theta = \{A_l, B_l\}_{l=1}^{L}\ \ (\mathrm{one\ shared\ adapter})",
    "eq_effective": r"\Delta W_l = s_l\,B_l A_l,\qquad s_l = \frac{\alpha}{r}",
    "eq_gauge": r"B_l' A_l' = \left(B_l Q\right)\left(Q^{-1} A_l\right) = B_l A_l \quad \mathrm{for\ every\ invertible}\ Q\in\mathbb{R}^{r\times r}\quad (\mathrm{e.g.}\ Q = k^{-1} I)",
    "eq_grpo_adv": r"\hat{A}_i = \mathrm{clip}\left(\frac{r_i - \mu_g}{\sigma_g},\,-c,\,c\right)\ \ \mathrm{if}\ \sigma_g > \epsilon_A,\qquad \hat{A}_i = 0\ \ \mathrm{otherwise}",
    "eq_grpo_loss": r"\mathcal{L}_{\mathrm{RL}} = -\frac{1}{\sum_i |y_i|}\sum_{i=1}^{G}\sum_{j=1}^{|y_i|} \min\left(\rho_{i,j}\hat{A}_i,\; \mathrm{clip}\left(\rho_{i,j}, 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right),\qquad \rho_{i,j} = \frac{\pi_\theta(y_{i,j}\,|\,x, y_{i,<j})}{\pi_{\mathrm{old}}(y_{i,j}\,|\,x, y_{i,<j})}",
    "eq_eff_grad": r"g_l = \frac{\partial \mathcal{L}_{\mathrm{RL}}}{\partial \Delta W_l} = \sum_{b,j} \frac{\partial \mathcal{L}_{\mathrm{RL}}}{\partial h_{l,b,j}}\; x_{l,b,j}^{\top}\ \in \mathbb{R}^{O\times I}",
    "eq_taylor": r"\mathcal{L}\left(\Delta W + \delta\right) - \mathcal{L}\left(\Delta W\right) = \left\langle g, \delta\right\rangle + \frac{1}{2}\,\mathrm{vec}(\delta)^{\top} H\,\mathrm{vec}(\delta) + O\left(\|\delta\|^3\right)",
    "eq_step_delta": r"\delta_l^{(s)} = \Delta W_l^{(s+1)} - \Delta W_l^{(s)}\qquad (\mathrm{observed\ after\ clipping\ and\ the\ AdamW\ step})",
    "eq_raw_score": r"\tilde{S}_l^{(s)} = \frac{\mathrm{ReLU}\left(-\,g_l^{(s)} \odot \delta_l^{(s)}\right)}{\delta_l^{(s)} \odot \delta_l^{(s)} + \epsilon}",
    "eq_ema": r"S_l^{(s)} = \beta\,S_l^{(s-1)} + (1-\beta)\,\mathrm{clip}_{q}\left(\tilde{S}_l^{(s)}\right),\qquad \beta = 0.99,\ q = 0.995",
    "eq_rank1": r"S_l \approx p_l\,q_l^{\top},\qquad p_l = S_l\,\mathbf{1},\qquad q_l = \frac{S_l^{\top}\mathbf{1}}{\mathbf{1}^{\top} S_l\,\mathbf{1}},\qquad \mathbf{1}^{\top}p_l\,q_l^{\top}\mathbf{1} = \mathbf{1}^{\top}S_l\mathbf{1}",
    "eq_D": r"D_l = s_l\left(B_l A_l - B_l^{\mathrm{ref}} A_l^{\mathrm{ref}}\right) = \Delta W_l - \Delta W_l^{\mathrm{ref}}",
    "eq_R": r"R_l = \left\| \mathrm{diag}\left(\sqrt{p_l}\right)\,D_l\,\mathrm{diag}\left(\sqrt{q_l}\right) \right\|_F^2 = \sum_{i,j} p_{l,i}\,q_{l,j}\,D_{l,ij}^2",
    "eq_gram": r"X_l = \sqrt{s_l}\left[\,P B_l,\; -P B_l^{\mathrm{ref}}\,\right],\quad Y_l = \sqrt{s_l}\left[\,A_l Q;\; A_l^{\mathrm{ref}} Q\,\right]\ (\mathrm{row\ stack}),\quad R_l = \mathrm{tr}\left(\left(X_l^{\top}X_l\right)\left(Y_l Y_l^{\top}\right)\right)",
    "eq_kl": r"\mathcal{L}_{\mathrm{KL}} = \frac{1}{|\mathcal{M}_b|}\sum_{m\in\mathcal{M}_b}\frac{1}{|y_m|}\sum_{j}\left[\sum_{v\in K_{m,j}} p^{\mathrm{ref}}_{v}\log\frac{p^{\mathrm{ref}}_{v}}{\pi_\theta(v)} + \bar{p}^{\mathrm{ref}}\log\frac{\bar{p}^{\mathrm{ref}}}{\bar{\pi}_\theta}\right],\quad \bar{p} = 1 - \sum_{v\in K} p_v",
    "eq_conflict": r"\kappa_l = \max\left(0,\; -\frac{\langle g_l^{\mathrm{RL}}, g_l^{\mathrm{anc}}\rangle}{\|g_l^{\mathrm{RL}}\|_F\,\|g_l^{\mathrm{anc}}\|_F + \epsilon}\right),\qquad \kappa_l = 0\ \mathrm{if}\ \min\left(\|g_l^{\mathrm{RL}}\|, \|g_l^{\mathrm{anc}}\|\right) < \tau",
    "eq_coef": r"m_l = \sqrt{\left(\mathbf{1}^{\top}p_l\right)\left(\mathbf{1}^{\top}q_l\right)},\qquad c_l = \mathrm{sg}\left[\min\left(c_{\max},\; \frac{L\,m_l\,\kappa_l}{\sum_{k=1}^{L} m_k\,\kappa_k + \epsilon}\right)\right]",
    "eq_dual": r"\bar{K}_s = \beta_K\,\bar{K}_{s-1} + (1-\beta_K)\,\mathcal{L}_{\mathrm{KL}}^{(s)},\qquad \lambda_f \leftarrow \min\left(\lambda_{\max},\; \max\left(0,\; \lambda_f + \eta_\lambda\left(\bar{K}_s - K^{*}\right)\right)\right)",
    "eq_total": r"\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{RL}} + \lambda_f\,\mathcal{L}_{\mathrm{KL}} + \lambda_p\sum_{l=1}^{L} c_l\,R_l",
    "eq_forgetting": r"F = \frac{1}{T-1}\sum_{j=1}^{T-1}\left(\max_{j \leq i < T} A_{i,j} - A_{T,j}\right)",
    "eq_final_avg": r"\bar{A}_T = \frac{1}{T}\sum_{j=1}^{T} A_{T,j},\qquad \mathrm{BWT} = \frac{1}{T-1}\sum_{j=1}^{T-1}\left(A_{T,j} - A_{j,j}\right)",
    "eq_wilson": r"\hat{p} \pm \Delta = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}},\qquad p_{\mathrm{McNemar}} = \min\left(1,\; 2\sum_{i=0}^{\min(b,c)} \binom{b+c}{i} 2^{-(b+c)}\right)",
    "eq_storage": r"|S_{\mathrm{dense}}| = \sum_{l} O_l\,I_l,\qquad |S_{\mathrm{rank1}}| = \sum_{l}\left(O_l + I_l\right)",
    # ---------------- Chapter 5: implemented HLoRA-RL
    "eq_hlora_path": r"\omega_l \leftarrow \omega_l - G_l \odot \left(U_l^{\mathrm{after}} - U_l^{\mathrm{before}}\right),\qquad U_l = s_l B_l A_l",
    "eq_hlora_omega": r"\Omega_l \leftarrow \Omega_l + \frac{\mathrm{ReLU}\left(\omega_l\right)}{\left(U_l^{\mathrm{end}} - U_l^{\mathrm{start}}\right)^{\odot 2} + \epsilon},\qquad \alpha_l = \frac{\exp\left(\|\Omega_l\|_2\right)}{\sum_{k}\exp\left(\|\Omega_k\|_2\right)}",
    "eq_hlora_loss": r"\mathcal{L} = \mathcal{L}_{\mathrm{RL}} + \lambda\sum_{l}\alpha_l\sum_{i,j}\Omega_{l,ij}\left(U_{l,ij} - U^{\mathrm{ref}}_{l,ij}\right)^2",
}
