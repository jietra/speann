"""
SPEANN Model Implementation
This module implements the SPEANN model, which simulates neural activity and synaptic plasticity.
This model can be used to train networks with specific synaptic dynamics.
"""

import math
from dataclasses import dataclass
import torch


@dataclass
class States:
    """
    States for the SPEANN model.
    This class holds the states used in the SPEANN model.
    Attributes:
        x (torch.Tensor): Neural activity tensor (Time step x Number of neurons).
        u (torch.Tensor): Membrane potential tensor (T x N).
        v (torch.Tensor): Synaptic potential tensor (T x N x N).
        w (torch.Tensor): Synaptic weights tensor (T x N x N).
        nij (torch.Tensor): Myelination state tensor.
        s (torch.Tensor): External stimulations history.
    """
    x: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    nij: torch.Tensor
    s: torch.Tensor

@dataclass
class HHStates:
    """
    Hodgkin Huxley states:
    Attributes:
        v (torch.Tensor): (N)
        m (torch.Tensor): (N)
        h (torch.Tensor): (N)
        n (torch.Tensor): (N)
        m_ca (torch.Tensor): (N)
    """
    v: torch.Tensor
    m: torch.Tensor
    h: torch.Tensor
    n: torch.Tensor
    m_ca: torch.Tensor

@dataclass
class AuxiliaryStates:
    """
    Auxiliary states for the SPEANN model.
    This class holds the auxiliary states used in the SPEANN model.
    Attributes:
        xx (torch.Tensor): Auxiliary state for synaptic plasticity.
        xxx (torch.Tensor): Relative view of synaptic connections.
        x_ (torch.Tensor): Dendritic trace tensor.
    """
    xx: torch.Tensor
    xxx: torch.Tensor
    x_: torch.Tensor


@dataclass
class SynapticPlasticityStates:
    """
    Synaptic plasticity states for the SPEANN model.
    This class holds the synaptic plasticity states used in the SPEANN model.
    Attributes:
        up (torch.Tensor): Synaptic plasticity state for potentiation.
        ud (torch.Tensor): Synaptic plasticity state for depression.
        y (torch.Tensor): Synaptic plasticity state for activity.
        p (torch.Tensor): Synaptic plasticity state for probability.
        z (torch.Tensor): Synaptic plasticity state for synaptic strength.
    """
    up: torch.Tensor
    ud: torch.Tensor
    y: torch.Tensor
    p: torch.Tensor
    z: torch.Tensor


@dataclass
class ModelContext:
    """
    Context for the SPEANN model.
    This class holds the context used in the SPEANN model.
    Attributes:
        T_ (int): Number of time steps.
        N_ (int): Number of neurons.
        dtype (torch.dtype): Data type for tensors.
        device (torch.device): Device for tensors.
    """
    time_steps: int
    neuron_count: int
    dtype: torch.dtype
    device: torch.device


@dataclass
class ModelData:
    """
    Model data for the SPEANN model.
    This class holds the data used in the SPEANN model.
    Attributes:
        states (States): States of the model.
        auxilary_states (AuxiliaryStates): Auxiliary states of the model.
        plasticity_states (SynapticPlasticityStates): Synaptic plasticity states of the model.
        context (ModelContext): Context of the model.
        parameters (dict): Parameters for the SPEANN model.
    """
    states: States
    hhstates: HHStates
    auxilary_states: AuxiliaryStates
    plasticity_states: SynapticPlasticityStates
    context: ModelContext
    parameters: dict


class SpeannModel:
    """
    SPEANN Model Class
    This class implements the SPEANN model, which simulates neural activity and synaptic plasticity.
    It includes methods for initializing the model, and running simulations.
    Attributes:
        data (ModelData): Data for the SPEANN model.
        i_ext (torch.Tensor): External current tensor.
        w0 (torch.Tensor): Initial synaptic weights.
    Methods:
        initialize_variables: Initializes all matrices of the simulation.
        run_simulation: Runs the main simulation loop.
        update_x_: Updates the dendritic trace _X_.
        update_i_ext: Updates the external current I_ext.
        update_u: Updates the membrane potential U_.
        update_x: Updates the neural activity X.
        update_xx: Updates the auxiliary state XX.
        x_relative_view: Computes a relative view of synaptic connections.
        update_xxx: Updates the relative view of synaptic connections XXX.
        update_neurons: Updates the state of neurons.
        update_synaptic_plasticity: Updates synaptic plasticity states.
        stimulation: Generates a stimulation pattern for training.
        train: Trains the model with specified parameters.
    """
    def __new__(cls, *args, **kwargs):
        """Override __new__ to ensure only one instance of the class is created."""
        if not hasattr(cls, 'instance'):
            cls.instance = super(SpeannModel, cls).__new__(cls)
        return cls.instance

    def __init__(self, data: ModelData):  # D, C, sp.state()
        """Initializes the SPEANN model with given parameters and data."""
        # If the instance is already initialized, do not reinitialize
        if hasattr(self, 'data'):
            return
        self.data = data
        self.i_ext = None
        self.w0 = None  # Initial synaptic weights
        # Initialize the model variables
        self.initialize_variables()

        # Initialize the model data
        # self.data.states.x = data.x      # 2 dim tensor (Time step x Number of neurons)
        # self.w = data.w                  # 3 dim tensor (T X N x N)
        # self.nij = data.nij
        # self.data.states.s = data.s      # stores all external stimulations history
        # stores all constant parameters (float, int...)
        # self.data.parameters = data.parameters

        # Internal attributes
        # self.data.auxilary_states.xx   = XX
        # self.data.auxilary_states.xxx  = XXX
        # self.data.auxilary_states.x_  = _X_              # dendritic trace T x N x N

        # synaptic plasticity attributes
        # self.data.plasticity_states.up = UP
        # self.data.plasticity_states.ud = UD
        # self.Y = Y #torch.zeros((self.T_, self.N_, self.N_), dtype=dtype, device=device)
        # self.data.plasticity_states.p = P
        # z = Z #torch.zeros((self.T_, self.N_, self.N_), dtype=dtype, device=device)

        # Initialization of auxiliary states and synaptic plasticity
        # self.aux = AuxiliaryStates(XX, XXX, _X_)
        # self.plasticity = SynapticPlasticityStates(UP, UD, Y, P, Z)

        # Dimensions and device
        # self.T_, self.N_ = X.shape  # T_ is a var, while N_ is const
        # self.dtype, self.data.context.device = X.dtype, X.device

        # Initialisation des variables

    def initialize_variables(self):
        """Initialize all matrices of the simulation"""
        # t, n, device
        #self.data.context.time_steps = data.context.time_steps if data.context.time_steps else 100
        #self.data.context.neuron_count = data.context.neuron_count \
        # if data.context.neuron_count \
        # else data.parameters["num_clusters"] * data.parameters["neurons_per_cluster"]
        #t, n = self.data.context.time_steps, self.data.context.neuron_count

        #self.data.context.dtype = data.context.dtype if data.context.dtype else torch.float
        #self.data.context.device = data.context.device \
        # if data.context.device \
        # else ("cuda" \
        # if torch.cuda.is_available() \
        # else "mps" if torch.backends.mps.is_available() else "cpu")
        #dtype, device = self.data.context.dtype, self.data.context.device

        #nij, UP, UD, Y, P, Z, parameters
        #self.data.parameters = data.parameters if data.parameters else parameters ### <===
        #self.data.states.s = data.states.s if data.states.s else torch.zeros(t, n, device=device)
        #self.data.states.x = data.states.x if data.states.x else torch.zeros(t, n, device=device)
        #w0 = self.data.parameters["w0"] # <> self.w0 vs self.data.states.w
        #self.data.states.w = data.states.w \
        # if data.states.w \
        # else torch.full(
        # (t, n, n), w0, dtype=dtype, device=device)-w0*torch.eye(n, dtype=dtype, device=device)
        #nij
        #self.data.auxilary_states.xx = data.auxilary_states.xx \
        # if data.auxilary_states.xx \
        # else torch.zeros(t, n, device=device)
        #self.data.auxilary_states.xxx = data.auxilary_states.xxx \
        # if data.auxilary_states.xxx \
        # else torch.zeros((n, n), device=device)
        #self.data.auxilary_states.x_ = data.auxilary_states.x_ \
        # if data.auxilary_states.x_ \
        # else torch.zeros((t, n, n), device=device)


        alpha, beta = self.data.parameters["alpha"], self.data.parameters["beta"]
        u0, uf = self.data.parameters["u0"], self.data.parameters["uf"]
        c_m, epsilon = self.data.parameters["C_m"], self.data.parameters["epsilon"]
        s, w = self.data.states.s, self.data.states.w
        y, z = self.data.plasticity_states.y, self.data.plasticity_states.z
        #dtype, device = self.data.context.dtype, self.data.context.device
        #self.data.states.u = torch.full((t, n), u0, dtype=dtype, device=device)
        # External current (normalized)
        self.i_ext = s[-1] * (uf-u0) * c_m / epsilon
        # Non tagged and non consolidated weights
        self.w0 = w[-1] / (1 + y * ((y - 1) / 2 - alpha * (2 - y)) + beta * z)

        # Hodgkin Huxley states initialization:
        #self.data.hhstates.v = torch.full((n,), -65.0, device=device)
        #self.data.hhstates.m = torch.full((n,), 0.05, device=device)
        #self.data.hhstates.h = torch.full((n,), 0.6, device=device)
        #self.data.hhstates.n = torch.full((n,), 0.32, device=device)
        #self.data.hhstates.m_ca = torch.full((n), , device=device)

        # Myelinization
        # self.m = torch.zeros(n, dtype=dtype, device=device)  # when M in param init?

    # Fonction alpha et beta pour les variables de gating
    # def alpha_m(self, V): return 0.1 * (V + 40) / (1 - torch.exp(-(V + 40) / 10))
    # def beta_m(self, V): return 4.0 * torch.exp(-(V + 65) / 18)
    # def m(self, V): return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))

    # def alpha_h(self, V): return 0.07 * torch.exp(-(V + 65) / 20)
    # def beta_h(self, V): return 1 / (1 + torch.exp(-(V + 35) / 10))
    # def h(self, V): return self.alpha_h(V) / (self.alpha_h(V) + self.beta_h(V))

    # def alpha_n(self, V): return 0.01 * (V + 55) / (1 - torch.exp(-(V + 55) / 10))
    # def beta_n(self, V): return 0.125 * torch.exp(-(V + 65) / 80)
    # def n(self, V): return self.alpha_n(V) / (self.alpha_n(V) + self.beta_n(V))

    # def alpha_m_Ca(self, V): return 1 / (1 + torch.exp(-(V + 50) / 7.4))
    # def beta_m_Ca(self, V): return 1 / (1 + torch.exp((V + 75) / 5.0))
    # def m_Ca(self, V): return self.alpha_m_Ca(V) / (self.alpha_m_Ca(V) + self.beta_m_Ca(V))
    # def h_Ca(self, V): return 1 / (1 + torch.exp((V + 60) / 5.0))

    # Courants ioniques
    # def I_Na(self, V, m, h): return self.data.parameters["g_Na"] * \
    # (V - self.data.parameters["E_Na"]) * m**3 * h
    # def I_K(self, V, n): return self.data.parameters["g_K"] * \
    # (V - self.data.parameters["E_K"]) * n**4
    # def I_L(self, V): return self.data.parameters["g_L"] * \
    # (V - self.data.parameters["E_L"])
    # def I_Ca(self, V, m, h): return self.data.parameters["g_Ca"] * \
    # (V - self.data.parameters["E_Ca"]) * m**2 * h

    def run_simulation(self, stimulation):
        """Execute the main simulation loop"""
        t_simulation = stimulation.shape[0]
        n = self.data.context.neuron_count
        dtype, device = self.data.context.dtype, self.data.context.device
        padding_x = torch.zeros(t_simulation, n, dtype=dtype, device=device)
        padding_w = torch.zeros(t_simulation, n, n, dtype=dtype, device=device)
        self.data.states.x = torch.cat([padding_x, self.data.states.x], dim=0)
        self.data.auxilary_states.xx = torch.cat(
            [padding_x, self.data.auxilary_states.xx], dim=0)
        self.data.states.u = torch.cat([padding_x, self.data.states.u], dim=0)
        self.data.states.w = torch.cat([padding_w, self.data.states.w], dim=0)
        self.data.auxilary_states.x_ = torch.cat(
            [padding_w, self.data.auxilary_states.x_], dim=0)
        self.data.states.nij = torch.cat(
            [padding_w, self.data.states.nij], dim=0)
        self.data.states.s = torch.cat(
            [stimulation, self.data.states.s], dim=0)
        self.data.context.time_steps += t_simulation
        for n in range(t_simulation):
            print(
                f"--------------- Simulation step {n+1}/{t_simulation} ---------------------------")
            # self.data.plasticity_states.update_myelinization()         #update nij
            self.update_synaptic_plasticity()  # update weights
            # self.update_neurons(n)              #update neural state
            self.update_x_()
            # print(f"_X_({n+1}) {self.data.auxilary_states.x_[-1]}")
            self.update_i_ext()
            # print(f"I_ext({n+1}): {self.i_ext}")
            self.update_u()
            # print(f"U_({n+1}): {self.data.states.u[-1]}")
            self.update_x()
            # print(f"X({n+1}): {self.data.states.x[-1]}")
            # self.update_XX(n)
            # print(f"XX({n+1}): {self.data.auxilary_states.xx[-1]}")
            self.update_xxx()
            # print(f"XXX({n+1}): {self.data.auxilary_states.xxx}")
        print("--------------- Simulation completed ---------------------------")
        # Return the final states
        # Note: The return values are the final states after the simulation
        x, w, u = self.data.states.x, self.data.states.w, self.data.states.u
        x_, y = self.data.auxilary_states.x_, self.data.plasticity_states.y
        z, p = self.data.plasticity_states.z, self.data.plasticity_states.p
        return x, w, u, x_, y, z, p

    # def run_simulation_m(self, Stimulation):
    #    """Exécute la simulation principale"""
    #    T_simulation = Stimulation.shape[0]
    #    paddingX = torch.zeros(T_simulation, self.N_,
    #                           dtype=self.dtype, device=self.data.context.device)
    #    paddingW = torch.zeros(T_simulation, self.N_,
    #                           self.N_, dtype=self.dtype, device=self.data.context.device)
    #    self.data.states.x = torch.cat([paddingX, self.data.states.x], dim=0)
    #    self.data.auxilary_states.xx = torch.cat([paddingX, self.data.auxilary_states.xx], dim=0)
    #    self.data.states.u = torch.cat([paddingX, self.data.states.u], dim=0)
    #    self.W = torch.cat([paddingW, self.W], dim=0)
    #    self.nij = torch.cat([paddingW, self.nij], dim=0)
    #    self.data.states.s = torch.cat([Stimulation, self.data.states.s], dim=0)
    #    self.T_ += T_simulation
    #    for n in range(T_simulation):
    #        self.update_myelinization()  # update nij
    #        self.update_synaptic_plasticity()  # update weights
    #        # self.update_neurons(n)              #update neural state
    #        self.update__X_()
    #        self.update_I_ext()
    #        self.update_U()
    #        self.update_X()
    #        # self.update_XX(n)
    #        self.update_XXX()
    #    return x, w, u, x_, y, z, p

    def update_x_(self):
        """Update the dendritic trace _X_"""
        eta_x = self.data.parameters["eta_X"]
        self.data.auxilary_states.x_ = torch.roll(
            self.data.auxilary_states.x_, -1, 0)
        self.data.auxilary_states.x_[-1] = torch.mul(
            self.data.auxilary_states.xxx / eta_x + self.data.auxilary_states.x_[-2],
            math.exp(-1 / eta_x)
            )

    def update_i_ext(self):
        """Update the external current I_ext"""
        epsilon, c_m = self.data.parameters["epsilon"], self.data.parameters["C_m"]
        uf, u0 = self.data.parameters["uf"], self.data.parameters["u0"]

        self.data.states.s = torch.roll(self.data.states.s, -1, 0)
        self.i_ext = c_m * \
            (self.data.states.s[-1] + torch.einsum("ij,ij->j",
             self.data.auxilary_states.x_[-1], self.data.states.w[-1])) * (uf-u0) / epsilon
        self.i_ext[self.data.states.s[-1] == -torch.inf] = 0.0

    def update_u(self):
        """Update the membrane potential U"""
        for _ in range(20):
            self.update_vmhn()
        self.data.states.u = torch.roll(self.data.states.u, -1, 0)
        self.data.states.u[-1] = self.data.hhstates.v

        #epsilon, c_m = self.data.parameters["epsilon"], self.data.parameters["C_m"]
        #uf, u0 = self.data.parameters["uf"], self.data.parameters["u0"]
        #us, uh = self.data.parameters["uS"], self.data.parameters["uh"]

        #v = self.data.states.u[-1]
        # update gating variables

        # update U: Mise à jour du potentiel membranaire
        #dv = (self.i_ext - self.I_Na(v) - self.I_K(v) -
        #      self.I_L(v) - self.I_Ca(v)) / c_m * epsilon
        ## dV = (self.I_ext - (156.3 * V - 3211.68)) / C_m_
        #self.data.states.u = torch.roll(self.data.states.u, -1, 0)
        #self.data.states.u[-1] = v + dv * epsilon

        #self.data.states.u[-1][self.data.states.u[-1] >= uf] = us
        ## Clamp U_ to [u0, uf]
        #self.data.states.u[-1] = torch.clamp(
        #    self.data.states.u[-1], min=u0, max=us)
        #self.data.states.u[-1][self.data.states.x[-1] == 1] = uh

    def update_vmhn(self):
        """
        Micro update U, according to HH modele: micro time step dt < epsilon
        (epsilon = 1ms, dt = 0.05)
        """
        v = self.data.hhstates.v
        m, h, n = self.data.hhstates.m, self.data.hhstates.h, self.data.hhstates.n
        dt = 0.05 # 1/20

        i_na = self.data.parameters["g_Na"] * m**3 * h * (v - self.data.parameters["E_Na"])
        i_k  = self.data.parameters["g_K"]  * n**4     * (v - self.data.parameters["E_K"])
        i_l  = self.data.parameters["g_L"]             * (v - self.data.parameters["E_L"])
        i_ext = self.i_ext

        dv = (i_ext - i_na - i_k - i_l) / self.data.parameters["C_m"]
        dm = (0.1*(v+40)/(1 - torch.exp(-(v+40)/10)))*(1 - m) - 4*torch.exp(-(v+65)/18)*m
        dh = 0.07*torch.exp(-(v+65)/20)*(1 - h) - h / (1 + torch.exp(-(v+35)/10))
        dn = 0.01*(v+55)/(1 - torch.exp(-(v+55)/10))*(1 - n) - 0.125*torch.exp(-(v+65)/80)*n

        self.data.hhstates.v = v + dt * dv
        self.data.hhstates.m = m + dt * dm
        self.data.hhstates.h = h + dt * dh
        self.data.hhstates.n = n + dt * dn

    def update_x(self):
        """Update the neural activity X"""
        self.data.states.x = torch.roll(self.data.states.x, -1, 0)
        self.data.states.x[-1] = (self.data.states.u[-1]
                                  >= self.data.parameters["uf"]).float()

    # def update_XX(self, n):
    #    """Met à jour la matrice XX"""
    #    self.data.auxilary_states.xx = torch.roll(self.data.auxilary_states.xx, -1, 0)
    #    self.data.auxilary_states.xx[-1] = self.data.states.x[-1]
    #    kR_ = self.data.parameters["k_R"]
    #    k_max = min(kR_ + 1, n + 1)
    #    indices_ = torch.arange(1, k_max, device=self.data.context.device)
    #    if indices_.numel() > 0:
    #        factors = 1 - self.data.auxilary_states.xx[-1 - indices_]
    #        self.data.auxilary_states.xx[-1] = self.data.states.x[-1] \
    #                                   * torch.cumprod(factors, dim=0)[-1]

    def x_relative_view(self, n):
        """Compute a relative view of synaptic connections"""
        eta_a = self.data.parameters["eta_A"]
        nb = self.data.context.neuron_count
        device = self.data.context.device

        indices = (n - self.data.states.nij[n]).int()
        # XX_n_shifted = self.data.auxilary_states.xx[indices.view(-1)]
        xx_n_shifted = self.data.states.x[indices.view(-1)]
        indicesbis = torch.arange(nb**2, device=device).view(nb, nb).int()
        xxx = xx_n_shifted[indicesbis, torch.arange(nb, device=device).int()]
        return xxx.T * torch.exp(-self.data.states.nij[n]/eta_a)

    def update_xxx(self):
        """Update the relative view of synaptic connections XXX"""
        # XXX_ = self.data.states.x_relative_view(-1)
        # XXX_.diagonal().copy_(self.data.states.x[-1])
        self.data.auxilary_states.xxx = self.x_relative_view(-1)  # XXX_

    def update_neurons(self, n):
        """Update the state of neurons"""
        eta_x = self.data.parameters["eta_X"]
        uf, u0 = self.data.parameters["uf"], self.data.parameters["u0"]
        kr = self.data.parameters["k_R"]
        us, uh = self.data.parameters["uS"], self.data.parameters["uh"]
        device = self.data.context.device

        # update _X_:
        # dendritic trace: doas not include current spikes
        self.data.auxilary_states.x_[-1] = torch.mul(
            self.data.auxilary_states.xxx/eta_x +
            self.data.auxilary_states.x_[-1],
            math.exp(-1/eta_x)
        )

        # init update XX':
        self.data.auxilary_states.xx = torch.roll(
            self.data.auxilary_states.xx,  -1, 0)
        # update XXX' (incomplete but sufficient - hence not saved):
        # Updates all XXX but XXX[j][j] which is not relevant
        xxx = self.x_relative_view(-1)
        # update U_:
        self.data.states.u = torch.roll(self.data.states.u, -1, 0)
        self.data.states.s = torch.roll(self.data.states.s, -1, 0)
        self.data.states.u[-1] = (self.data.states.s[-1] + torch.einsum(
            "ij,ij->j",
            self.data.auxilary_states.x_[-1], self.data.states.w[-1]
        )) * (uf-u0) + u0
        self.data.states.u[-1][self.data.states.u[-1] >= uf] = us
        self.data.states.u[-1][self.data.auxilary_states.xx[-2] == 1] = uh
        self.data.states.u[-1] = torch.maximum(
            self.data.states.u[-1], torch.tensor(uh))
        # update X:
        self.data.states.x = torch.roll(self.data.states.x,  -1, 0)
        self.data.states.x[-1] = (self.data.states.u[-1] >= uf).float()
        # update XX:
        self.data.auxilary_states.xx[-1] = self.data.states.x[-1]
        # <--------------------------------------- ########
        k_max = min(kr + 1, n + 1)
        indices_ = torch.arange(1, k_max, device=device)
        if indices_.numel() > 0:
            factors = 1 - self.data.auxilary_states.xx[-1 - indices_]
            self.data.auxilary_states.xx[-1] = self.data.states.x[-1] * \
                torch.cumprod(factors, dim=0)[-1]

        # update XXX:
        xxx.diagonal().copy_(self.data.states.x[-1])
        self.data.auxilary_states.xxx = xxx

    def update_synaptic_plasticity(self):
        """Update synaptic plasticity states"""
        u0 = self.data.parameters["u0"]
        # eta_d = self.data.parameters["eta_D"]
        # exp_factor_p = math.exp(-1 / self.data.parameters["eta_P"])
        # exp_factor_d = math.exp(-1 / self.data.parameters["eta_D"])
        # w0 = self.data.parameters["w0"]

        # update UP-u0 & UD-u0 (at time n)
        self.data.plasticity_states.up = u0 + torch.mul(
            (self.data.plasticity_states.up-u0) +
            (self.data.states.u[-1]-u0)/self.data.parameters["eta_P"],
            math.exp(-1 / self.data.parameters["eta_P"])
        )
        self.data.plasticity_states.ud = u0 + torch.mul(
            (self.data.plasticity_states.ud-u0) +
            (self.data.states.u[-1]-u0)/self.data.parameters["eta_D"],
            math.exp(-1 / self.data.parameters["eta_D"])
        )

        # update Z, P, Y and W
        self.update_z_p_y_w()

    def compute_p_y0(self):
        """Compute the probabilities p_y0"""
        ul, uh = self.data.parameters["uL"], self.data.parameters["uH"]
        #al, ah = self.data.parameters["AL"], self.data.parameters["AH"]
        epsilon = self.data.parameters["epsilon"]
        # compute proba densities rho_l & rho_h (at time n)
        rho_l = self.data.parameters["AL"] * torch.mul(
            self.data.auxilary_states.xxx,
            torch.relu(self.data.plasticity_states.ud - ul)
            )
        rho_h = self.data.parameters["AH"] * torch.mul(
            torch.mul(
                self.data.auxilary_states.x_[-1],
                torch.relu(self.data.plasticity_states.up - ul)
                ),
            torch.relu(self.data.states.u[-1] - uh))

        # compute probabilities
        proba_l = 1 - torch.exp(-rho_l * epsilon)
        proba_h = 1 - torch.exp(-rho_h * epsilon)
        proba_lh = proba_l + proba_h
        # when proba_lh < 1 but I don't want it to be zero to avoid nan in following calculations
        proba_lh_safe = torch.clamp(proba_lh, min=0.1)
        # when proba_lh >= 1
        proba_lh_geq_1 = torch.heaviside(
            proba_lh.cpu() - 1,
            torch.tensor(0.)
            ).to(self.data.context.device)
        r_l = proba_lh_geq_1 * proba_l / proba_lh_safe + (1-proba_lh_geq_1) * proba_l
        r_h = proba_lh_geq_1 * proba_h / proba_lh_safe + (1-proba_lh_geq_1) * proba_h
        r_0 = (1-proba_lh_geq_1) * (1-proba_lh)
        p_y0 = torch.stack([r_0, r_l, r_h], dim=-1)
        p_y0 = torch.clamp(p_y0, min=0., max=1.)

        return p_y0

    def update_z_p_y_w(self):
        """Update Z, P, Y and W"""
        alpha, beta = self.data.parameters["alpha"], self.data.parameters["beta"]
        eta_z = self.data.parameters["eta_Z"]
        tau_p = self.data.parameters["tau_P"]
        gamma = self.data.parameters["gamma"]
        p_y1, p_y2 = self.data.parameters["p_y1"], self.data.parameters["p_y2"]
        epsilon = self.data.parameters["epsilon"]
        #kp = self.data.parameters["kP"]
        #np = self.data.parameters["NP"]
        y = self.data.plasticity_states.y
        z, p = self.data.plasticity_states.z, self.data.plasticity_states.p

        # update Z & P
        z = z + (z * (z - alpha) * (1 - z) + \
                 gamma * p * (-y * (2. - y) + \
                              y * (y - 1) / 2)) * epsilon / eta_z
        self.data.plasticity_states.z = torch.clamp(z, min=0., max=1.)
        self.data.plasticity_states.p = p * (1 - epsilon / tau_p) + \
            (1-p) * self.data.parameters["kP"] * \
                epsilon*((y >= 1).float().sum(0) >= self.data.parameters["NP"]).float()

        # finishing update Y
        y = y.unsqueeze(-1)
        p_y0 = self.compute_p_y0()
        probabilities = torch.where(y == 0, p_y0,
                                    torch.where(y == 1, p_y1,
                                                p_y2))
        y = torch.multinomial(
            probabilities.view(-1, 3),
            num_samples=1
            ).view(self.data.context.neuron_count, self.data.context.neuron_count)
        self.data.plasticity_states.y = y

        # update W
        self.data.states.w = torch.roll(self.data.states.w, -1, 0)
        adjustments = y * ((y - 1) / 2 -
                               alpha * (2 - y)) + beta * z
        # adjustments >=0 meaning it has only a strengthening effect (even for negative weights)
        adjustments = torch.clamp(adjustments, min=0)
        self.data.states.w[-1] = torch.clamp(self.w0 * (1 + adjustments), min=-1, max=1)

    # def update_myelinization(self):
    #    alphaM_, betaM_ = self.data.parameters["alpha_M"], self.data.parameters["beta_M"]
    #    gammaM_ = self.data.parameters["gamma_M"]
    #    c0_, epsilon = self.data.parameters["c_0"], self.data.parameters["epsilon"]
    #    D_ = self.data.parameters["D"]
    #    # update M
    #    dM = betaM_ * 0.5*(z + self.data.auxilary_states.x_[-1]).sum(0) - gammaM_ * self.M
    #    self.M = torch.clamp(self.M + dM * epsilon, min=0)
    #    # update nij
    #    c = c0_ + alphaM_ * self.M
    #    Theta_M = D_ / c
    #    self.nij = torch.roll(self.nij, -1, 0)
    #    self.nij[-1] = torch.ceil(Theta_M / epsilon).int()

    def stimulation(self, i, j, k):
        """Entraînement du modèle"""
        neurons_per_cluster_ = self.data.parameters["neurons_per_cluster"]
        num_clusters_ = self.data.parameters["num_clusters"]
        device = self.data.context.device
        nij = self.data.states.nij[-1].int()
        nb = self.data.context.neuron_count
        m = 1+(max(i, j) // neurons_per_cluster_)
        e0 = torch.ones((neurons_per_cluster_ * m),
                        device=device) * -torch.inf
        e0 = torch.concat((e0, torch.zeros(
            neurons_per_cluster_*(num_clusters_ - m), device=device)), dim=0)
        e0[i] = 1
        e0[j] = 1
        e1 = torch.ones((neurons_per_cluster_), device=device) * -torch.inf
        e1 = torch.concat(
            (torch.zeros(neurons_per_cluster_*(num_clusters_-1), device=device), e1), dim=0)
        e1[k] = 1
        # mini = min(nij[i][k].item(), nij[j][k].item())
        maxi = max(nij[i][k].item(), nij[j][k].item())
        st = torch.cat([
            e0.unsqueeze(0),
            torch.zeros(nb, device=device).unsqueeze(0).repeat(maxi-1, 1),
            e1.unsqueeze(0)
        ], dim=0)
        # if mini != maxi:
        #    st = torch.cat(
        #       [st, torch.zeros(nb, device=device).unsqueeze(0).repeat(maxi-mini-1, 1),
        #       e1.unsqueeze(0)],
        #       dim=0
        #       )
        return st

    def train(self, i, j, k, epochs=300, repetitions=10, pause=300):
        """Entraînement du modèle"""
        s_pause = torch.zeros(
            pause,
            self.data.context.neuron_count,
            device=self.data.context.device
            )
        for _ in range(epochs):
            for _ in range(repetitions):
                # Stimulation externe (one-hot)
                st = self.stimulation(i, j, k)
                results = self.run_simulation(st)
                results = self.run_simulation(torch.zeros(
                    (
                        max(
                            self.data.states.nij[-1][i][k].int().item(),
                            self.data.states.nij[-1][j][k].int().item()
                        ),
                        self.data.context.neuron_count
                    ),
                    device=self.data.context.device
                ))
            results = self.run_simulation(s_pause)
        return results

    # def train_m(self, i, j, k, epochs=300, repetitions=10, pause=300):
    #    """Entraînement du modèle"""
    #    S_pause = torch.zeros(pause, self.N_, device=self.data.context.device)
    #    for n in range(epochs):
    #        for r in range(repetitions):
    #            # Stimulation externe (one-hot)
    #            St = self.stimulation(i, j, k)
    #            results = self.run_simulation_m(St)
    #            results = self.run_simulation_m(torch.zeros(
    #                (
    #                    max(
    #                        self.nij[-1][i][k].int().item(),
    #                        self.nij[-1][i][k].int().item()
    #                    ),
    #                    self.N_
    #                ),
    #                device=self.data.context.device
    #            ))
    #        results = self.run_simulation_m(S_pause)
    #    return results
