import yaml
import numpy as np
import math
from Company import company
from gym import spaces


class market_and_env:
    def __init__(self, path_config):
        self.path_config = path_config
        self.config = yaml.load(open(self.path_config, "r"), Loader=yaml.FullLoader)[
            "market"
        ]
        self.cap_schedule = self.config["cap_schedule"]
        self.max_step = self.config["max_step"]
        self.lowest_cap = self.config["lowest_cap"]
        self.highest_cap = self.config["highest_cap"]
        self.fine_multiplier = self.config["fine_multiplier"]
        self.damage_linear = self.config["damage_linear"]
        self.damage_quadratic = self.config["damage_quadratic"]
        self.radiative_forcing_rate = self.config["radiative_forcing_rate"]
        self.climate_sensitivity = self.config["climate_sensitivity"]
        self.temp_0 = self.config["temp_0"]
        self.eps = self.config["eps"]
        self.companies = [
            company(path_config=self.path_config)
            for _ in range(self.config["nb_companies"])
        ]
        self.cumulative_co2_emissions = self.config["cumulative_co2_emissions"]
        self.s = 0
        self.scc = 0
        self.delta_temp = 0

        self.observation_space = spaces.Dict(
            {
                "cumulative_co2_emissions": spaces.Box(
                    low=0, high=np.inf, shape=(1,)
                ),  # float
                "scc": spaces.Box(low=0, high=np.inf, shape=(1,)),  # float
                "delta_temp": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # float
                "cash_stock": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # float
                "carbon_stock": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                ),  # float
                "prod_stock": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # float
                "wage": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # float
                "bankrupt": spaces.Discrete(2),  # int {0, 1}
                "labor": spaces.Box(low=0, high=np.inf, shape=(1,)),  # float
                "step": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int),  # int
            }
        )
        self.action_space = spaces.Dict(
            {
                "price": spaces.Box(low=0, high=np.inf, shape=(1,)),  # float
                "quantity": spaces.Box(low=0, high=np.inf, shape=(1,)),  # float
                "delta_carbon_credit": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,)
                ),  # float
                "delta_wage": spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),  # float
                "invest_tech_decarbon": spaces.Box(
                    low=0, high=np.inf, shape=(1,)
                ),  # float
                "invest_tech_prod": spaces.Box(low=0, high=np.inf, shape=(1,)),  # float
                "invest_capital": spaces.Box(low=0, high=np.inf, shape=(1,)),  # float
            }
        )

    def reset(self):
        self.cumulative_co2_emissions = self.config["cumulative_co2_emissions"]
        self.s = 0
        self.scc = 0
        self.delta_temp = 0
        for c in self.companies:
            c.reset(self)

    def cap(self, t, sector=None):
        return (
            self.lowest_cap
            + (self.highest_cap - self.lowest_cap)
            * (1 + np.cos(math.pi * (t) / self.max_step))
            / 2
        )

    def damage_factor(self, delta_temp):
        return 1 / (
            1
            + self.damage_linear * delta_temp
            + self.damage_quadratic * delta_temp**2
        )

    def fine(self, t, sector=None):
        return self.scc * self.fine_multiplier

    def delta_radiative_forcing(self, old_co2, new_co2):
        return self.radiative_forcing_rate * np.log(new_co2 / old_co2)

    def delta_temperature(self, old_co2, new_co2):
        delta_RF = self.delta_radiative_forcing(old_co2, new_co2)
        return self.climate_sensitivity * delta_RF

    def calc_SCC(self):
        all_co2_emissions = sum(
            [c.carbon_emission for c in self.companies if not c.done]
        )
        all_net_profit = sum([c.net_profit for c in self.companies if not c.done])
        new_co2 = self.cumulative_co2_emissions + all_co2_emissions
        self.delta_temp = self.delta_temperature(self.cumulative_co2_emissions, new_co2)
        damage_factor = self.damage_factor(self.delta_temp)
        self.scc = (
            (1 - damage_factor) * all_net_profit / (float(self.eps) + all_co2_emissions)
        )
        self.cumulative_co2_emissions = new_co2
        self.s += 1
        return self.scc

    def step(self, companies_actions):
        companies_market_activity = [None] * len(self.companies)
        companies_emission_count = [None] * len(self.companies)
        companies_investment = [None] * len(self.companies)
        for i in range(len(self.companies)):
            if self.companies[i].done:
                continue
            companies_market_activity[i] = self.companies[i].step_market_activity(
                companies_actions[i], self
            )

            companies_emission_count[i] = self.companies[i].step_emission_count(
                companies_market_activity[i][2], self
            )

        self.calc_SCC()
        for i in range(len(self.companies)):
            if self.companies[i].done:
                continue
            companies_investment[i] = self.companies[i].step_investment(
                companies_actions[i],
                self,
                *companies_market_activity[i],
                companies_emission_count[i]
            )

        self.next_state = {
            "market": [self.cumulative_co2_emissions, self.scc, self.delta_temp],
            "companies": [c.next_states for c in self.companies],
        }
        return (
            self.next_state,
            [c.cash_stock for c in self.companies],
            [c.done for c in self.companies],
            [c.info for c in self.companies],
        )

    def get_obs_company(self, i):
        return {
            "cumulative_co2_emissions": self.cumulative_co2_emissions,  # (0, +inf), float
            "scc": self.scc,  # (0, +inf), float
            "delta_temp": self.delta_temp,  # (-inf, +inf), float
            "cash_stock": self.companies[i].cash_stock,  # (-inf, +inf), float
            "carbon_stock": self.companies[i].carbon_stock,  # (-inf, +inf), float
            "prod_stock": self.companies[i].prod_stock,  # (-inf, +inf), float
            "wage": self.companies[i].wage,  # (-inf, +inf), float
            "bankrupt": self.companies[i].bankrupt,  # {0, 1}, int
            "labor": self.companies[i].labor,  # (0, +inf), float
            "step": self.companies[i].s,  # (0, +inf), int
        }

    def render(self):
        print("step: ", self.s)
        print("cumulative_co2_emissions: ", self.cumulative_co2_emissions)
        print("SCC: ", self.scc)
        print("delta_temp: ", self.delta_temp)
