from Company import company
from Market import market_and_env
import numpy as np


def random_actions(n_companies):
    return [
        {
            "price": np.random.randint(5, 15),
            "quantity": np.random.randint(250, 350),
            "delta_carbon_credit": np.random.randint(100, 400),
            "delta_wage": np.random.randint(-2, 10),
            "invest_tech_decarbon": np.random.randint(0, 155000),
            "invest_tech_prod": np.random.randint(0, 1000),
            "invest_capital": np.random.randint(0, 1000),
        }
        for _ in range(n_companies)
    ]


if __name__ == "__main__":
    np.random.seed(0)
    configs = "config_yamls\\carbon_simulation\\default.yaml"
    market = market_and_env(configs)
    market.reset()
    for s in range(market.max_step):
        companies_actions = random_actions(len(market.companies))
        next_states, rewards, dones, infos = market.step(companies_actions)
        market.render()
