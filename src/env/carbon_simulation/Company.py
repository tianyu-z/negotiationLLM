import yaml
import numpy as np


class company:
    def __init__(self, path_config):
        self.path_config = path_config
        self.config = yaml.load(open(self.path_config, "r"), Loader=yaml.FullLoader)[
            "company"
        ]
        self.sector_name = self.config["sector_name"]
        self.id = f"{self.sector_name}_{self.config['id']}"
        self.tech_decarbon = self.config["tech_decarbon"]
        self.tech_prod = self.config["tech_prod"]
        self.capital = self.config["capital"]
        self.labor = self.config["labor"]
        self.wage = self.config["wage"]  # wage -> labor
        self.demand_elasticity = self.config["demand_elasticity"]
        self.supply_elasticity = self.config["supply_elasticity"]
        self.demand_scale = self.config["demand_scale"]
        self.supply_scale = self.config["supply_scale"]
        self.unit_capital_alpha = self.config["unit_capital_alpha"]
        self.unit_labor_alpha = self.config["unit_labor_alpha"]
        self.carbon_emission_scale = self.config["carbon_emission_scale"]
        self.carbon_emission_elasticity = self.config["carbon_emission_elasticity"]
        self.impact_tech_decarbon_scale = self.config["impact_tech_decarbon_scale"]
        self.impact_tech_decarbon_elasticity = self.config[
            "impact_tech_decarbon_elasticity"
        ]
        self.impact_tech_prod_scale = self.config["impact_tech_prod_scale"]
        self.impact_tech_prod_elasticity = self.config["impact_tech_prod_elasticity"]
        self.impact_wage_labor_scale = self.config["impact_wage_labor_scale"]
        self.impact_wage_labor_elasticity = self.config["impact_wage_labor_elasticity"]
        self.impact_capital_scale = self.config["impact_capital_scale"]
        self.impact_capital_elasticity = self.config["impact_capital_elasticity"]
        self.bankrupt_threshold = self.config["bankrupt_threshold"]
        self.discount_factor = self.config["discount_factor"]
        self.discount_type = self.config["discount_type"]

        self.cash_stock = 0
        self.prod_stock = 0
        self.carbon_credit = 0
        self.s = 0
        self.bankrupt = False
        self.done = False

    def reset(self, market):
        self.cash_stock = 0
        self.prod_stock = 0
        self.carbon_credit = market.cap(t=0)
        self.s = 0
        self.bankrupt = False
        self.done = False

    def step_market_activity(self, actions, market):
        # stuff need to change before you pay
        price_set_by_company = actions["price"]
        quantity_produce = actions["quantity"]
        self.carbon_credit += actions["delta_carbon_credit"]
        profit = self.calc_profit(quantity_produce, price_set_by_company)
        profit += self.cash_stock
        self.net_profit = (
            self.calc_profit(quantity_produce, price_set_by_company)
            - self.wage * self.labor
        )

        return profit, price_set_by_company, quantity_produce

    def step_emission_count(self, quantity_produce, market):
        self.carbon_emission = self.get_carbon_emission(
            quantity_produce, self.tech_decarbon
        )
        # if you don't have enough carbon credit, you need to pay fine
        if self.carbon_credit < self.get_carbon_emission(
            quantity_produce, self.tech_decarbon
        ):
            fine_emission = self.carbon_emission - self.carbon_credit
            self.carbon_credit = market.cap(self.s)
        else:
            self.carbon_credit = (
                self.carbon_credit - self.carbon_emission + market.cap(self.s)
            )
            fine_emission = 0
        return fine_emission

    def step_investment(
        self,
        actions,
        market,
        profit,
        price_set_by_company,
        quantity_produce,
        fine_emission,
    ):
        self.cash_stock = (
            profit
            - self.carbon_credit * market.scc
            - fine_emission * market.fine(self.s)
            - self.wage * self.labor
        )

        # stuff need to change after you pay
        self.wage += actions["delta_wage"]
        self.cash_stock = self.invest_tech_decarbon(
            self.cash_stock, actions["invest_tech_decarbon"]
        )
        self.cash_stock = self.invest_tech_prod(
            self.cash_stock, actions["invest_tech_prod"]
        )
        self.cash_stock = self.invest_capital(
            self.cash_stock, actions["invest_capital"]
        )
        self.wage_influence_labor(actions["delta_wage"])
        if self.cash_stock <= float(self.bankrupt_threshold):
            self.bankrupt = True
        self.s += 1
        self.next_states = {
            "cash_stock": self.cash_stock,
            "carbon_credit": self.carbon_credit,
            "prod_stock": self.prod_stock,
            "wage": self.wage,
            "bankrupt": self.bankrupt,
            "labor": self.labor,
            "step": self.s,
        }
        reward = self.cash_stock
        self.done = self.bankrupt or self.s >= market.max_step or self.labor <= 1
        self.info = {}
        return self.next_states, reward, self.done, self.info

    def step(self, actions, market):
        profit, price_set_by_company, quantity_produce = self.step_market_activity(
            actions, market
        )
        emission_cost = self.step_emission_count(quantity_produce, market)
        return self.step_investment(
            actions,
            market,
            quantity_produce,
            price_set_by_company,
            profit,
            emission_cost,
        )

    def calc_quantity(self, price, issupply):
        assert price >= 0, "price cannot be negative"
        # assume the relationship between price and quantity is a power function:
        # log-log linear function. Q = scale * P^elasticity
        if issupply:
            quantity = self.supply_scale * (price) ** (self.supply_elasticity)
        else:
            quantity = self.demand_scale * (price) ** (self.demand_elasticity)
        return quantity

    def calc_price(self, quantity, issupply):
        assert quantity >= 0, "quantity cannot be negative"
        # assume the relationship between price and quantity is a power function:
        # log-log linear function. P = scale(-1/elasticity) * Q^(1/elasticity)
        if issupply:
            price = (self.supply_scale) ** (-1 / self.supply_elasticity) * (
                quantity
            ) ** (1 / self.supply_elasticity)
        else:
            price = (self.demand_scale) ** (-1 / self.demand_elasticity) * (
                quantity
            ) ** (1 / self.demand_elasticity)
        return price

    def calc_revenue(self, quantity_produce, price_set_by_company):
        assert quantity_produce >= 0, "quantity cannot be negative"
        assert price_set_by_company >= 0, "price cannot be negative"
        mkt_demand = self.calc_quantity(
            price_set_by_company, issupply=False
        )  # based on the price set by the company,
        # what is the demand of the market looks like?

        if (
            self.prod_stock + quantity_produce < mkt_demand
        ):  # the company can supply the market is
            # less than the market demand
            self.prod_stock = 0
            revenue = (self.prod_stock + quantity_produce) * price_set_by_company
        else:  # the company can supply the market is more
            # than the market demand
            self.prod_stock = self.prod_stock + quantity_produce - mkt_demand
            revenue = mkt_demand * price_set_by_company

        return revenue

    def get_unit_cost(self, tech_prod):
        assert self.unit_capital_alpha + self.unit_labor_alpha == 1, "alpha != 1"
        assert tech_prod >= 0, "tech_prod cannot be negative"
        unit_cost = (
            (tech_prod) ** (-1)
            * self.capital ** (-self.unit_capital_alpha)
            * self.labor ** (-self.unit_labor_alpha)
        )
        return unit_cost

    def get_carbon_emission(self, quantity_produce, tech_decarbon):
        assert quantity_produce >= 0, "quantity cannot be negative"
        assert tech_decarbon >= 0, "tech_decarbon cannot be negative"
        assert self.carbon_emission_elasticity < 0, "elasticity can only be negative"
        assert self.carbon_emission_scale > 0, "scale can only be positive"
        carbon_emission = (
            self.carbon_emission_scale
            * quantity_produce
            * tech_decarbon**self.carbon_emission_elasticity
        )
        return carbon_emission

    def calc_prod_cost(self, quantity_produce):
        assert quantity_produce >= 0, "quantity cannot be negative"
        cost = (
            self.labor * self.wage
            + self.get_unit_cost(self.tech_prod) * quantity_produce
        )
        return cost

    def calc_profit(self, quantity_produce, price_set_by_company):
        assert quantity_produce >= 0, "quantity cannot be negative"
        assert price_set_by_company >= 0, "price cannot be negative"
        revenue = self.calc_revenue(quantity_produce, price_set_by_company)
        prod_cost = self.calc_prod_cost(quantity_produce)
        profit = revenue - prod_cost
        return profit

    def invest_tech_decarbon(self, money_in, invest_tech_decarbon):
        assert invest_tech_decarbon >= 0, "investment cannot be negative"
        assert (
            self.impact_tech_decarbon_elasticity > 0
        ), "elasticity can only be positive"
        assert self.impact_tech_decarbon_scale > 0, "scale can only be positive"
        self.tech_decarbon += (
            self.impact_tech_decarbon_scale
            * invest_tech_decarbon**self.impact_tech_decarbon_elasticity
        )
        money_out = money_in - invest_tech_decarbon
        return money_out

    def invest_tech_prod(self, money_in, invest_tech_prod):
        assert invest_tech_prod >= 0, "investment cannot be negative"
        assert self.impact_tech_prod_elasticity > 0, "elasticity can only be positive"
        assert self.impact_tech_prod_scale > 0, "scale can only be positive"
        self.tech_prod += (
            self.impact_tech_prod_scale
            * invest_tech_prod**self.impact_tech_prod_elasticity
        )
        money_out = money_in - invest_tech_prod
        return money_out

    def invest_capital(self, money_in, invest_capital):
        assert invest_capital >= 0, "investment cannot be negative"
        self.capital = (
            self.discount_onetime(self.capital)
            + self.impact_capital_scale
            * invest_capital**self.impact_capital_elasticity
        )
        money_out = money_in - invest_capital
        return money_out

    def discount_onetime(self, x, p=None):
        # assert self.discount_type in [
        #     "cos",
        #     "exp",
        # ], f"discount_type {self.discount_type} not supported"
        # if self.discount_type == "cos":
        #     return x * (1 - self.discount_factor) ** p
        # elif self.discount_type == "exp":
        #     return x * np.exp(-self.discount_factor)
        # else:
        #     raise NotImplementedError
        return x * np.exp(-self.discount_factor)

    def wage_influence_labor(self, delta_wage):
        self.labor *= self.impact_wage_labor_scale * np.exp(
            delta_wage * self.impact_wage_labor_elasticity
        )
        if isinstance(self.net_profit, complex):
            print("debug")
        return
