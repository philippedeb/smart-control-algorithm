import copy
from typing import Dict, List, NamedTuple, Tuple
import numpy as np

"""
A concise implementation of the proposed smart control algorithm by Philippe de Bekker.
"""

__author__ = "Philippe de Bekker"
__copyright__ = "Copyright (C) 2022 Philippe de Bekker"
__version__ = "1.0"
__maintainer__ = "Philippe de Bekker"
__email__ = "contact@philippedebekker.com"
__status__ = "Prototype"

class Battery(NamedTuple):
    """
    Battery properties for simulating a natural operational environment.
    """
    
    # Complete battery capacity [kWh]
    capacity: float
    
    # Minimum battery capacity based on Depth of Discharge [kWh]
    min_capacity: float
    
    # Battery capacity at the start of the lookahead window [kWh]
    initial_capacity: float
    
    # Ratio of efficiency per simulated timestamp when charging the battery [%]
    efficiency_charging: float
    
    # Ratio of efficiency per simulated timestamp when discharging the battery [%]
    efficiency_discharging: float
    
    # Maximum amount of energy that can be charged per timestamp in the simulation [kW * h = kWh]
    max_chargable_energy: float
    
    # Maximum amount of energy that can be discharged per timestamp in the simulation [kW * h = kWh]
    max_dischargable_energy: float

class LookaheadWindow():
    
    def __init__(self, window_size: int, 
                 residual_power: List[float],
                 export_tariffs: List[float],
                 import_tarrifs: List[float],
                 battery: Battery) -> None:
        """
        Lookahead window that can be used for some timestamp in a simulation to compute an optimal decision based on forecasts of power, demand and tariffs. 

        Args:
            window_size (int): amount of timestamps
            residual_power (List[float]): (estimated) residual power at each timestamp (i.e. power minus demand) [kW * h = kWh]
            export_tariffs (List[float]): (estimated) export tariff at each timestamp [£/kWh]
            import_tarrifs (List[float]): (estimated) import tariff at each timestamp [£/kWh]
            battery (Battery): properties of simulated battery
        """
        
        self.window_size: int = window_size
        self.residual_power: np.ndarray = np.array(residual_power)
        
        self.initial_capacity = battery.initial_capacity
        self.battery = battery
        self.room_to_charge: np.ndarray = np.full((window_size,), max(0, battery.capacity - battery.initial_capacity))
        
        self.charging_boundary: float = battery.max_chargable_energy
        self.charging_boundaries: np.ndarray = np.full((window_size,), self.charging_boundary)
        
        self.discharging_boundary: float = battery.max_dischargable_energy
        self.discharging_boundaries: np.ndarray = np.full((window_size,), self.discharging_boundary)
        
        self.start_index = 0
        self.throughput_values: np.ndarray =  np.full((window_size,), max(0, battery.capacity - battery.initial_capacity))
        
        self.export_tariffs: np.ndarray = np.array(export_tariffs)
        self.import_tariffs: np.ndarray = np.array(import_tarrifs)
    
    def compute(self) -> Dict[str, float]:
        """
        Implements the proposed smart control algorithm by Philippe de Bekker.
        Step 5 is omitted as the simulated environment does not allow charging the battery using imported energy, however, a comment that explains the implementation is provided. 
        Calculates the optimal decision to take for some timestamp based on the provided lookahead window.

        Returns:
            Dict[str, float]: {
                "opt_net_value": optimal amount of energy to export/import [kWh],
                "opt_charge": optimal amount of energy to charge the battery [kWh],
                "opt_discharge": optimal amount of energy to discharge the battery [kWh]
            }
        """
        
        for t in range(0, self.window_size):
            
            # We skip excess power (past excess power can be used in the future)
            if self.residual_power[t] >= 0:
                continue
            
            #####################################################################################################
            # Use battery first (other future timestamps can later on swap with this discharged energy as well) #
            #####################################################################################################
            dischargable_excess_demand = min(-self.residual_power[t], self.discharging_boundaries[t] * self.battery.efficiency_discharging)
            dischargable_battery_soc = max(0, self.initial_capacity - self.battery.min_capacity)
            battery_energy = min(dischargable_excess_demand / self.battery.efficiency_discharging, dischargable_battery_soc)
            
            if battery_energy > 0:
                self.discharging_boundaries[t] -= battery_energy
                self.initial_capacity -= battery_energy
                self.room_to_charge[t:] = [e + battery_energy for e in self.room_to_charge[t:]]
                self.residual_power[t] += battery_energy * self.battery.efficiency_discharging
            
            
            ########################################################################################
            # Charge using past excess power (sorted by ascending selling prices) to discharge now #
            ########################################################################################
            temp_throughput_values, start_index = self.__get_rtc_throughput_values(t)
            self.throughput_values = temp_throughput_values
            self.start_index = start_index
            ascending_selling_prices = np.argsort(self.export_tariffs[:t])
            ascending_buying_prices = np.argsort(self.import_tariffs[:min(t + 1, max(0, self.window_size - 1))])
            
            
            def charge(input_amount: float, at_t: int, for_t: int) -> None:
                
                # Charge at_t
                battery_amount = input_amount * self.battery.efficiency_charging
                self.residual_power[at_t] -= input_amount
                self.charging_boundaries[at_t] -= battery_amount
                
                
                # Update throughput & room to charge from now on
                self.room_to_charge[at_t:for_t] = [e - battery_amount for e in self.room_to_charge[at_t:for_t]]
                temp_throughput_values, start_index = self.__get_rtc_throughput_values(t)
                self.start_index = start_index
                self.throughput_values = temp_throughput_values
                
                # Discharge for_t
                output_amount = battery_amount * self.battery.efficiency_discharging
                self.residual_power[for_t] += output_amount
                self.discharging_boundaries[for_t] -= battery_amount

            
            for i in ascending_selling_prices:
        
                # There is no excess demand to cover anymore or can be discharged
                if self.residual_power[t] >= 0 or self.discharging_boundaries[t] == 0:
                    break
                
                # The price of selling in the past vs. covering demand with bought energy now is more profitable
                if self.export_tariffs[i] > self.import_tariffs[t]:
                    break
                
                # There is no throughput from this index anymore, look further ahead
                if i < self.start_index:
                    continue
                
                # There is excess power at this timestamp, attempt to charge as much as is needed and possible
                if self.residual_power[i] > 0:

                    # Charge as much as needed (and as possible) from current excess power
                    curr_chargable_excess_power = min(self.residual_power[i], min(self.throughput_values[i], self.charging_boundaries[i]) / self.battery.efficiency_charging)       # How much we can charge at i
                    dischargable_excess_demand = min(-self.residual_power[t], self.discharging_boundaries[t] * self.battery.efficiency_discharging)                                 # How much we want at t
                    input_amount = min(dischargable_excess_demand / self.battery.efficiency_discharging, curr_chargable_excess_power)
                    charge(input_amount=input_amount, at_t=i, for_t=t)
            
            #########################################################################################
            # Then, we charge using past discharged energy (swap: at other timestamp we buy energy) #
            #########################################################################################
            for i in ascending_buying_prices:
                
                # There is no excess demand to cover anymore or can be discharged
                if self.residual_power[t] >= 0 or self.discharging_boundaries[t] == 0:
                    break
                
                # We are better off buying energy at our current timestamp, cheaper!
                if self.import_tariffs[t] <= self.import_tariffs[i]:
                    break
                
                # There is no throughput from this index anymore, look further ahead
                if i < self.start_index:
                    continue
                
                discharged = self.discharging_boundary - self.discharging_boundaries[i]
                if discharged > 0:
                    
                    # Charge as much as needed (and as possible) from past discharged energy
                    curr_dischargable_past_energy = min(discharged, self.throughput_values[i])
                    dischargable_excess_demand = min(-self.residual_power[t], self.discharging_boundaries[t] * self.battery.efficiency_discharging)
                    
                    output_energy = min(curr_dischargable_past_energy * self.battery.efficiency_discharging, dischargable_excess_demand)
                    battery_energy = output_energy / self.battery.efficiency_discharging
                    
                    # Buy energy at that timestamp (equivalent to adding excess demand which needs to be bought later)
                    self.residual_power[i] -= output_energy
                    self.discharging_boundaries[i] += battery_energy
                    
                    # Discharge at current timestamp
                    self.residual_power[t] += output_energy
                    self.discharging_boundaries[t] -= battery_energy
                    
                    
                    # Update throughput & room to charge from past moment until now
                    self.room_to_charge[i:t] = [e - battery_energy for e in self.room_to_charge[i:t]]
                    temp_throughput_values, start_index = self.__get_rtc_throughput_values(t)
                    self.throughput_values = temp_throughput_values
                    self.start_index = start_index
                    
            ##################################################################################################################################
            # If profitable, we also look at whether buying energy in the past and charging the battery with it to discharge now is possible #
            ##################################################################################################################################
            """
            TODO (this scenario was not applicable in the simulated environment of the research):
            Sort on import energy prices until temp throughput is zero and buy as much energy as is needed while cheaper and is possible to charge at that moment
            """
            
            ####################################################
            # Otherwise, we buy energy (keep as excess demand) #
            ####################################################
    
        # Left over residual power at original timestamp (0): sell all excess power & buy all excess demand   
        return {
            "opt_net_value": self.residual_power[0],
            "opt_charge": self.charging_boundary - self.charging_boundaries[0],
            "opt_discharge": self.discharging_boundary - self.discharging_boundaries[0]
        }
    
    
    def __get_rtc_throughput_values(self, t: int) -> Tuple[List[int], int]:
        """
        Gets the maximal throughput per timestamp before timestamp t - needed for the the room_to_charge list. 
        In addition, the starting index of the first positive throughput after some bottleneck is provided. 
        RtC values of 0 are a bottleneck for throughput, these simulated timestamps cannot carry more energy that is needed at t.

        Args:
            t (int): index of current timestamp

        Returns:
            Tuple[List[int], int]: 
                - list of max throughput per timestamp
                - starting index of positive throughput
        """
        
        throughput_values = []
        max_throughput: float = float("inf")
        
        start_index = 0
        start_index_changed = False
        
        for i in range(max(0, self.window_size - 1), -1, -1):
            temp_rtc = self.room_to_charge[i]
            max_throughput = min(max_throughput, temp_rtc)
            if not start_index_changed and max_throughput == 0:
                start_index = min(i + 1, t)
                start_index_changed = True
            throughput_values.append(max_throughput)

        throughput_values.reverse()
        return throughput_values, start_index

class OutputModel(NamedTuple):
    # Amount of energy imported during a simulation [kWh]
    energy_bought: np.ndarray
    
    # Amount of energy exported during a simulation [kWh]
    energy_sold: np.ndarray
    
    # Total profit made by exporting energy during a simulation [£]
    profit: float
    
    # Total loss made by importing energy during a simulation [£]
    loss: float

class OptimizedModel():
    
    def __init__(self, battery: Battery,
                 residual_power: List[float], 
                 export_tariffs: List[float],
                 import_tariffs: List[float],
                 lookahead: int,
                 time: int) -> None:
        """
        Creates a model that can be used for simulating an environment to run the smart control algorithm proposed by Philippe de Bekker.

        Args:
            battery (Battery): properties of simulated battery in provided environment
            residual_power (List[float]): generated power subtracted by demand per timestamp
            export_tariffs (List[float]): export tariff per timestamp
            import_tariffs (List[float]): import tariff per timestamp
            lookahead (int): amount of timestamps that are provided in the lookahead window
            time (int): duration of the simulation (amount of timestamps)
        """

        self.battery = battery
        self.capacity = 0
        
        self.residual_power = residual_power
        self.export_tariffs = export_tariffs
        self.import_tariffs = import_tariffs
        
        self.lookahead = lookahead
        self.time = time

    
    def run(self, show_progress: bool = False) -> OutputModel:
        """
        Runs a smart control algorithm in a simulated environment provided by the user to ultimately assess the performance by returning various statistics.

        Args:
            show_progress (bool, optional): Show progress of algorithm in the console (useful for long runs). Defaults to False.

        Returns:
            OutputModel: NamedTuple consisting of energy_bought, energy_sold, profit, loss
        """
        
        length_res, length_exp, length_imp = len(self.residual_power), len(self.export_tariffs), len(self.import_tariffs)

        time        = min([self.time, length_res, length_exp, length_imp])
        lookahead   = min(time, self.lookahead)
        capacity    = self.battery.initial_battery_capacity
        
        profit = 0
        loss   = 0
        
        energy_bought = np.zeros(time)
        energy_sold   = np.zeros(time)
        
        if show_progress:
            print('[MODEL STARTED RUNNING]')
        
        for t in np.arange(time):
            
            if show_progress and t % 1000 == 0:
                print(f"Progress: {(float(t) / time * 100):,.1f}%")
            
            start = t
            end   = min(time, t + lookahead)
            window_size = end - start
            
            # Note: can be replaced with estimations (forecasts) or altered values by functions
            lookahead_window_residual_power = copy.deepcopy(self.residual_power[start:end])
            lookahead_window_export_tariffs = copy.deepcopy(self.export_tariffs[start:end])
            lookahead_window_import_tariffs = copy.deepcopy(self.import_tariffs[start:end])
            
            lookahead_window = LookaheadWindow(
                window_size          = window_size,
                residual_power       = lookahead_window_residual_power,
                export_tariffs       = lookahead_window_export_tariffs,
                import_tarrifs       = lookahead_window_import_tariffs,
                battery              = self.battery
            )

            opt_estimation = lookahead_window.compute()
            opt_charge     = min([opt_estimation["opt_charge"], self.battery.max_chargable_energy, self.battery.capacity - capacity])
            opt_discharge  = min([opt_estimation["opt_discharge"], self.battery.max_dischargable_energy, max(0, capacity - self.battery.min_capacity)])
            opt_res_power  = self.residual_power[t] - opt_charge / self.battery.efficiency_charging + opt_discharge * self.battery.efficiency_discharging
            
            capacity += opt_charge - opt_discharge
            if opt_res_power >= 0:
                profit += opt_res_power * lookahead_window.export_tariffs[0]
                energy_sold[t] = opt_res_power
            else:
                loss -= opt_res_power * lookahead_window.import_tariffs[0]
                energy_bought[t] = -opt_res_power
        
        return OutputModel(energy_bought, energy_sold, profit, loss)
