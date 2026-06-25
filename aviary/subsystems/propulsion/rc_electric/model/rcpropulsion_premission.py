import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output


class RCPropPreMission(om.Group):
    """Calculate RC electric propulsion premission motor and battery properties."""

    def initialize(self):
        
        add_aviary_option(self, Aircraft.Engine.Motor.KV_EQ_SLOPE)    # m = KV_EQ_SLOPE
        add_aviary_option(self, Aircraft.Engine.Motor.KV_EQ_INT)      # b = KV_EQ_INT
      
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.name = 'rcpropulsion_premission'

    def setup(self):
        #battery mass
        # battery voltage
        # idle current
        # max continuous current
        # motor mass
        
        #TODO: CITE!
        self.add_subsystem(
            'energy_calc',
            om.ExecComp(
                #TODO: may add options here
                'energy = voltage_in * (battery_mass * 7.3 - 0.246)',
                energy={'val': 0.0, 'units': 'W*h'},
                voltage_in={'val': 0.0, 'units': 'V'},
                battery_mass={'val': 0.0, 'units': 'kg'},
            ),
            promotes_inputs=[('battery_mass', Aircraft.Battery.MASS), ('voltage_in', Aircraft.Battery.VOLTAGE)],
            promotes_outputs=[('energy', Aircraft.Battery.ENERGY_CAPACITY)],
        )

        self.add_subsystem(
            'motor_resistance_calc',
            om.ExecComp(
                'resistance = 0.0467 * idle_current ** -1.892', 
                idle_current={'val': 0.0, 'units': 'A'},
                resistance={'val': 0.0, 'units': 'ohm'}
            ),
            promotes_inputs=[('idle_current', Aircraft.Engine.Motor.IDLE_CURRENT)],
            promotes_outputs=[('resistance', Aircraft.Engine.Motor.RESISTANCE)]
        )
        
        #TODO: Cite
        self.add_subsystem(
            'motor_kv_calc',
            om.ExecComp(
                'kv = m * max_current / (motor_mass * 1000.0) + b', # The KV empirical fit appears to use motor mass in grams. Aviary provides motor_mass here in kg, so convert kg -> g. or else the number becomes unrealistically high.

                kv={'val': 0.0, 'units': 'rpm/V'},
                max_current={'val': 0.0, 'units': 'A'},
                motor_mass={'val': 0.0, 'units': 'kg'},
                m=self.options[Aircraft.Engine.Motor.KV_EQ_SLOPE],
                b=self.options[Aircraft.Engine.Motor.KV_EQ_INT],
            ),
            promotes_inputs=[
                ('max_current', Aircraft.Engine.Motor.MAX_CONT_CURRENT),
                ('motor_mass', Aircraft.Engine.Motor.MASS),
            ],
            promotes_outputs=[('kv', Aircraft.Engine.Motor.KV)]
        )
        # commented out for now, may add back in later
        # self.add_constraint(Aircraft.Engine.Motor.KV, upper=540, units='rpm/V')
        # self.add_subsystem(
        #     'total_mass',
        #     om.ExecComp(
        #         'engine_mass = batt_mass + motor_mass',
        #         batt_mass={'val': 0.0, 'units': 'kg'},
        #         motor_mass={'val': 0.0, 'units': 'kg'},
        #         engine_mass={'val': 0.0, 'units': 'kg'},
        #     ),
        #     promotes_inputs=[('batt_mass', Aircraft.Battery.MASS), ('motor_mass', Aircraft.Engine.Motor.MASS)],
        #     promotes_outputs=[('engine_mass', Aircraft.Engine.MASS)]
        # )