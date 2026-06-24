import unittest

import numpy as np
import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from packaging import version

from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
from aviary.subsystems.propulsion.propulsion_mission import PropulsionMission, PropulsionSum
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import get_path
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.validation_cases.validation_tests import get_flops_inputs
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


    # 'clean' test using GASP-derived engine deck
nn = 20

prob = om.Problem()
options = AviaryValues()
options.set_val(Settings.VERBOSITY, 0)

options = options
options.set_val(Aircraft.Engine.NUM_ENGINES, 2)
options.set_val(Aircraft.Engine.NUM_WING_ENGINES, 2)


engine = RCBuilder()
preprocess_propulsion(options, engine_models=[engine])

prob.model = PropulsionMission(
    num_nodes=nn, aviary_options=options, engine_models=[engine]
)

setup_model_options(prob, options)
prob.setup(force_alloc_complex=True)

prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.linspace(0, 1, nn))
prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')

# self.prob.set_val(
#     Aircraft.Engine.SCALE_FACTOR,
#     options.get_val(Aircraft.Engine.SCALE_FACTOR),
#     units='unitless',
# )
prob.run_model()

# print(f"thrust: {prob.get_val(Dynamic.Vehicle.Propulsion.THRUST, units ='N')}")
battery_power = prob.get_val('rc_electric.base.battery.power', units='W')
esc_power = prob.get_val('rc_electric.base.esc.power', units='W')
motor_power = prob.get_val('rc_electric.base.motor.power', units='W')
prop_power = prob.get_val('rc_electric.base.prop_power', units='W')
power_residual = battery_power + esc_power + motor_power - prop_power
#print("Bat_pow W", battery_power, "Esc_Pow W", esc_power,"Motor_Pow W", motor_power, "Prop_Pow W", prop_power)