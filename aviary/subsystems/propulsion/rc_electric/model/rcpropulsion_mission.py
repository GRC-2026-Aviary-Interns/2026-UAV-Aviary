import numpy as np
import openmdao.api as om

from aviary.subsystems.propulsion.rc_electric.model.rc_performance import \
    Battery, ElectronicSpeedController, Motor, PropCoefficients, Propeller, PowerImplicit, Vectorization, RangeClamp
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic


class RCPropMission(om.Group):
    """Calculates the mission performance (ODE) of a single electric RCMotor."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
            default=None,
        )
        self.name = 'rcpropulsion_mission'

    def setup(self):
        nn = self.options['num_nodes']

        # Full throttle (=1.0) drives the max-thrust chain below.
        self.set_input_defaults('full_throttle', val=np.ones(nn), units='unitless')
        # Provide a single promoted default for battery voltage so promoted inputs
        # from battery, battery_max, and electric_power are not ambiguous at setup.
        self.set_input_defaults(Aircraft.Battery.VOLTAGE, val=22.2, units='V')

        self.add_subsystem(
            'battery', 
            Battery(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Battery.VOLTAGE,  
                Aircraft.Battery.RESISTANCE,
                Dynamic.Vehicle.Propulsion.CURRENT,
            ]
        )

        self.add_subsystem(
            'esc', 
            ElectronicSpeedController(num_nodes=nn), 
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.THROTTLE,
                Dynamic.Vehicle.Propulsion.CURRENT
            ],
        )

        self.add_subsystem(
            'motor', 
            Motor(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Engine.Motor.IDLE_CURRENT, 
                Aircraft.Engine.Motor.MAX_CONT_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE, 
                Aircraft.Engine.Motor.KV,
                Dynamic.Vehicle.Propulsion.CURRENT,
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.RPM,
                ]
        )


        self.add_subsystem('vectorize_geo', Vectorization(num_nodes=nn), 
            promotes_inputs=[Aircraft.Engine.Propeller.DIAMETER, Aircraft.Engine.Propeller.PITCH],
            promotes_outputs=['temp_diameter', 'temp_pitch']
            )

        # Clamp RPM into the propeller surrogate's trained range so out-of-range
        # operating points (which the optimizer can produce) return boundary ct/cp
        # instead of NaN. The true RPM still drives the Propeller thrust formula.
        self.add_subsystem(
            'rpm_clamp',
            RangeClamp(num_nodes=nn, lower=16.7, upper=183.3, units='rev/s'),
            promotes_inputs=[('x_in', Dynamic.Vehicle.Propulsion.RPM)],
        )

        self.add_subsystem(
            'propco',
            PropCoefficients(method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY,
                'temp_diameter',
                'temp_pitch',
            ],
            promotes_outputs=['ct', 'cp']
        )
        self.connect('rpm_clamp.x_out', 'propco.' + Dynamic.Vehicle.Propulsion.RPM)

        self.add_subsystem(
            'prop', 
            Propeller(num_nodes=nn), 
            promotes_inputs=[
                Aircraft.Engine.Propeller.DIAMETER, 
                Dynamic.Vehicle.Propulsion.RPM, 
                'ct', 
                'cp', 
                # Aircraft.Engine.NUM_ENGINES, 
                Dynamic.Atmosphere.DENSITY
                ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER, 
                Dynamic.Vehicle.Propulsion.THRUST,
                # Dynamic.Vehicle.Propulsion.THRUST_MAX,
                ]
        )

        self.add_subsystem(
            'power_net',
            PowerImplicit(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.PROP_POWER,
            ],
            promotes_outputs=[
                Dynamic.Vehicle.Propulsion.CURRENT,
            ]
        )

        # Electrical power drawn from the battery (= open-circuit voltage * current).
        # This is the rate the battery energy depletes, so endurance = energy / power.
        # Promoting it as ELECTRIC_POWER_IN lets PropulsionSum roll it up into
        # ELECTRIC_POWER_IN_TOTAL (already a phase timeseries output).
        self.add_subsystem(
            'electric_power',
            om.ExecComp(
                'p_elec = v_batt * current',
                p_elec={'val': np.zeros(nn), 'units': 'W'},
                v_batt={'val': 22.2, 'units': 'V'},
                current={'val': np.zeros(nn), 'units': 'A'},
            ),
            promotes_inputs=[
                ('v_batt', Aircraft.Battery.VOLTAGE),
                ('current', Dynamic.Vehicle.Propulsion.CURRENT),
            ],
            promotes_outputs=[('p_elec', Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN)],
        )




        self.add_subsystem(
            'battery_max',
            Battery(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Battery.VOLTAGE,
                Aircraft.Battery.RESISTANCE,
                (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
            ]
        )

        self.add_subsystem(
            'esc_max',
            ElectronicSpeedController(num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Vehicle.Propulsion.THROTTLE, 'full_throttle'),
                (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
            ]
        )

        self.add_subsystem(
            'motor_max',
            Motor(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Engine.Motor.IDLE_CURRENT,
                Aircraft.Engine.Motor.MAX_CONT_CURRENT,
                Aircraft.Engine.Motor.RESISTANCE,
                Aircraft.Engine.Motor.KV,
                (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
                ],
            promotes_outputs=[
                (Dynamic.Vehicle.Propulsion.RPM, Dynamic.Vehicle.Propulsion.RPM_MAX),
                ('current_constraint', 'current_constraint_max'),
                ]
        )

        self.add_subsystem(
            'rpm_max_clamp',
            RangeClamp(num_nodes=nn, lower=16.7, upper=183.3, units='rev/s'),
            promotes_inputs=[('x_in', Dynamic.Vehicle.Propulsion.RPM_MAX)],
        )

        self.add_subsystem(
            'propco_max',
            PropCoefficients(method='lagrange2', extrapolate=True, training_data_gradients=True, vec_size=nn),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY,
                'temp_diameter',
                'temp_pitch',
            ],
            promotes_outputs=[('ct', 'ct_max'), ('cp', 'cp_max')]
        )
        self.connect('rpm_max_clamp.x_out', 'propco_max.' + Dynamic.Vehicle.Propulsion.RPM)

        self.add_subsystem(
            'prop_max',
            Propeller(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Engine.Propeller.DIAMETER,
                (Dynamic.Vehicle.Propulsion.RPM, Dynamic.Vehicle.Propulsion.RPM_MAX),
                ('ct', 'ct_max'),
                ('cp', 'cp_max'),
                Dynamic.Atmosphere.DENSITY
                ],
            promotes_outputs=[
                (Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.PROP_POWER_MAX),
                (Dynamic.Vehicle.Propulsion.THRUST, Dynamic.Vehicle.Propulsion.THRUST_MAX)
                ]
        )

        self.add_subsystem(
            'power_net_max',
            PowerImplicit(num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.PROP_POWER_MAX),
            ],
            promotes_outputs=[
                (Dynamic.Vehicle.Propulsion.CURRENT, Dynamic.Vehicle.Propulsion.CURRENT_MAX),
            ]
        )

        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')

        self.connect('battery_max.voltage_out', 'esc_max.voltage_in')
        self.connect('esc_max.voltage_out', 'motor_max.voltage_in')
        self.connect('esc_max.current_out', 'motor_max.current')
        #TODO Alex from phase builder base import add_control

        #self.add_constraint('current_constraint', upper=0, ref=1e2)
        #self.add_constraint(Dynamic.Vehicle.Propulsion.RPM_MAX, lower=1, upper=125, ref=1e3, units='rps')

        #Constraints to prevent ill-fated surrogate model predictions
        # self.add_constraint('ct_max', lower=0, upper=0.12, ref=1.0, units='unitless')
        # self.add_constraint('cp_max', lower=0.0034, upper=0.08, ref=1.0, units='unitless')

        self.connect('battery.power', 'power_net.power_batt')
        self.connect('esc.power', 'power_net.power_esc')
        self.connect('motor.power', 'power_net.power_motor')

        self.connect('battery_max.power', 'power_net_max.power_batt')
        self.connect('esc_max.power', 'power_net_max.power_esc')
        self.connect('motor_max.power', 'power_net_max.power_motor')

        # NonlinearBlockGS (fixed-point) is used rather than Newton+DirectSolver
        # because the propeller metamodel (PropCoefficients) has flat / extrapolated
        # regions with zero gradient. A DirectSolver factorization of the
        # battery<->esc<->motor<->prop<->power_net current cycle goes singular there,
        # whereas this fixed-point iteration tolerates it.
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["maxiter"] = 40
        self.nonlinear_solver.options["use_aitken"] = True
        self.nonlinear_solver.options["err_on_non_converge"] = False

        self.linear_solver = om.LinearBlockGS()

        self.options['auto_order'] = True

        # Newton + DirectSolver alternative (kept for reference): converges the
        # current_flow state in fewer iterations when the metamodel is well inside
        # its trained region, but goes singular on flat extrapolated regions.
        # self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        # self.nonlinear_solver.options["maxiter"] = 15
        # self.nonlinear_solver.options["err_on_non_converge"] = False
        # self.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        # self.nonlinear_solver.linesearch.options["bound_enforcement"] = "scalar"
        # self.linear_solver = om.DirectSolver(assemble_jac=True)

        # # self.add_constraint(Dynamic.Vehicle.Propulsion.CURRENT, lower=0)