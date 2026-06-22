import numpy as np
import openmdao.api as om

from aviary.mission.base_ode import BaseODE as _BaseODE
from aviary.mission.flops_based.ode.mission_EOM import MissionEOM

from aviary.subsystems.propulsion.throttle_allocation import ThrottleAllocator
from aviary.variable_info.enums import SpeedType, ThrottleAllocation
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class EnergyODE(_BaseODE):
    """The base class for all energy method ODE components."""

    def initialize(self):
        super().initialize()

        self.options.declare(
            'use_actual_takeoff_mass',
            default=True, #TODO alex find why this doesnt work
            desc='flag to use actual takeoff mass in the climb phase, otherwise assume 100 kg fuel burn',
        )
        # TODO throttle enforcement & allocation should be moved to BaseODE for
        # use in 2DOF
        self.options.declare(
            'throttle_enforcement',
            default='path_constraint',
            values=['path_constraint', 'boundary_constraint', 'bounded', 'control', None],
            desc='Flag to enforce engine throttle bounds as path constraints, boundary '
            'constraints, solver bounds. You can also select "control" to turn throttle into a '
            'control, which allows you to assign a value or let the optimizer choose it.',
        )
        self.options.declare(
            'throttle_allocation',
            default=ThrottleAllocation.FIXED,
            types=ThrottleAllocation,
            desc='Flag that determines how to handle throttles for multiple engines.',
        )

    def setup(self):
        options = self.options
        nn = options['num_nodes']
        aviary_options = options['aviary_options']
        num_engine_type = len(aviary_options.get_val(Aircraft.Engine.NUM_ENGINES))

        self.add_atmosphere(input_speed_type=SpeedType.MACH)

        # add execcomp to compute velocity_rate based off mach_rate and sos
        self.add_subsystem(
            name='velocity_rate_comp',
            subsys=om.ExecComp(
                'velocity_rate = mach_rate * sos',
                mach_rate={'units': '1/s', 'shape': (nn,)},
                sos={'units': 'm/s', 'shape': (nn,)},
                velocity_rate={'units': 'm/s**2', 'shape': (nn,)},
                has_diag_partials=True,
            ),
            promotes_inputs=[
                ('mach_rate', Dynamic.Atmosphere.MACH_RATE),
                ('sos', Dynamic.Atmosphere.SPEED_OF_SOUND),
            ],
            promotes_outputs=[('velocity_rate', Dynamic.Mission.VELOCITY_RATE)],
        )

        throttle_enforcement = options['throttle_enforcement']

        sub1 = self.add_subsystem('solver_sub', om.Group(), promotes=['*'])

        if throttle_enforcement == 'control':
            solver_group = None
            core_needs_solver = False
        else:
            solver_group = sub1
            core_needs_solver = True

        self.add_core_subsystems(solver_group=solver_group)

        ext_needs_solver = self.add_external_subsystems(solver_group=sub1)

        sub1.add_subsystem(
            name='mission_EOM',
            subsys=MissionEOM(num_nodes=nn),
            promotes_inputs=[
                Dynamic.Mission.VELOCITY,
                Dynamic.Vehicle.MASS,
                Dynamic.Vehicle.Propulsion.THRUST_MAX_TOTAL,
                Dynamic.Vehicle.DRAG,
                Dynamic.Mission.ALTITUDE_RATE,
                Dynamic.Mission.VELOCITY_RATE,
            ],
            promotes_outputs=[
                Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS,
                Dynamic.Mission.ALTITUDE_RATE_MAX,
                Dynamic.Mission.DISTANCE_RATE,
                'thrust_required',
            ],
        )

        # THROTTLE Section
        # TODO: Split this out into a function that can be used by the other ODEs.
        # TODO: Need a thrust residual ref in the phase_info.
        thrust_res_ref = 1.0 #TODO Alex mod from 1.0e6 ADD AS OPTION
        if num_engine_type > 1:
            # Multi Engine

            sub1.add_subsystem(
                name='throttle_balance',
                subsys=om.BalanceComp(
                    name='aggregate_throttle',
                    units='unitless',
                    val=np.ones((nn,)),
                    lhs_name='thrust_required',
                    rhs_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                    eq_units='lbf',
                    normalize=False,
                    res_ref=thrust_res_ref,
                ),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

            sub1.add_subsystem(
                'throttle_allocator',
                ThrottleAllocator(
                    num_nodes=nn, throttle_allocation=self.options['throttle_allocation']
                ),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        else:
            # Single Engine

            if throttle_enforcement == 'control':
                self.add_subsystem(
                    'throttle_balance',
                    om.ExecComp(
                        'thrust_residual=thrust_required-thrust',
                        thrust={'val': np.ones((nn,)), 'units': 'lbf'},
                        thrust_required={'val': np.ones((nn,)), 'units': 'lbf'},
                        thrust_residual={'val': np.ones((nn,)), 'units': 'lbf'},
                        has_diag_partials=True,
                    ),
                    promotes_inputs=[
                        ('thrust', Dynamic.Vehicle.Propulsion.THRUST_TOTAL),
                        'thrust_required',
                    ],
                    promotes_outputs=['*'],
                )
                self.add_constraint('thrust_residual', ref=thrust_res_ref, equals=0.0)
            else:
                # Add a balance comp to compute throttle based on the required thrust.
                sub1.add_subsystem(
                    name='throttle_balance',
                    subsys=om.BalanceComp(
                        name=Dynamic.Vehicle.Propulsion.THROTTLE,
                        units='unitless',
                        val=np.ones((nn,)),
                        lhs_name='thrust_required',
                        rhs_name=Dynamic.Vehicle.Propulsion.THRUST_TOTAL,
                        eq_units='lbf',
                        normalize=False,
                        lower=0.0 if throttle_enforcement == 'bounded' else None,
                        upper=1.0 if throttle_enforcement == 'bounded' else None,
                        res_ref=thrust_res_ref,
                    ),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

        self.set_input_defaults(Dynamic.Vehicle.Propulsion.THROTTLE, val=np.ones(nn) * 0.7, units='unitless')
        ############## ALEX ADDITION ###############
        self.set_input_defaults(Dynamic.Atmosphere.MACH_RATE, val=np.zeros(nn))
        ############## ALEX ADDITION ###############
        self.set_input_defaults(Dynamic.Atmosphere.MACH, val=np.ones(nn), units='unitless')
        self.set_input_defaults(Dynamic.Vehicle.MASS, val=np.ones(nn), units='kg')
        self.set_input_defaults(Dynamic.Mission.VELOCITY, val=np.ones(nn), units='m/s')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE, val=np.ones(nn), units='m')
        self.set_input_defaults(Dynamic.Mission.ALTITUDE_RATE, val=np.zeros(nn), units='m/s')

        if options['use_actual_takeoff_mass']:
            exec_comp_string = 'initial_mass_residual = initial_mass - mass[0]'
            initial_mass_string = Mission.Takeoff.FINAL_MASS
        else:
            exec_comp_string = 'initial_mass_residual = initial_mass - mass[0] - 100.'
            initial_mass_string = Mission.Summary.GROSS_MASS

        # Experimental: Add a component to constrain the initial mass to be equal
        # to design gross weight.
        # initial_mass_residual_constraint = om.ExecComp(
        #     exec_comp_string,
        #     initial_mass={'units': 'kg'},
        #     mass={'units': 'kg', 'shape': (nn,)},
        #     initial_mass_residual={'units': 'kg', 'res_ref': 1.0},
        # )

        # self.add_subsystem(
        #     'initial_mass_residual_constraint',
        #     initial_mass_residual_constraint,
        #     promotes_inputs=[
        #         ('initial_mass', initial_mass_string),
        #         ('mass', Dynamic.Vehicle.MASS),
        #     ],
        #     promotes_outputs=['initial_mass_residual'],
        # )

        if core_needs_solver or ext_needs_solver:
            # atol/rtol of 1e-10 is unrealistically tight for a metamodel-based
            # thrust residual (~1 lbf scale) and left the Newton failing to converge
            # in the default 10 iterations. 1e-8 with more iterations converges
            # cleanly while still being well below solver/optimizer tolerances.
            sub1.nonlinear_solver = om.NewtonSolver(
                solve_subsystems=True,
                atol=1.0e-8,
                rtol=1.0e-8,
                maxiter=60,
            )
            print_level = 2

            sub1.nonlinear_solver.linesearch = om.BoundsEnforceLS()
            sub1.linear_solver = om.DirectSolver(assemble_jac=True)
            # Use a DENSE assembled Jacobian. Only the dense LU path honors
            # err_on_singular=False; the sparse (splu) path always raises on a
            # singular factorization. solver_sub is small, so dense is cheap.
            sub1.options['assembled_jac_type'] = 'dense'
            # Robustness: do NOT raise when the cruise solve fails to converge. The
            # optimizer probes infeasible trial points (e.g. a too-slow motor that
            # can't produce cruise thrust); raising there crashes the whole run.
            # Returning the non-converged (still finite) state instead lets the
            # optimizer see a bad point and step away.
            sub1.nonlinear_solver.options['err_on_non_converge'] = False
            # Likewise, don't raise if the assembled Jacobian is singular when
            # computing derivatives at a flat-gradient (windmilling) operating point;
            # return least-squares derivatives so the optimizer can continue.
            sub1.linear_solver.options['err_on_singular'] = False
            sub1.nonlinear_solver.options['iprint'] = print_level

        self.options['auto_order'] = True