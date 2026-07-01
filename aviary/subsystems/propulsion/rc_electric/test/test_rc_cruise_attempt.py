import unittest
import subprocess
import sys
from pathlib import Path
import aviary.api as av
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder 
from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_premission import RCPropPreMission
from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_mission import RCPropMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.dbf_variables import Aircraft, Dynamic


class MeanPowerComp(om.ExplicitComponent):
    # Simple smoother objective input: average cruise electric power over the full phase.
    def setup(self):
        self.add_input('p_cruise_kw', shape_by_conn=True, units='kW')
        self.add_output('p_avg_kw', val=1.0, units='kW')
        self.declare_partials('p_avg_kw', 'p_cruise_kw', method='fd')

    def compute(self, inputs, outputs):
        outputs['p_avg_kw'] = np.mean(inputs['p_cruise_kw'])


""" Expected Values """

class TestRCPropMission(unittest.TestCase):
    
    def test_residual(self):
        nn = 3

        prob = om.Problem()
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, 1)
        prob.model.add_subsystem('rc_prop_group', RCPropMission(num_nodes=nn, aviary_options= options), promotes=['*'])

        # Solve the implicit current balance with Newton for this residual test.
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.nonlinear_solver.options['maxiter'] = 30
        prob.model.nonlinear_solver.options['err_on_non_converge'] = True
        prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        prob.model.nonlinear_solver.linesearch.options['bound_enforcement'] = 'scalar'
        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
        
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
        prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.full(nn, 0.8))
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
        prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
        prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
        prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
        prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
        prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
        prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')

        prob.run_model()

        battery_power = prob.get_val('battery.power', units='W')
        esc_power = prob.get_val('esc.power', units='W')
        motor_power = prob.get_val('motor.power', units='W')

    def test_premission_calcs(self):
        prob = om.Problem()
        options = AviaryValues()
        options.set_val(Aircraft.Engine.Motor.KV_EQ_SLOPE, 2105.53674)
        options.set_val(Aircraft.Engine.Motor.KV_EQ_INT, -80.83469)

        prob.model.add_subsystem(
            'rc_calcs', RCPropPreMission(aviary_options=options), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.MASS, 0.707, units='kg')
        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
        prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
        prob.set_val(Aircraft.Engine.Motor.MASS, 0.288, units='kg')

        prob.run_model()

        kv = prob.get_val(Aircraft.Engine.Motor.KV, 'rpm/V')
        resistance = prob.get_val(Aircraft.Engine.Motor.RESISTANCE, 'ohm')
        energy = prob.get_val(Aircraft.Battery.ENERGY_CAPACITY, 'W*h')

        kv_expected = 796.472285
        resistance_expected = 0.05582266503
        energy_expected = 109.11522
        assert_near_equal(kv, kv_expected, tolerance=1e-9)
        assert_near_equal(resistance, resistance_expected, tolerance=1e-9)
        assert_near_equal(energy, energy_expected, tolerance=1e-9)
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=
1e-12)
    
       
    


# NOTE: no @use_tempdirs here. DBFMassBuilder reads its airfoil CSV via a repo-root-
# relative path (like the dbf_based_mass unit tests), so this must run from the repo root.
class TestRCCruiseAttempt(unittest.TestCase):
    def setUp(self):
        self.phase_info = {'pre_mission': {'include_takeoff': False,  'optimize_mass': False, 'external_subsystems': [DBFMassBuilder()]},
   
    'cruise': {
        'subsystem_options': {'aerodynamics': {'method': 'external'}},
        'external_subsystems': [CustomAeroBuilder()],
        'user_options': {   # matches Cruise_Attempt.py exactly
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_initial': (0.0538, 'unitless'),
            'mach_final': (0.0538, 'unitless'),
            'altitude_optimize': False,
            'altitude_initial': (200.0, 'ft'),
            'altitude_final': (200.0, 'ft'),
            'distance_initial': (0.0, 'm'),
            'distance_ref': (1000.0, 'm'),
            'target_distance': (1000.0, 'm'),
            'mass_ref': (4.0, 'kg'),
            'throttle_enforcement': 'bounded',
            'throttle_bounds': ((0.2, 0.9), 'unitless'),
            'time_initial': (0.0, 's'),
            'time_duration_bounds': ((20, 90.0), 's'),
        },
        'initial_guesses': {
            'time': ([0.0, 54.7], 's'),
            'distance': ([0.0, 1000.0], 'm'),
            'mach': ([0.0538, 0.0538], 'unitless'),
        },
    },
    
    'post_mission': {
        'include_landing': False,
        # 'target_range': (200, 'ft'),
        # 'constraint_range':True, 
    }
        }   

    def test_subsystems_in_cruise_attempt(self):
        phase_info = self.phase_info.copy()

        prob = av.AviaryProblem(verbosity=0)
        prob.options['group_by_pre_opt_post'] = True

        # Load aircraft and options data from user
        # Allow for user overrides here
        # add engine builder
        prob.load_inputs(
            'validation_cases/validation_data/test_models/small_scale_uav.csv',
            phase_info,
        )

       
        prob.load_external_subsystems(external_subsystems=[RCBuilder(), CustomAeroBuilder()])

        prob.aviary_inputs.set_val(Aircraft.Engine.Propeller.PITCH, 12.0, units='inch')


        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()
        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver('IPOPT', use_coloring=False, max_iter=50)
        prob.driver.opt_settings['print_level'] = 0
        prob.driver.opt_settings['mu_strategy'] = 'adaptive'
        prob.driver.opt_settings['tol'] = 1e-5
        prob.driver.opt_settings['acceptable_tol'] = 1e-4
        prob.driver.opt_settings['acceptable_iter'] = 10

        prob.driver.options["debug_print"] = []


        prob.add_design_variables()

        # Aviary already added these with transport scale (ref=175e3 lbm, upper=None), and
        # add_design_var won't overwrite. Drop the existing entries, then re-add at UAV scale.
        del prob.model._static_design_vars['aircraft:design:gross_mass']
        del prob.model._static_design_vars['mission:gross_mass']
        prob.model.add_design_var('aircraft:design:gross_mass', units='kg', lower=2.0, upper=20.0, ref=7.0)
        prob.model.add_design_var('mission:gross_mass', units='kg', lower=2.0, upper=20.0, ref=7.0)

        prob.model.add_subsystem(
            'mean_power_comp',
            MeanPowerComp(),
        )

        prob.model.add_subsystem(
            'endurance_comp',
            om.ExecComp(
                # Simple robustness change: use phase-mean power instead of one point.
                'endurance = energy / (1000.0 * p_avg_kw + 1.0e-3)',
                endurance={'val': 1.0, 'units': 'h'},
                energy={'val': 1.0, 'units': 'W*h'},
                p_avg_kw={'val': 1.0, 'units': 'kW'},
            ),
        )
        prob.model.connect('aircraft:battery:energy_capacity', 'endurance_comp.energy')
        prob.model.connect(
            'traj.cruise.timeseries.electric_power_in_total',
            'mean_power_comp.p_cruise_kw',
        )
        prob.model.connect('mean_power_comp.p_avg_kw', 'endurance_comp.p_avg_kw')
        prob.model.add_objective('endurance_comp.endurance', ref=-1.0)
        
        prob.model.set_input_defaults('aircraft:battery:voltage', val=22.2, units='V')
        prob.model.set_input_defaults('aircraft:engine:motor:idle_current', val=2.2, units='A')
        prob.model.set_input_defaults('aircraft:engine:motor:max_cont_current', val=100.0, units='A')

        prob.setup()

        prob.set_solver_print(level=0)

        # Set initial guesses for key variables to help SLSQP converge to a solution.
        prob.set_initial_guesses()
        prob.set_val('aircraft:design:gross_mass', 7.0, units='kg')
        prob.set_val('mission:gross_mass', 7.0, units='kg')
        prob.set_val('aircraft:engine:motor:mass', 0.55, units='kg')   # mid of [0.47, 0.65] -> rpm_max in-grid
        prob.set_val('aircraft:engine:motor:idle_current', 2.0, units='A')
        prob.set_val('aircraft:battery:voltage', 25.2, units='V')

        _rpm_targets = [
            'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.rotations_per_minute',
            'traj.phases.cruise.rhs_all.core_propulsion.rc_electric.rotations_per_minute',
        ]
        _current_targets = [
            'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.current_flow',
            'traj.phases.cruise.rhs_all.core_propulsion.rc_electric.current_flow',
        ]
        _throttle_targets = [
            'traj.phases.cruise.rhs_all.solver_sub.throttle',
            'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.throttle',
            'traj.phases.cruise.rhs_all.throttle',
        ]
        for _t in _rpm_targets:
            try:
                prob.set_val(_t, val=3750.0, units='rpm')
                break
            except Exception:
                continue
        for _t in _current_targets:
            try:
                prob.set_val(_t, val=12.0, units='A')
                break
            except Exception:
                continue
        for _t in _throttle_targets:
            try:
                prob.set_val(_t, val=0.54, units='unitless')
                break
            except Exception:
                continue

        # Step 1 (continuation): single-pass model run to settle coupled states.
        prob.run_aviary_problem(run_driver=False, suppress_solver_print = True, make_plots=False)

        # Step 2: optimization from that settled point.
        prob.run_aviary_problem(suppress_solver_print = True, make_plots=False)

        # Regression intent: this setup should run without NaN/Inf in the solved-throttle
        # powertrain loop (the prior failure was a NaN Jacobian row on throttle).
        endurance = prob.get_val('endurance_comp.endurance', units='h')[0]
        self.assertTrue(np.isfinite(endurance) and endurance > 0.0)

        gross_mass = prob.get_val('mission:gross_mass', units='kg')[0]
        self.assertTrue(np.isfinite(gross_mass) and 2.0 <= gross_mass <= 20.0)

        # Invariants: motor mass stays within its [0.25, 0.65] kg DV bound, and the
        # 1 km cruise distance target is met.
        mm = prob.get_val('aircraft:engine:motor:mass', units='kg')[0]
        self.assertTrue(0.25 < mm < 0.65, f"Motor mass {mm} kg is outside expected bounds.")

        i_curve = prob.get_val('traj.cruise.controls:current_flow', units='A')
        self.assertFalse(np.isnan(i_curve).any(), 'Current control contains NaN values.')

        self.assertLess(
            abs(prob.get_val('cruise_distance_constraint.distance_resid')[0]), 1e-2,
            "Cruise distance constraint is not satisfied after optimization.")

    def test_cruise_attempt_script_smoke(self):
        repo_root = Path(__file__).resolve().parents[5]
        script = repo_root / 'aviary' / 'models' / 'aircraft' / 'small_uav' / 'Cruise_Attempt.py'

        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=900,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=(
                'Cruise_Attempt.py exited non-zero.\n'
                f'STDOUT (tail):\n{result.stdout[-4000:]}\n\n'
                f'STDERR (tail):\n{result.stderr[-4000:]}'
            ),
        )





if __name__ == '__main__':
    unittest.main()