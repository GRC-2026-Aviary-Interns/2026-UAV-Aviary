"""
Define an aviary mission with an NPSS defined engine. During pre-mission the engine is designed and
an engine deck is made. During the mission the deck is used for performance. Weight is estimated
using the default Aviary method. The engine model was developed using NPSS v3.2.
"""
from copy import deepcopy
import numpy as np
import aviary.api as av
from pathlib import Path
from aviary.examples.small_uav.phases.dbf_example_energy_phase import phase_info
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder
import inspect
from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_mission import RCPropMission

print("\nUSING RCPropMission FROM:")
print(inspect.getfile(RCPropMission))

rc_prop = RCBuilder()

phase_info = deepcopy(phase_info)

# phase_info.pop('climb')
# phase_info.pop('cruise')
phase_info.pop('descent')

phase_info['pre_mission']['external_subsystems'] = [DBFMassBuilder()]

phase_info['climb']['external_subsystems'] = [CustomAeroBuilder()]
phase_info['climb']['subsystem_options']['core_aerodynamics'] = {
    'method': 'external',
}

phase_info['climb']['user_options']['time_duration_bounds'] = ((20.0, 90.0), 's')
phase_info['climb']['initial_guesses']['time'] = ([0.0, 40.0], 's')



phase_info['climb']['user_options']['mach_optimize'] = False
phase_info['climb']['user_options']['mach_initial'] = (0.02, 'unitless')
phase_info['climb']['user_options']['mach_final'] = (0.02, 'unitless')
phase_info['climb']['initial_guesses']['mach'] = ([0.02, 0.02], 'unitless')

phase_info['climb']['user_options']['altitude_optimize'] = False
phase_info['climb']['user_options']['altitude_initial'] = (0.0, 'ft')
phase_info['climb']['user_options']['altitude_final'] = (200.0, 'ft')
phase_info['climb']['initial_guesses']['altitude'] = ([0.0, 200.0], 'ft')
phase_info['cruise']['external_subsystems'] = [CustomAeroBuilder()]
phase_info['cruise']['subsystem_options']['core_aerodynamics'] = {
    'method': 'external',
}
phase_info['cruise']['user_options']['mach_optimize'] = False
phase_info['cruise']['user_options']['mach_initial'] = (0.02, 'unitless')
phase_info['cruise']['user_options']['mach_final'] = (0.02, 'unitless')

phase_info['cruise']['user_options']['altitude_optimize'] = False
phase_info['cruise']['user_options']['altitude_initial'] = (200.0, 'ft')
phase_info['cruise']['user_options']['altitude_final'] = (200.0, 'ft')
phase_info['climb']['initial_guesses']['mass'] = ([4.6, 4.6], 'kg')

phase_info['cruise']['user_options']['time_duration_bounds'] = ((5, 80.0), 's')
phase_info['cruise']['initial_guesses']['time'] = ([0.0, 10.0], 's')

phase_info['cruise']['initial_guesses']['mass'] = ([4.6, 4.6], 'kg')

prob = av.AviaryProblem(verbosity=1)
prob.options['group_by_pre_opt_post'] = True

# Load aircraft and options data from user
# Allow for user overrides here
# add engine builder
prob.load_inputs(
    'models/aircraft/test_aircraft/small_scale_uav.csv',
    phase_info,
    engine_builders=[rc_prop],
)

prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()

prob.add_phases()
prob.add_post_mission_systems()

# Link phases and variables
prob.link_phases()

prob.add_driver('IPOPT')
prob.driver.opt_settings['max_iter'] = 300
prob.driver.opt_settings['tol'] = 1.0E-6
prob.driver.options["debug_print"] = []


prob.add_design_variables()
# Electric aircraft: no fuel is burned, so initial mass should equal final mass.
# gross takeoff mass = initial mass = final mass

prob.model.add_constraint(
    'mission:summary:fuel_burned',
    equals=0.0,
    units='lbm',
    ref=1.0,
)

prob.add_objective('time')

prob.setup()

print("\nDESIGN VARS:")
print(prob.model.get_design_vars().keys())

print("\nMOTOR MASS DESIGN VAR INFO:")
print(prob.model.get_design_vars()['aircraft:engine:motor:mass'])

print("\nMOTOR MAX CONT CURRENT DESIGN VAR INFO:")
print(prob.model.get_design_vars()['aircraft:engine:motor:max_cont_current'])

prob.set_initial_guesses()
#TODO: ALEX, ask find_feasible question
# prob.find_feasible()

prob.set_val('traj.climb.controls:current_flow', np.full((3, 1), 1.0), units='A')
prob.set_val('traj.cruise.controls:current_flow', np.full((3, 1), 22.0), units='A')

prob.set_val('traj.climb.controls:current_flow_max', np.full((3, 1), 20.0), units='A')
prob.set_val('traj.cruise.controls:current_flow_max', np.full((3, 1), 20.0), units='A')


failed = prob.run_aviary_problem(suppress_solver_print=False)
print("FAILED =", failed)


print("\nDESIGN VARIABLES WITH UNITS")
for name, meta in prob.model.get_design_vars().items():
    units = meta["units"]
    val = prob.get_val(name, units=units)
    print(f"{name}: {val} {units}")

print("\n" + "="*70)
print("OPTIMIZATION CHECK SUMMARY")
#output with units 
checks = [
    'cruise_distance_constraint.distance_resid',
    'gtow_constraint.GTOW',
    'mission:constraints:mass_residual',
    'mission:summary:fuel_burned',
    'traj.climb.rhs_all.thrust_residual',
    'traj.cruise.rhs_all.thrust_residual',
    'traj.climb.t_duration',
    'traj.cruise.t_duration'
    ]

for name in checks:
    try:
        val = prob.get_val(name)
        print(f"\n{name}:")
        print(val)
        print("max abs =", abs(val).max())
    except Exception as e:
        print(f"\n{name}: NOT FOUND")
        print(e)


output_file = Path(__file__).parent / "level2_newvars.txt"

with open(output_file, "w") as f:
    prob.model.list_vars(print_arrays=True, out_stream=f, units=True)
       
