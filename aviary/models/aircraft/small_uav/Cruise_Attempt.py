from copy import deepcopy
import numpy as np
import openmdao.api as om
import aviary.api as av
from aviary.models.aircraft.small_uav.phases.dbf_example_energy_phase import phase_info
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder


class MeanPowerComp(om.ExplicitComponent):
    def setup(self):
        self.add_input('p_cruise_kw', shape_by_conn=True, units='kW')
        self.add_output('p_avg_kw', val=1.0, units='kW')
        self.declare_partials('p_avg_kw', 'p_cruise_kw', method='fd')

    def compute(self, inputs, outputs):
        outputs['p_avg_kw'] = np.mean(inputs['p_cruise_kw'])



# Builders
# defaults to feedforward power balance; change to power_balance_mode='solver' for solver-based power balance
rc_prop = RCBuilder()

phase_info = deepcopy(phase_info)



phase_info.pop('climb')
phase_info.pop('descent')


#Pre-Mission Mass Model

phase_info['pre_mission']['include_takeoff'] = False
phase_info['pre_mission']['optimize_mass'] = False
phase_info['pre_mission']['external_subsystems'] = [DBFMassBuilder()]


#Setup Cruise

phase_info['cruise']['external_subsystems'] = [CustomAeroBuilder()]

phase_info['cruise']['subsystem_options']['aerodynamics'] = {
    'method': 'external',
}

phase_info['cruise']['user_options'].update({
    'num_segments': 5,
    'order': 3,

    
    'mach_optimize': False,
    'mach_initial': (0.0538, 'unitless'),
    'mach_final': (0.0538, 'unitless'),

    # Level Cruise
    'altitude_optimize': False,
    'altitude_initial': (200.0, 'ft'),
    'altitude_final': (200.0, 'ft'),

    #Distance_Target
    'distance_initial': (0.0, 'm'),
    'distance_ref': (1000.0, 'm'),
    'target_distance': (1000.0, 'm'),

    #Scaling
    'mass_ref': (4.0, 'kg'),

    #Aviary Solves throttle

    'throttle_enforcement': 'bounded',

    #Time 
    'time_initial': (0.0, 's'),
    'time_duration_bounds': ((20,90.0), 's'),


})


phase_info['cruise']['user_options'].pop('electric_current_polynomial_order', None)
phase_info['cruise']['user_options'].pop('electric_current_max_polynomial_order', None)

phase_info['cruise']['initial_guesses'] = {
    'time': ([0.0, 54.7], 's'),   # 1 km at ~18.29 m/s (60 ft/s) ~= 54.7 s
    'distance': ([0.0, 1000.0], 'm'),
    'mach': ([0.0538, 0.0538], 'unitless'),

}

# -----------------------------
# Aviary problem
# -----------------------------
prob = av.AviaryProblem(verbosity=0)
prob.options['group_by_pre_opt_post'] = True

prob.load_inputs(
    'validation_cases/validation_data/test_models/small_scale_uav.csv',
    phase_info,
)
prob.load_external_subsystems(external_subsystems=[rc_prop, CustomAeroBuilder()])

for _k in ('mission:design:gross_mass', 'aircraft:design:gross_mass', 'mission:gross_mass'):
    try:
        print("Loaded gross mass key:", _k)
        print("Loaded gross mass kg:", prob.aviary_inputs.get_val(_k, units='kg'))
        print("Loaded gross mass lbm:", prob.aviary_inputs.get_val(_k, units='lbm'))
        break
    except KeyError:
        continue

prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()


prob.add_driver('IPOPT', use_coloring=False)
prob.driver.opt_settings['print_level'] = 5
prob.driver.opt_settings['mu_strategy'] = 'adaptive'
prob.driver.opt_settings['tol'] = 1e-5
prob.driver.opt_settings['acceptable_tol'] = 1e-4
prob.driver.opt_settings['acceptable_iter'] = 10
prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons"]

prob.add_design_variables()

# Objective: MAXIMIZE endurance = battery energy / cruise electric power.
prob.model.add_subsystem(
    'mean_power_comp',
    MeanPowerComp(),
)

prob.model.add_subsystem(
    'endurance_comp',
    om.ExecComp(
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



prob.set_initial_guesses()

prob.set_val('aircraft:design:gross_mass', 7.0, units='kg')
prob.set_val('mission:gross_mass', 7.0, units='kg')
prob.set_val('aircraft:battery:voltage', 25.2, units='V')

# Start the motor-sizing design variables strictly INSIDE bounds. The CSV
# motor mass (0.131 kg, also seen as 0.637) is below the [0.68, 1.12] kg DV bound,
# and an infeasible-to-bounds start makes IPOPT crash. 
prob.set_val('aircraft:engine:motor:mass', 0.45, units='kg')   # mid of [0.25, 0.65] -> KV ~370
prob.set_val('aircraft:engine:motor:idle_current', 2.0, units='A')


_rpm_targets = [
    'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.rotations_per_minute',
    'traj.phases.cruise.rhs_all.core_propulsion.rc_electric.rotations_per_minute',
]
_current_targets = [
    'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.current_flow',
    'traj.phases.cruise.rhs_all.core_propulsion.rc_electric.current_flow',
]
# The throttle-balance solver defaults throttle to 1.0, which drives the prop to a
# very different operating point than the cruise solution; seeding a mid-range
# throttle keeps the Newton's first linearization well-conditioned.
_throttle_targets = [
    'traj.phases.cruise.rhs_all.solver_sub.throttle',
    'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.throttle',
    'traj.phases.cruise.rhs_all.throttle',
]
for _t in _rpm_targets:
    try:
        prob.set_val(_t, val=3750.0, units='rpm')
        print(f"Initial RPM guess set at: {_t}")
        break
    except Exception:
        continue
for _t in _current_targets:
    try:
        prob.set_val(_t, val=12.0, units='A')
        print(f"Initial current guess set at: {_t}")
        break
    except Exception:
        continue
for _t in _throttle_targets:
    try:
        prob.set_val(_t, val=0.54, units='unitless')
        print(f"Initial throttle guess set at: {_t}")
        break
    except Exception:
        continue

# Run in two steps for continuation robustness.
prob.run_aviary_problem(run_driver=False, suppress_solver_print=True, make_plots=False)
prob.run_aviary_problem(suppress_solver_print=True, make_plots=False)

print("\n===== MASS CHECK =====")

print("Design Gross Mass")
print("kg :", prob.get_val('aircraft:design:gross_mass', units='kg'))
print("lbm:", prob.get_val('aircraft:design:gross_mass', units='lbm'))

print("\nTrajectory Mass")
print("kg :", prob.get_val('traj.cruise.states:mass', units='kg'))
print("lbm:", prob.get_val('traj.cruise.states:mass', units='lbm'))

print("\nSummary Gross Mass")
print("kg :", prob.get_val('mission:gross_mass', units='kg'))
print("lbm:", prob.get_val('mission:gross_mass', units='lbm'))

# Save variable list -- needs cruise_only_vars.txt in order to work
# with open("aviary/examples/small_uav/cruise_only_vars.txt", "w") as f:
#     prob.model.list_vars(print_arrays=True, out_stream=f, units=True)