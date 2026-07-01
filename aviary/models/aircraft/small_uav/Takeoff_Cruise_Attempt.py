from copy import deepcopy
import numpy as np
import openmdao.api as om
import aviary.api as av
from aviary.models.aircraft.small_uav.phases.dbf_example_energy_phase import phase_info
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder



# Builders
rc_prop = RCBuilder()

phase_info = deepcopy(phase_info)


phase_info.pop('descent')


#Pre-Mission Mass Model

phase_info['pre_mission']['include_takeoff'] = False
phase_info['pre_mission']['optimize_mass'] = False
phase_info['pre_mission']['external_subsystems'] = [DBFMassBuilder()]


#Setup Cruise and Climb
phase_info['climb']['external_subsystems'] = [CustomAeroBuilder()]
phase_info['climb']['subsystem_options']['core_aerodynamics'] = {
    'method': 'external',
}

phase_info['climb']['user_options'].update({
    'num_segments': 5,
    'order': 3,
    'mach_optimize': True,
    'mach_polynomial_order': 1,
    'mach_initial': (0.0538, 'unitless'),   # cruise speed; was 0.538 (typo) and 0.002 (standstill -> RangeRate crash)
    'mach_final': (0.0538, 'unitless'),
    'altitude_optimize': True,
    'altitude_polynomial_order': 1,
    'altitude_initial': (0, 'ft'),
    'altitude_final': (200.0, 'ft'),
    #'distance_initial': (0, 'ft'),
    'distance_ref': (360.0, 'm'),
    'throttle_enforcement': 'bounded',
    # Keep climb near high power while avoiding singular behavior at exactly 1.0.
    'throttle_bounds': ((0.75, 0.98), 'unitless'),
    'electric_current_polynomial_order': 3,
    'electric_current_max_polynomial_order': 3,
    'time_initial': (0.0, 'min'),
    'time_duration_bounds': ((15, 30), 's'),
    'constraints': {
        'mach': {
            'upper': 0.145773,
            'units': 'unitless',
            'type': 'path',
        }
    },
})

phase_info['climb']['initial_guesses'] = {
    'time': ([0.0, 20], 's'),
    'mach': ([0.0538, 0.0538], 'unitless'),
    'distance': ([0.0, 360], 'm'),
}


phase_info['cruise']['external_subsystems'] = [CustomAeroBuilder()]

phase_info['cruise']['subsystem_options']['core_aerodynamics'] = {
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
    'altitude_initial': (200, 'ft'),
    'altitude_final': (200.0, 'ft'),

    #Distance_Target
    'distance_initial': (0.0, 'm'),
    'distance_ref': (1000.0, 'm'),
    'target_distance': (1000.0, 'm'),

    #Scaling
    'mass_ref': (4.0, 'kg'),

    #Aviary Solves throttle

    'throttle_enforcement': 'bounded',
    'throttle_bounds': ((0.2, 0.9), 'unitless'),

    #Time 
    'time_initial': (0.0, 's'),
    'time_duration_bounds': ((25,90), 's'),


})

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
prob.load_external_subsystems(external_subsystems=[rc_prop])

print("Loaded gross mass kg:",
      prob.aviary_inputs.get_val('aircraft:design:gross_mass', units='kg'))
print("Loaded gross mass lbm:",
      prob.aviary_inputs.get_val('aircraft:design:gross_mass', units='lbm'))

prob.check_and_preprocess_inputs()

prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()


prob.add_driver('IPOPT', use_coloring=False)
prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons"]

prob.add_design_variables()

# Objective: MAXIMIZE endurance = battery energy / cruise electric power.
# (Unlike minimizing time at fixed speed, this genuinely depends on the motor/
# battery sizing, so it's a well-posed optimization for the powertrain.)
# add_objective minimizes objective/ref, so a negative ref maximizes endurance.
prob.model.add_subsystem(
    'endurance_comp',
    om.ExecComp(
        'endurance = energy / (p_cruise + 1.0e-3)',  # energy [W*h] / power [W] -> [h]
        endurance={'val': 1.0, 'units': 'h'},
        energy={'val': 1.0, 'units': 'W*h'},
        p_cruise={'val': 1.0, 'units': 'W'},
    ),
)
prob.model.connect('aircraft:battery:energy_capacity', 'endurance_comp.energy')
prob.model.connect(
    'traj.cruise.timeseries.electric_power_in_total',
    'endurance_comp.p_cruise',
    src_indices=[0],
)
prob.model.add_objective('endurance_comp.endurance', ref=-1.0)

prob.setup()

prob.set_solver_print(level=0)



prob.set_initial_guesses()


prob.set_val('aircraft:engine:motor:mass', 0.45, units='kg')   # mid of [0.25, 0.65] -> KV ~370
prob.set_val('aircraft:engine:motor:idle_current', 2.0, units='A')


prob.set_val('aircraft:battery:voltage', 25.2, units='V')
print("Battery voltage set to:", prob.get_val('aircraft:battery:voltage', units='V'))

# Climb warm-start: higher power demand means higher RPM/current/throttle
_climb_rpm_targets = [
    'traj.phases.climb.rhs_all.solver_sub.core_propulsion.rc_electric.rotations_per_minute',
    'traj.phases.climb.rhs_all.core_propulsion.rc_electric.rotations_per_minute',
]
_climb_current_targets = [
    'traj.phases.climb.rhs_all.solver_sub.core_propulsion.rc_electric.current_flow',
    'traj.phases.climb.rhs_all.core_propulsion.rc_electric.current_flow',
]
_climb_throttle_targets = [
    'traj.phases.climb.rhs_all.solver_sub.throttle',
    'traj.phases.climb.rhs_all.solver_sub.core_propulsion.rc_electric.throttle',
    'traj.phases.climb.rhs_all.throttle',
]
for _t in _climb_rpm_targets:
    try:
        prob.set_val(_t, val=4800.0, units='rpm')
        print(f"Initial climb RPM guess set at: {_t}")
        break
    except Exception:
        continue
for _t in _climb_current_targets:
    try:
        prob.set_val(_t, val=65.0, units='A')
        print(f"Initial climb current guess set at: {_t}")
        break
    except Exception:
        continue
for _t in _climb_throttle_targets:
    try:
        prob.set_val(_t, val=0.90, units='unitless')
        print(f"Initial climb throttle guess set at: {_t}")
        break
    except Exception:
        continue


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





# --- Diagnostics: why does the driver drive motor:mass to 2.187 while set_val=0.45? ---
# final_setup() so list_driver_vars / get_design_var_values are callable.
prob.final_setup()
print("motor:mass model get_val (kg):", prob.get_val('aircraft:engine:motor:mass', units='kg'))
_dv_u = prob.driver.get_design_var_values(driver_scaling=False)
_dv_s = prob.driver.get_design_var_values(driver_scaling=True)
for _k in _dv_u:
    if 'motor' in _k and 'mass' in _k:
        print(f"DRIVER DV '{_k}': unscaled={_dv_u[_k]}  scaled={_dv_s[_k]}")
try:
    prob.list_driver_vars(desvar_opts=['lower', 'upper', 'ref', 'ref0', 'scaler', 'adder'])
except Exception as _e:
    print("list_driver_vars failed:", _e)

# Run
prob.run_aviary_problem(suppress_solver_print=False)

print("\n===== MASS CHECK =====")

print("Design Gross Mass")
print("kg :", prob.get_val('aircraft:design:gross_mass', units='kg'))
print("lbm:", prob.get_val('aircraft:design:gross_mass', units='lbm'))

print("\nTrajectory Mass")
print("kg :", prob.get_val('traj.cruise.states:mass', units='kg'))
print("lbm:", prob.get_val('traj.cruise.states:mass', units='lbm'))

print("\nSummary Gross Mass")
print("kg :", prob.get_val('mission:summary:gross_mass', units='kg'))
print("lbm:", prob.get_val('mission:summary:gross_mass', units='lbm'))

# Save variable list
with open("aviary/examples/small_uav/climb_&_cruise_vars.txt", "w") as f:
    prob.model.list_vars(print_arrays=True, out_stream=f, units=True)