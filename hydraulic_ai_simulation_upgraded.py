"""
AI-Driven Energy Optimization of Hydraulic Power Units
Upgraded educational VPython simulation.

Install:
    python3 -m pip install vpython

Run:
    python3 hydraulic_ai_simulation_upgraded.py

Features:
- Interactive 3D hydraulic power-unit model
- Live pressure, flow, power, temperature and motor-speed monitoring
- Online exponential-moving-average demand prediction
- Adaptive motor-speed control (rule-based, not a trained AI model)
- Baseline-versus-optimized power comparison
- Cumulative energy comparison and estimated savings
- CSV data export and simulation reset

Educational simulation only. This is not a validated hydraulic design
or a controller for real machinery.
"""

from vpython import (
    canvas, vector, color, box, cylinder, sphere, curve, label,
    rate, slider, button, wtext, graph, gcurve, checkbox
)
import math
import csv
from datetime import datetime

# --------------------------- SIMULATION SETTINGS ---------------------------

DT = 0.04
BASE_DEMAND = 0.58
MIN_SPEED = 0.30
MAX_SPEED = 1.00

MAX_PRESSURE_BAR = 180.0
MAX_FLOW_LPM = 42.0
MOTOR_RATED_KW = 7.5
OIL_HEAT_GAIN = 0.020
COOLING_RATE = 0.004
AMBIENT_TEMPERATURE = 25.0
LOG_INTERVAL = 0.5

# ------------------------------- 3D SCENE ----------------------------------

scene = canvas(
    title="<b>AI-Driven Energy Optimization of Hydraulic Power Units</b>",
    width=1180,
    height=700,
    background=vector(0.035, 0.055, 0.085)
)
scene.center = vector(0, 1.8, 0)
scene.range = 7.3
scene.forward = vector(-1.0, -0.45, -1.0)
scene.caption = (
    "\n<b>3D MODEL CONTROLS</b>  Drag to rotate | Scroll to zoom | Right-drag to pan\n"
    "<i>Educational simulation only — not for real-machine control.</i>\n\n"
)

# Floor and platform
box(pos=vector(0, -0.12, 0), size=vector(12, 0.2, 6),
    color=vector(0.12, 0.16, 0.21))
box(pos=vector(0, 0.02, 0), size=vector(8.4, 0.12, 3.8),
    color=vector(0.27, 0.31, 0.36))

# Reservoir
box(pos=vector(-3.2, 0.72, 0), size=vector(2.0, 1.25, 1.55),
    color=vector(0.34, 0.48, 0.57), opacity=0.72)
box(pos=vector(-3.2, 0.28, 0), size=vector(1.82, 0.12, 1.40),
    color=vector(0.08, 0.36, 0.68), opacity=0.85)
label(pos=vector(-3.2, 1.55, 0), text="OIL RESERVOIR",
      height=13, box=False, color=color.white)

# Motor, shaft and pump
motor = cylinder(pos=vector(-1.55, 0.72, 0), axis=vector(0.95, 0, 0),
                 radius=0.48, color=vector(0.18, 0.42, 0.72))
cylinder(pos=vector(-1.63, 0.72, 0), axis=vector(0.12, 0, 0),
         radius=0.51, color=vector(0.62, 0.68, 0.73))
motor_shaft = cylinder(pos=vector(-0.62, 0.72, 0),
                       axis=vector(0.18, 0, 0), radius=0.11,
                       color=vector(0.82, 0.85, 0.88))
cylinder(pos=vector(-0.55, 0.72, 0), axis=vector(0.62, 0, 0),
         radius=0.36, color=vector(0.78, 0.52, 0.20))
cylinder(pos=vector(-0.02, 0.72, 0), axis=vector(0.12, 0, 0),
         radius=0.40, color=vector(0.9, 0.75, 0.42))
label(pos=vector(-1.15, 1.48, 0), text="ELECTRIC MOTOR",
      height=13, box=False, color=color.white)
label(pos=vector(-0.28, 1.22, 0), text="PUMP",
      height=12, box=False, color=color.white)

# Directional control valve
box(pos=vector(1.15, 0.75, 0), size=vector(1.15, 0.72, 0.92),
    color=vector(0.62, 0.34, 0.13))
for x in (0.82, 1.15, 1.48):
    cylinder(pos=vector(x, 1.13, 0), axis=vector(0, 0.12, 0),
             radius=0.10, color=vector(0.88, 0.82, 0.63))
label(pos=vector(1.15, 1.38, 0), text="DIRECTIONAL VALVE",
      height=12, box=False, color=color.white)

# Hydraulic cylinder and moving rod
cylinder(pos=vector(3.35, 1.35, 0), axis=vector(1.45, 0, 0),
         radius=0.34, color=vector(0.55, 0.60, 0.66))
cylinder(pos=vector(4.75, 1.35, 0), axis=vector(0.18, 0, 0),
         radius=0.39, color=vector(0.78, 0.80, 0.82))
rod = cylinder(pos=vector(4.80, 1.35, 0), axis=vector(0.85, 0, 0),
               radius=0.12, color=vector(0.83, 0.86, 0.90))
piston_head = cylinder(pos=vector(4.65, 1.35, 0), axis=vector(0.12, 0, 0),
                       radius=0.30, color=vector(0.30, 0.34, 0.40))
label(pos=vector(4.0, 2.0, 0), text="HYDRAULIC CYLINDER",
      height=13, box=False, color=color.white)

def pipe(points, pipe_color=vector(0.72, 0.77, 0.83), thickness=0.075):
    return curve(pos=points, radius=thickness, color=pipe_color)

pipe([vector(-3.2, 0.85, 0.75), vector(-1.2, 0.85, 0.75),
      vector(-1.2, 1.18, 0.75), vector(-0.4, 1.18, 0.75)])
pipe([vector(0.1, 0.95, 0.35), vector(0.55, 0.95, 0.35),
      vector(0.55, 1.0, 0.35), vector(1.15, 1.0, 0.35)])
pipe([vector(1.7, 0.95, 0.35), vector(2.35, 0.95, 0.35),
      vector(2.35, 1.35, 0.35), vector(3.35, 1.35, 0.35)])
pipe([vector(1.15, 0.48, -0.35), vector(1.15, 0.25, -0.35),
      vector(-3.2, 0.25, -0.35), vector(-3.2, 0.45, -0.35)],
     pipe_color=vector(0.18, 0.48, 0.78))

# Animated oil-flow particles
flow_path = [
    vector(-3.2, 0.85, 0.75), vector(-1.2, 0.85, 0.75),
    vector(-1.2, 1.18, 0.75), vector(-0.4, 1.18, 0.75),
    vector(0.55, 1.0, 0.35), vector(1.15, 1.0, 0.35),
    vector(2.35, 1.35, 0.35), vector(3.35, 1.35, 0.35)
]
flow_particles = [
    sphere(pos=flow_path[0], radius=0.085,
           color=vector(0.12, 0.72, 1.0), emissive=True)
    for _ in range(12)
]

pressure_tag = label(pos=vector(0.1, 2.1, 0), text="PRESSURE: -- bar",
                     height=13, box=True, color=color.white, opacity=0.18)
flow_tag = label(pos=vector(1.3, 2.1, 0), text="FLOW: -- L/min",
                 height=13, box=True, color=color.white, opacity=0.18)

# ---------------------------- LIVE DASHBOARD -------------------------------

scene.append_to_caption("\n\n<b>LIVE SYSTEM MONITORING</b>\n")
status_text = wtext(text="Status: Initializing...\n")
pressure_text = wtext(text="Pressure: -- bar\n")
flow_text = wtext(text="Flow: -- L/min\n")
power_text = wtext(text="Optimized electrical power: -- kW\n")
baseline_power_text = wtext(text="Baseline electrical power: -- kW\n")
temp_text = wtext(text="Oil temperature: -- °C\n")
speed_text = wtext(text="Motor speed: -- %\n")
saving_text = wtext(text="Estimated energy saving: -- %\n")
prediction_text = wtext(text="Predicted demand: -- %\n")
confidence_text = wtext(text="Prediction confidence indicator: -- %\n")
energy_text = wtext(text="Energy — baseline: -- kWh | optimized: -- kWh\n")

scene.append_to_caption("\n<b>DEMAND AND CONTROL</b>\n")
scene.append_to_caption("Required hydraulic demand: ")
demand_value = BASE_DEMAND
demand_label = wtext(text="58%\n")

def set_demand(widget):
    global demand_value
    demand_value = widget.value
    demand_label.text = str(round(demand_value * 100)) + "%\n"

demand_slider = slider(min=0.15, max=1.0, value=BASE_DEMAND,
                       length=280, bind=set_demand)

auto_enabled = True
scene.append_to_caption("Adaptive motor optimization: ")
auto_label = wtext(text=" ON\n")

def set_auto(widget):
    global auto_enabled
    auto_enabled = widget.checked
    auto_label.text = " ON\n" if auto_enabled else " OFF — fixed-speed baseline\n"

auto_checkbox = checkbox(bind=set_auto, checked=True)

paused = False
def toggle_pause(_):
    global paused
    paused = not paused
    pause_button.text = "Resume simulation" if paused else "Pause simulation"

pause_button = button(text="Pause simulation", bind=toggle_pause)
scene.append_to_caption("   ")
reset_button = None

# ------------------------------- LIVE GRAPHS --------------------------------

scene.append_to_caption("\n\n<b>LIVE TREND GRAPHS</b>\n")
pressure_graph = graph(title="Hydraulic pressure",
                       xtitle="Simulation time (s)", ytitle="bar",
                       width=560, height=230)
pressure_curve = gcurve(graph=pressure_graph, color=color.cyan)

power_graph = graph(title="Electrical power: baseline vs optimized",
                    xtitle="Simulation time (s)", ytitle="kW",
                    width=560, height=230)
optimized_curve = gcurve(graph=power_graph, color=color.orange)
baseline_curve = gcurve(graph=power_graph, color=color.white)

scene.append_to_caption(
    "\n<b>Graph key:</b> <font color='orange'>Optimized power</font>  "
    "<font color='white'>Baseline power</font>\n"
)

# ----------------------- PREDICTION AND STATE --------------------------------

predicted_demand = BASE_DEMAND
prediction_error = 0.0
sim_time = 0.0
oil_temperature = 32.0
cylinder_position = 0.50
energy_baseline = 0.0
energy_optimized = 0.0
total_energy_saved = 0.0
particle_phase = 0.0
log_rows = []
last_log_time = -LOG_INTERVAL
last_values = {
    "pressure": 0.0, "flow": 0.0, "optimized_kw": 0.0,
    "baseline_kw": 0.0, "speed": 0.0, "confidence": 0.0
}

def clamp(value, low, high):
    return max(low, min(high, value))

def interpolate_path(points, t):
    t = t % 1.0
    count = len(points) - 1
    scaled = t * count
    index = min(int(scaled), count - 1)
    local_t = scaled - index
    return points[index] * (1 - local_t) + points[index + 1] * local_t

def reset_simulation(_):
    global sim_time, predicted_demand, prediction_error, oil_temperature
    global cylinder_position, energy_baseline, energy_optimized
    global total_energy_saved, particle_phase, log_rows, last_log_time
    global last_values
    sim_time = 0.0
    predicted_demand = demand_value
    prediction_error = 0.0
    oil_temperature = 32.0
    cylinder_position = 0.50
    energy_baseline = 0.0
    energy_optimized = 0.0
    total_energy_saved = 0.0
    particle_phase = 0.0
    log_rows = []
    last_log_time = -LOG_INTERVAL
    update_simulation.motor_speed = 0.75
    last_values = {
        "pressure": 0.0, "flow": 0.0, "optimized_kw": 0.0,
        "baseline_kw": 0.0, "speed": 0.0, "confidence": 0.0
    }
    status_text.text = "Status: Reset — simulation restarting\n"

reset_button = button(text="Reset simulation", bind=reset_simulation)
scene.append_to_caption("   ")

def export_csv(_):
    filename = "hydraulic_simulation_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "time_s", "demand_percent", "predicted_demand_percent",
            "pressure_bar", "flow_lpm", "motor_speed_percent",
            "optimized_power_kw", "baseline_power_kw",
            "oil_temperature_c", "estimated_energy_saving_percent"
        ])
        writer.writerows(log_rows)
    status_text.text = "Status: CSV exported to " + filename + "\n"

export_button = button(text="Export CSV data", bind=export_csv)
scene.append_to_caption("\n")

# ---------------------------- SIMULATION UPDATE -----------------------------

def update_simulation():
    global sim_time, predicted_demand, prediction_error, oil_temperature
    global cylinder_position, energy_baseline, energy_optimized
    global total_energy_saved, particle_phase, last_log_time

    measured_demand = clamp(
        demand_value + 0.045 * math.sin(sim_time * 0.55)
        + 0.018 * math.sin(sim_time * 1.7), 0.10, 1.0
    )

    # Online exponential moving average predictor (not a trained ML model).
    old_prediction = predicted_demand
    predicted_demand = 0.88 * predicted_demand + 0.12 * measured_demand
    prediction_error = 0.90 * prediction_error + 0.10 * abs(
        measured_demand - old_prediction
    )
    confidence = clamp(1.0 - prediction_error * 2.5, 0.55, 0.99)

    baseline_speed = 0.92
    if auto_enabled:
        target_speed = clamp(0.20 + 0.82 * predicted_demand,
                             MIN_SPEED, MAX_SPEED)
    else:
        target_speed = baseline_speed

    current_speed = getattr(update_simulation, "motor_speed", 0.75)
    current_speed += (target_speed - current_speed) * 0.075
    update_simulation.motor_speed = current_speed

    # Simplified relationships for demonstration.
    flow_lpm = MAX_FLOW_LPM * current_speed
    pressure_bar = clamp(
        35 + 125 * measured_demand / max(current_speed, 0.25),
        20, MAX_PRESSURE_BAR
    )
    hydraulic_kw = (pressure_bar * flow_lpm) / 600.0
    efficiency = 0.82
    optimized_kw = hydraulic_kw / efficiency
    baseline_flow = MAX_FLOW_LPM * baseline_speed
    baseline_kw = ((pressure_bar * baseline_flow) / 600.0) / efficiency
    optimized_kw = min(optimized_kw, MOTOR_RATED_KW * 1.25)
    baseline_kw = min(baseline_kw, MOTOR_RATED_KW * 1.25)

    losses = max(0.0, optimized_kw - hydraulic_kw)
    oil_temperature += (
        losses * OIL_HEAT_GAIN
        - (oil_temperature - AMBIENT_TEMPERATURE) * COOLING_RATE
    )

    energy_optimized += optimized_kw * DT / 3600.0
    energy_baseline += baseline_kw * DT / 3600.0
    if energy_baseline > 0:
        total_energy_saved = (
            (energy_baseline - energy_optimized) / energy_baseline * 100
        )

    # Cylinder stroke follows changing demand.
    target_position = 0.25 + 0.72 * measured_demand
    cylinder_position += (target_position - cylinder_position) * 0.035
    rod.pos = vector(4.80, 1.35, 0)
    rod.axis = vector(0.45 + cylinder_position * 0.8, 0, 0)
    piston_head.pos = vector(4.65 + cylinder_position * 0.25, 1.35, 0)

    # Flow-particle speed follows motor speed.
    particle_phase += DT * (0.12 + current_speed * 0.22)
    for i, particle in enumerate(flow_particles):
        particle.pos = interpolate_path(
            flow_path, particle_phase + i / len(flow_particles)
        )

    pressure_tag.text = "PRESSURE: " + str(round(pressure_bar, 1)) + " bar"
    flow_tag.text = "FLOW: " + str(round(flow_lpm, 1)) + " L/min"
    pressure_text.text = "Pressure: " + str(round(pressure_bar, 1)) + " bar\n"
    flow_text.text = "Flow: " + str(round(flow_lpm, 1)) + " L/min\n"
    power_text.text = "Optimized electrical power: " + str(round(optimized_kw, 2)) + " kW\n"
    baseline_power_text.text = "Baseline electrical power: " + str(round(baseline_kw, 2)) + " kW\n"
    temp_text.text = "Oil temperature: " + str(round(oil_temperature, 1)) + " °C\n"
    speed_text.text = "Motor speed: " + str(round(current_speed * 100)) + " %\n"
    saving_text.text = "Estimated energy saving: " + str(round(total_energy_saved, 1)) + " %\n"
    prediction_text.text = "Predicted demand: " + str(round(predicted_demand * 100)) + " %\n"
    confidence_text.text = "Prediction confidence indicator: " + str(round(confidence * 100)) + " %\n"
    energy_text.text = (
        "Energy — baseline: " + format(energy_baseline, ".6f")
        + " kWh | optimized: " + format(energy_optimized, ".6f") + " kWh\n"
    )

    if pressure_bar >= 155:
        status_text.text = "Status: HIGH PRESSURE INDICATOR — check demand\n"
    elif oil_temperature >= 65:
        status_text.text = "Status: ELEVATED OIL TEMPERATURE INDICATOR\n"
    elif auto_enabled:
        status_text.text = "Status: ADAPTIVE OPTIMIZATION ACTIVE\n"
    else:
        status_text.text = "Status: BASELINE / FIXED-SPEED MODE\n"

    pressure_curve.plot(sim_time, pressure_bar)
    optimized_curve.plot(sim_time, optimized_kw)
    baseline_curve.plot(sim_time, baseline_kw)

    if sim_time - last_log_time >= LOG_INTERVAL:
        log_rows.append([
            round(sim_time, 3), round(measured_demand * 100, 2),
            round(predicted_demand * 100, 2), round(pressure_bar, 2),
            round(flow_lpm, 2), round(current_speed * 100, 2),
            round(optimized_kw, 3), round(baseline_kw, 3),
            round(oil_temperature, 2), round(total_energy_saved, 2)
        ])
        last_log_time = sim_time

    last_values.update({
        "pressure": pressure_bar, "flow": flow_lpm,
        "optimized_kw": optimized_kw, "baseline_kw": baseline_kw,
        "speed": current_speed, "confidence": confidence
    })

# ------------------------------- MAIN LOOP ----------------------------------

print("Hydraulic 3D simulation started.")
print("Use the demand slider, optimization checkbox, reset and CSV export controls.")

while True:
    rate(25)
    if paused:
        continue
    sim_time += DT
    update_simulation()
