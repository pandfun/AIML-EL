import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your model
regressor_rf = joblib.load('random_forest_model.pkl')

def get_preloaded_values():
    return {
        'event_id': 0,
        'relative_speed': 13792.0,
        'relative_velocity_t': -12637.0,
        'relative_velocity_n': -5525.9,
        't_j2k_sma': 6996.918867,
        't_j2k_ecc': 0.003996555165,
        'c_recommended_od_span': 15.85,
        'c_actual_od_span': 15.85,
        'c_cd_area_over_mass': 0.348701,
        'c_j2k_sma': 7006.60732,
        'c_j2k_inc': 74.04573457,
        'c_cn_r': 0.4739756563,
        'c_ctdot_n': -0.8142492109,
        'c_cndot_r': 0.2498551603,
        'c_cndot_n': 0.7221862484,
        'c_cndot_tdot': -0.6684865872,
        't_h_apo': 646.7454388,
        't_h_per': 590.8182944,
        'c_h_per': 606.4433894,
        'geocentric_latitude': -73.57409487,
        'azimuth': -23.61876887,
    }

descriptions = {
    'event_id': "Unique identifier for the event.",
    'relative_speed': "Relative speed between the objects involved in the event (m/s).",
    'relative_velocity_t': "Tangential component of the relative velocity (m/s).",
    'relative_velocity_n': "Normal component of the relative velocity (m/s).",
    't_j2k_sma': "Semi-major axis of the object's orbit in the J2000 reference frame (km).",
    't_j2k_ecc': "Orbital eccentricity of the object in the J2000 reference frame.",
    'c_recommended_od_span': "Recommended operational duration span for the event (minutes).",
    'c_actual_od_span': "Actual operational duration span for the event (minutes).",
    'c_cd_area_over_mass': "Area-to-mass ratio of the object at the event time (m²/kg).",
    'c_j2k_sma': "Semi-major axis of the object's orbit in the J2000 reference frame (km).",
    'c_j2k_inc': "Inclination of the object's orbit in the J2000 reference frame (degrees).",
    'c_cn_r': "A specific parameter related to the object's relative motion.",
    'c_ctdot_n': "Another component of the object's motion at the event time.",
    'c_cndot_r': "Rate of change of a certain object-related parameter.",
    'c_cndot_n': "Another rate of change of a specific object-related parameter.",
    'c_cndot_tdot': "Another rate of change of the object's parameter.",
    't_h_apo': "Apogee height of the object's orbit at the event time (km).",
    't_h_per': "Perigee height of the object's orbit at the event time (km).",
    'c_h_per': "A specific parameter for the object's perigee in the event (km).",
    'geocentric_latitude': "Latitude in the geocentric reference frame (degrees).",
    'azimuth': "Azimuthal angle at the event time (degrees)."
}

# Define limits for each input field
limits = {
    'event_id': {'min': 0, 'max': 1000000, 'step': 1, 'format': '%d'},
    'relative_speed': {'min': 0.0, 'max': 30000.0},
    'relative_velocity_t': {'min': -30000.0, 'max': 30000.0},
    'relative_velocity_n': {'min': -30000.0, 'max': 30000.0},
    't_j2k_sma': {'min': 6000.0, 'max': 45000.0},
    't_j2k_ecc': {'min': 0.0, 'max': 1.0},
    'c_recommended_od_span': {'min': 0.0, 'max': 60.0},
    'c_actual_od_span': {'min': 0.0, 'max': 60.0},
    'c_cd_area_over_mass': {'min': 0.0, 'max': 10.0},
    'c_j2k_sma': {'min': 6000.0, 'max': 45000.0},
    'c_j2k_inc': {'min': 0.0, 'max': 180.0},
    'c_cn_r': {'min': -1.0, 'max': 1.0},
    'c_ctdot_n': {'min': -1.0, 'max': 1.0},
    'c_cndot_r': {'min': -1.0, 'max': 1.0},
    'c_cndot_n': {'min': -1.0, 'max': 1.0},
    'c_cndot_tdot': {'min': -1.0, 'max': 1.0},
    't_h_apo': {'min': 0.0, 'max': 4000.0},
    't_h_per': {'min': 0.0, 'max': 4000.0},
    'c_h_per': {'min': 0.0, 'max': 4000.0},
    'geocentric_latitude': {'min': -90.0, 'max': 90.0},
    'azimuth': {'min': -180.0, 'max': 180.0},
}

st.title("Satellite and Space Object Collision Prediction")
st.write("Fill the following fields or use preloaded values:")

# Optional: Display a table of all input constraints
if st.checkbox("Show Input Constraints"):
    constraints_data = []
    for key, lim in limits.items():
        constraints_data.append({
            "Attribute": key,
            "Description": descriptions.get(key, ""),
            "Min": lim.get('min'),
            "Max": lim.get('max')
        })
    st.table(pd.DataFrame(constraints_data))

preloaded_values = get_preloaded_values()
inputs = {}

for key, value in preloaded_values.items():
    # Add allowed range to the description text
    lim = limits.get(key, {})
    range_text = f" (Allowed range: {lim.get('min')} to {lim.get('max')})" if lim else ""
    st.write(f"**{key.replace('_', ' ').capitalize()}**: {descriptions[key]}{range_text}")
    
    # Input field with limits applied
    if key == 'event_id':
        inputs[key] = st.number_input(
            f"{key.replace('_', ' ').capitalize()}",
            min_value=lim.get('min', 0),
            max_value=lim.get('max', 1000000),
            value=int(value),
            step=lim.get('step', 1),
            format=lim.get('format', '%d')
        )
    else:
        inputs[key] = st.number_input(
            f"{key.replace('_', ' ').capitalize()}",
            min_value=lim.get('min', float(value)),
            max_value=lim.get('max', float(value)),
            value=float(value),
            format="%f"
        )
    st.write("\n\n")

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    st.write("Input DataFrame for prediction:", input_df)
    
    prediction = regressor_rf.predict(input_df)
    predicted_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction

    predicted_value = abs(predicted_value)

    st.success(f"Predicted Collision Risk Score: {predicted_value:.4f}")
    st.info(
        "This score represents the estimated risk or severity of a potential satellite collision. "
        "Higher values indicate a higher risk or more severe potential collision event. "
    )
