import streamlit as st
import pandas as pd
import joblib
import numpy as np

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
    'relative_speed': "Relative speed between the objects involved in the event.",
    'relative_velocity_t': "Tangential component of the relative velocity.",
    'relative_velocity_n': "Normal component of the relative velocity.",
    't_j2k_sma': "Semi-major axis of the object's orbit in the J2000 reference frame.",
    't_j2k_ecc': "Orbital eccentricity of the object in the J2000 reference frame.",
    'c_recommended_od_span': "Recommended operational duration span for the event.",
    'c_actual_od_span': "Actual operational duration span for the event.",
    'c_cd_area_over_mass': "Area-to-mass ratio of the object at the event time.",
    'c_j2k_sma': "Semi-major axis of the object's orbit in the J2000 reference frame (another measurement).",
    'c_j2k_inc': "Inclination of the object's orbit in the J2000 reference frame.",
    'c_cn_r': "A specific parameter related to the object's relative motion.",
    'c_ctdot_n': "Another component of the object's motion at the event time.",
    'c_cndot_r': "Rate of change of a certain object-related parameter.",
    'c_cndot_n': "Another rate of change of a specific object-related parameter.",
    'c_cndot_tdot': "Another rate of change of the object's parameter.",
    't_h_apo': "Apogee height of the object's orbit at the event time.",
    't_h_per': "Perigee height of the object's orbit at the event time.",
    'c_h_per': "A specific parameter for the object's perigee in the event.",
    'geocentric_latitude': "Latitude in the geocentric reference frame.",
    'azimuth': "Azimuthal angle at the event time."
}

st.title("Satellite and Space Object Collision Prediction")
st.write("Fill the following fields or use preloaded values:")

preloaded_values = get_preloaded_values()

inputs = {}
for key, value in preloaded_values.items():
    st.write(f"**{key.replace('_', ' ').capitalize()}**: {descriptions[key]}")

    inputs[key] = st.number_input(f"{key.replace('_', ' ').capitalize()}", value=float(value), format="%f")

    st.write("\n\n")
    st.write("\n\n")
    

# st.write("Input values:", inputs)

if st.button("Predict"):
    input_df = pd.DataFrame([inputs])

    st.write("Input DataFrame for prediction:", input_df)
    
    prediction = regressor_rf.predict(input_df)

    st.write("Prediction Output:", prediction)

    if isinstance(prediction, (list, np.ndarray)):
        st.success(f"Prediction: {prediction[0]}")
    else:
        st.success(f"Prediction: {prediction}")
