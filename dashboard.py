import streamlit as st
import pandas as pd
import time
import altair as alt

# 1. Page Configuration
st.set_page_config(
    page_title="Pipeline Guard Dashboard",
    page_icon="🚰",
    layout="wide"
)

# 2. Initialize Session State for Alerts
if 'alert_triggered' not in st.session_state:
    st.session_state.alert_triggered = False

# 3. Custom Styling
st.markdown("""
<style>
    .stButton>button {
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# 4. Header
st.title("🚰 Pipeline Guard: Real-Time Leak Detection")
st.markdown("Monitoring live sensor data from the urban water distribution network.")
st.divider()

LOG_FILE = 'prediction_log.csv'
dashboard_placeholder = st.empty()

# --- ALERT SCREEN FUNCTION (The Polished Control Room) ---
def show_alert_screen(latest_row):
    """Displays the blocking alert screen when a leak is found."""
    with dashboard_placeholder.container():
        # 1. Critical Header
        st.error(f"### 🚨 CRITICAL ALERT: {latest_row['prediction_status'].upper()}")
        
        # 2. Incident Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Time of Incident", latest_row['time'])
        c2.metric("Severity Class", f"Class {latest_row['prediction_class']}")
        c3.metric("Current Pressure", f"{latest_row['pressure']} Bar")
        c4.metric("Current Flow", f"{latest_row['flow_rate']} L/m")
        
        st.markdown("---")
        
        # 3. Operator Control Panel
        st.subheader("👨‍🔧 Control Room Actions")
        
        # Row 1: Mitigation Actions
        col_a, col_b = st.columns(2)
        
        with col_a:
            if st.button("📉 Throttle Pressure (-50%)", use_container_width=True):
                st.toast("⚠️ Valves throttling to 50%... Pressure reduction initiated.")
                time.sleep(1)
                st.success("Valves set to 50%. Loss rate minimized.")

        with col_b:
            if st.button("🛑 EMERGENCY SHUTOFF", type="primary", use_container_width=True):
                st.toast("⛔ SHUTOFF SIGNAL SENT.")
                st.error("SECTOR ISOLATED. WATER SUPPLY CUT.")
        
        # Row 2: Logistics & ML Feedback
        col_c, col_d = st.columns(2)
        
        with col_c:
            if st.button("🚑 Dispatch Repair Crew", use_container_width=True):
                st.info(f"📍 GPS Coordinates sent to Field Team Alpha. ETA: 15 mins.")
        
        with col_d:
            if st.button("🤖 False Alarm (Flag for Retraining)", use_container_width=True):
                st.session_state.alert_triggered = False
                st.toast("Data point flagged for model retraining.", icon="🧠")
                time.sleep(1)
                st.rerun()

        # 4. Operator Compliance Log
        st.markdown("---")
        with st.expander("📝 Operator Incident Log (Required for Resolution)", expanded=True):
            notes = st.text_area("Enter details to resolve:", placeholder="E.g., Pipe burst confirmed at Sector 4. Crew dispatched.")
            
            if st.button("✅ Mark Issue Resolved & Resume Monitoring", type="secondary", use_container_width=True):
                if notes:
                    st.success("Incident logged. Resuming monitoring...")
                    st.session_state.alert_triggered = False
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("⚠️ Protocol requires incident notes before resolving.")

        st.warning("⚠️ System Halted: Active operator input required.")

# --- MAIN LOOP ---
while True:
    # A. Alert Mode: If triggered, stop the loop and show the alert screen
    if st.session_state.alert_triggered:
        try:
            df = pd.read_csv(LOG_FILE)
            latest = df.iloc[-1]
            show_alert_screen(latest)
            st.stop() # Stop execution here to wait for user interaction
        except Exception:
            pass

    # B. Normal Monitoring Mode
    try:
        df = pd.read_csv(LOG_FILE)
        
        if not df.empty:
            with dashboard_placeholder.container():
                latest = df.iloc[-1]
                previous = df.iloc[-2] if len(df) > 1 else latest

                # 1. CHECK FOR LEAK TRIGGER
                if latest['prediction_class'] != 0:
                    st.session_state.alert_triggered = True
                    st.rerun()

                # 2. STATUS BANNER
                st.success(f"## ✅ System Normal: No Leaks Detected")

                # 3. METRICS
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pressure (Bar)", f"{latest['pressure']:.2f}", f"{latest['pressure'] - previous['pressure']:.2f}")
                with col2:
                    st.metric("Flow Rate (L/m)", f"{latest['flow_rate']:.2f}", f"{latest['flow_rate'] - previous['flow_rate']:.2f}")
                with col3:
                    st.metric("System Status", "Secure", delta_color="off")

                # 4. CHARTS
                st.divider()
                st.subheader("Live Sensor Trends")
                
                # FIX: .reset_index() ensures Altair can see the index column
                chart_data = df.tail(50).reset_index()

                pressure_chart = alt.Chart(chart_data).mark_line(color='#1f77b4').encode(
                    x=alt.X('index', title='Time Step'),
                    y=alt.Y('pressure', title='Pressure', scale=alt.Scale(zero=False)),
                    tooltip=['time', 'pressure']
                ).properties(title="Pressure Sensor", height=300)

                flow_chart = alt.Chart(chart_data).mark_line(color='#ff7f0e').encode(
                    x=alt.X('index', title='Time Step'),
                    y=alt.Y('flow_rate', title='Flow Rate', scale=alt.Scale(zero=False)),
                    tooltip=['time', 'flow_rate']
                ).properties(title="Flow Rate Sensor", height=300)

                c1, c2 = st.columns(2)
                c1.altair_chart(pressure_chart, use_container_width=True)
                c2.altair_chart(flow_chart, use_container_width=True)

        else:
            st.info("Waiting for data stream...")

    except FileNotFoundError:
        st.warning("Log file not found. Please start 'simulate.py'!")
    except Exception as e:
        st.error(f"Error: {e}")
    
    time.sleep(1)