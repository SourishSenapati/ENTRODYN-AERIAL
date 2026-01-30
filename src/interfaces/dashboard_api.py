"""
Entrodyn Sentinel Dashboard API.
Exposes swarm health, live gas data, and predictive maintenance metrics to the frontend.
"""

from flask import Flask, jsonify
import json
import os
import time
import random

# Initialize Flask App
app = Flask(__name__)

# Mock Data Store (Replace with SQLite/Redis in production)
DATA_STORE = {
    "swarm_health": "OPTIMAL",
    "active_drones": 6,
    "last_consensus": time.time(),
    "gas_readings": []
}


@app.route('/api/status', methods=['GET'])
def get_system_status():
    """
    Returns the real-time health of the swarm.
    Used by the 'Sentinel' Command Center.
    """
    # Simulate partial fleet degradation
    health = "OPTIMAL"
    if random.random() < 0.1:
        health = "MAINTENANCE_REQUIRED"

    return jsonify({
        "status": health,
        "timestamp": time.time(),
        "active_nodes": 6,
        "consensus_protocol": "PBFT_ACTIVE",
        "reliability_score": "99.9999%"
    })


@app.route('/api/auditor/state', methods=['GET'])
def get_auditor_state():
    """
    Fetches the persisted state of the DeepAuditor.
    Allows the dashboard to see if sensors are saturated.
    """
    state_file = "auditor_state.json"
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
                return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"status": "NO_STATE_FOUND"})


@app.route('/api/predictive/corrosion', methods=['GET'])
def get_corrosion_forecast():
    """
    Predictive Maintenance Endpoint.
    Uses (Simulated) KAN Model inference to predict pipe failure.
    """
    # In live system, this would call the KANSplineLayer on historical visual data

    return jsonify({
        "sector": "Pipeline_Section_4B",
        "current_rust_level": "MODERATE",
        "predicted_failure_days": 14,
        "recommendation": "SCHEDULE_INSPECTION_PRIORITY_HIGH"
    })


if __name__ == '__main__':
    print("Starting Entrodyn Sentinel API...")
    app.run(port=5000, debug=True)
