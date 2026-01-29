"""
Open Source Data Integration Service.
Fetches real-time environmental data from Open-Meteo (Open Source Weather API).
This allows the Digital Twin to run simulations against REAL operational conditions.
"""

import requests
import json


class OpenMeteoService:
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    # Default: Kolkata
    def get_current_conditions(self, latitude=22.5726, longitude=88.3639):
        """
        Fetches current wind speed and temperature.

        Args:
            latitude (float): Location Lat.
            longitude (float): Location Long.

        Returns:
            dict: {'wind_speed': float, 'temp': float}
        """
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true"
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            weather = data.get('current_weather', {})
            return {
                'wind_speed': weather.get('windspeed', 0.0),  # km/h
                'temp': weather.get('temperature', 25.0),    # Celsius
                'is_real_data': True
            }
        except Exception as e:
            print(f"[WARNING] Open-Meteo Data Fetch Failed: {e}")
            print("[INFO] Falling back to synthetic local data.")
            return {
                'wind_speed': 5.0,  # Fallback safe value
                'temp': 28.0,
                'is_real_data': False
            }


if __name__ == "__main__":
    # Test the service
    service = OpenMeteoService()
    print(json.dumps(service.get_current_conditions(), indent=2))
