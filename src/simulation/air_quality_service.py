"""
Open Source Air Quality Integration.
Fetches real-time pollution data from OpenAQ.
Used to identify high-risk industrial zones for drone deployment.
"""

import requests


class OpenAQService:
    def __init__(self):
        self.base_url = "https://api.openaq.org/v2/latest"

    def get_local_pollution(self, city="Kolkata"):
        """
        Fetches latest PM2.5 and PM10 readings for a city.

        Args:
            city (str): Name of the city (e.g., 'Kolkata', 'Delhi').

        Returns:
            list: List of dictionaries containing location and measurement data.
        """
        try:
            params = {
                "city": city,
                "parameter": ["pm25", "pm10"],
                "limit": 3
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            processed_data = []

            for res in results:
                location = res.get('location')
                measurements = res.get('measurements', [])
                for m in measurements:
                    processed_data.append({
                        'location': location,
                        'parameter': m.get('parameter'),
                        'value': m.get('value'),
                        'unit': m.get('unit')
                    })

            return processed_data

        except Exception as e:
            print(f"[WARNING] OpenAQ Fetch Failed: {e}")
            return []


if __name__ == "__main__":
    service = OpenAQService()
    data = service.get_local_pollution()
    for entry in data:
        print(
            f"Location: {entry['location']} | {entry['parameter'].upper()}: {entry['value']} {entry['unit']}")
