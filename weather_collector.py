import pyowm
from datetime import datetime
import dateutil.parser


def _get_nearest(data, dt):
    reference_times = []
    for elem in data:
        reference_times.append(dateutil.parser.parse(elem.reference_time('iso')))

    idx = 0
    for ref in reference_times:
        if ref.time().hour == dt.time().hour:
            return data[idx]
        idx += 1


class WeatherCollector:

    def __init__(self, lat: float, lon: float) -> None:
        self.lat = lat
        self.lon = lon

    def get_weather_data(self, dt: datetime) -> (float, float, float, float, str):
        """Returns the temperature (°C), pressure (hPa), wind speed (m/s), rain (mm) and the detailed weather status"""
        pass


class WeatherCollectorOWM(WeatherCollector):

    def __init__(self, lat=50.5251, lon=4.6107) -> None:
        super().__init__(lat, lon)
        self.all_data = {}

    def get_weather_data(self, dt: datetime) -> (float, float, float, float, str):

        if dt.year not in self.all_data.keys():
            temperature, pressure, wind, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year] = {dt.month: {dt.day: {dt.hour: [temperature, pressure, wind, rain, detailed_status]}}}
        elif dt.month not in self.all_data[dt.year].keys():
            temperature, pressure, wind, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year][dt.month] = {dt.day: {dt.hour: [temperature, pressure, wind, rain, detailed_status]}}
        elif dt.day not in self.all_data[dt.year][dt.month].keys():
            temperature, pressure, wind, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year][dt.month][dt.day] = {dt.hour: [temperature, pressure, wind, rain, detailed_status]}
        elif dt.hour not in self.all_data[dt.year][dt.month][dt.day].keys():
            temperature, pressure, wind, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year][dt.month][dt.day][dt.hour] = [temperature, pressure, wind, rain, detailed_status]
        else:
            temperature, pressure, wind, rain, detailed_status = self.all_data[dt.year][dt.month][dt.day][dt.hour]

        return float(temperature), float(pressure), float(wind), float(rain), str(detailed_status)

    def _get_data(self, dt):
        with open("./resources/api-keys/openweathermap/APIKEY.txt", "r") as f:
            api_key = f.read()

        owm = pyowm.OWM(api_key)
        mgr = owm.weather_manager()

        print("[INFO] Call to OWM API")
        weather_data = mgr.one_call_history(lat=self.lat, lon=self.lon, dt=int(dt.timestamp())).forecast_hourly
        weather = _get_nearest(weather_data, dt)

        if "1h" in weather.rain.keys():
            rain = weather.rain["1h"]
        else:
            rain = 0
        return weather.temperature("celsius")['temp'], weather.pressure['press'], weather.wind()['speed'], rain, weather.detailed_status
