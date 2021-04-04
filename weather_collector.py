from datetime import datetime, timedelta
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
        """
        Returns the temperature (°C), humidity (%), pressure (hPa), wind speed (m/s),
        sun exposure (min), rain (mm) and the detailed weather status
        """
        pass


class WeatherCollectorMS(WeatherCollector):
    """
    MS API based implementation of the WeatherCollector interface
    """

    def __init__(self, lat=50.5251, lon=4.6107) -> None:
        from meteostat import Stations

        super().__init__(lat, lon)
        stations = Stations().nearby(lat=lat, lon=lon)
        self.station = stations.fetch(1)
        self.all_data = {}

        self.weather_condition = {
            1: "Clear", 2: "Fair", 3: "Cloudy", 4: "Overcast", 5: "Fog", 6: "Freezing Fog", 7: "Light Rain", 8: "Rain",
            9: "Heavy Rain", 10: "Freezing Rain", 11: "Heavy Freezing Rain", 12: "Sleet", 13: "Heavy Sleet",
            14: "Light Snowfall", 15: "Snowfall", 16: "Heavy Snowfall", 17: "Rain Shower", 18: "Heavy Rain Shower",
            19: "Sleet Shower", 20: "Heavy Sleet Shower", 21: "Snow Shower", 22: "Heavy Snow Shower", 23: "Lightning",
            24: "Hail", 25: "Thunderstorm", 26: "Heavy Thunderstorm", 27: "Storm"
        }

    def get_weather_data(self, dt: datetime) -> (float, float, float, float, str):

        if dt.year not in self.all_data.keys():
            temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year] = {dt.month: {dt.day: {dt.hour: [temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status]}}}
        elif dt.month not in self.all_data[dt.year].keys():
            temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year][dt.month] = {dt.day: {dt.hour: [temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status]}}
        elif dt.day not in self.all_data[dt.year][dt.month].keys():
            temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year][dt.month][dt.day] = {dt.hour: [temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status]}
        elif dt.hour not in self.all_data[dt.year][dt.month][dt.day].keys():
            temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year][dt.month][dt.day][dt.hour] = [temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status]
        else:
            temperature, humidity, pressure, wind, sun_exposure, rain, detailed_status = self.all_data[dt.year][dt.month][dt.day][dt.hour]

        return float(temperature), float(humidity), float(pressure), float(wind), float(sun_exposure), float(rain), str(detailed_status)

    def _get_data(self, dt):
        from meteostat import Hourly

        print("[INFO] Call to MS API")
        dt2 = dt - timedelta(hours=1)
        data = Hourly(self.station, start=dt2, end=dt)
        data = data.fetch()

        row = data.iloc[0]
        return row['temp'], row['rhum'], row['pres'], round(row['wspd'] / 3.6, 3), \
               row['tsun'], row['prcp'], self.weather_condition.setdefault(row['coco'], "Unknown")


class WeatherCollectorOWM(WeatherCollector):
    """
    OWM API based implementation of the WeatherCollector interface
    Warning: limited to 5 days back maximum
    """

    def __init__(self, lat=50.5251, lon=4.6107) -> None:
        super().__init__(lat, lon)
        self.all_data = {}

    def get_weather_data(self, dt: datetime) -> (float, float, float, float, str):

        if dt.year not in self.all_data.keys():
            temperature, pressure, wind, rain, detailed_status = self._get_data(dt)
            self.all_data[dt.year] = {
                dt.month: {dt.day: {dt.hour: [temperature, pressure, wind, rain, detailed_status]}}}
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

        return float(temperature), None, float(pressure), float(wind), None, float(rain), str(detailed_status)

    def _get_data(self, dt):
        import pyowm
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
        return weather.temperature("celsius")['temp'], weather.pressure['press'], weather.wind()[
            'speed'], rain, weather.detailed_status
