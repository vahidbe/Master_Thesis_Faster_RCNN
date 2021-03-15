import csv
import os
from weather_collector import WeatherCollectorOWM
import datetime


def update_meteorological_data(model_name, input_results_filename, lat=50.5251, lon=4.6107):
    output_results_filename = "./results/{}".format(model_name)
    if not os.path.exists(output_results_filename):
        os.mkdir(output_results_filename)

    input_record_path = os.path.join(output_results_filename,
                                     "{}.csv".format(input_results_filename))
    output_record_path = os.path.join(output_results_filename,
                                      "detections.csv")
    fieldnames = ['date',
                  'class', 'probability',
                  'x1', 'y1', 'x2', 'y2',
                  'temperature', 'pressure', 'wind', 'rain', 'weather description']
    with open(input_record_path, 'r', newline='') as fr:
        reader = csv.DictReader(fr, fieldnames=fieldnames)

        weather_collector = WeatherCollectorOWM(lat, lon)
        row = next(reader)
        try:
            print("=== Parsing {}".format(input_results_filename))
            while True:
                if row["temperature"] == "empty":
                    dt = datetime.datetime.strptime(row["date"], '%d-%m-%Y_%H-%M-%S')

                    temperature, pressure, wind, rain, detailed_status = weather_collector.get_weather_data(dt)
                    row["temperature"] = temperature
                    row["pressure"] = pressure
                    row["wind"] = wind
                    row["rain"] = rain
                    row["weather description"] = detailed_status

                with open(output_record_path, 'a', newline='') as fw:
                    writer = csv.DictWriter(fw, fieldnames=fieldnames)
                    writer.writerow(row)

                row = next(reader)
        except StopIteration:
            print("=== Parsing completed, exiting.")


if __name__ == "__main__":
    import argparse
    import csv

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Add meteorological data from detection of the last 5 days.')
    parser.add_argument('--model_name', required=False,
                        metavar="name_of_your_model", default='model',
                        help='Name of the model that produced detection')
    parser.add_argument('--lat', required=False, default="50.5251",
                        metavar="Latitude of the detections",
                        help="Latitude of the area where the detections were made")
    parser.add_argument('--lon', required=False, default="4.6107",
                        metavar="Longitude of the detections",
                        help="Longitude of the area where the detections were made")
    parser.add_argument('--input_file', required=True,
                        metavar="name_of_input_detection_file.csv",
                        help="Name of the raw detection CSV file produced during the detection")
    args = parser.parse_args()

    update_meteorological_data(args.model_name, args.input_file, float(args.lat), float(args.lon))
