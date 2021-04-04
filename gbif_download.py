import requests
import os
import json


if __name__ == "__main__":

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Download images from the links in the gbif file.')
    parser.add_argument('--filename', required=True,
                        metavar="name_of_the_gbif_file",
                        help='Name of the gbif containing the links')
    parser.add_argument('--class_name', required=True,
                        metavar="class_name",
                        help='Name of the class to be downloaded')
    parser.add_argument('--output_dir', required=False,
                        metavar="output_directory_path",
                        help='Path to the output directory')
    parser.add_argument('--n_images', required=False, default=float("inf"),
                        metavar="number of images to download",
                        help='Number of images to download')
    parser.add_argument('--from_image', required=False, default="0",
                        metavar="image to start from",
                        help='Image to start from')
    args = parser.parse_args()

    n_images = int(args.n_images)
    from_image = int(args.from_image)

    class_name = args.class_name
    input_filename = args.filename
    output_path = args.output_dir

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    count = 0
    generic_name = "{}{:04d}.jpg"
    with open(input_filename, 'r', encoding="utf8") as fr:
        fr.readline()
        for line in fr:
            if count < from_image:
                count += 1
                continue

            if count >= n_images:
                break

            elements = line.split("\t")
            link = elements[3]
            print(link)
            response = requests.get(link)
            with open(os.path.join(output_path, generic_name.format(class_name, count)), 'wb') as fw:
                fw.write(response.content)
            count += 1
