import argparse
import pandas as pd
import pathlib
import numpy as np
from skimage import color, io
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
import json


def warp(img, initial_coords, final_coords):
    """
    initial and final coords are a list of tuples with x and y values.
    """
    from skimage import transform
    import numpy as np

    # for_transform = transform.estimate_transform(
    #     "projective", np.array(initial_coords), np.array(final_coords)
    # )
    # transformed = transform.warp(img, for_transform.inverse)

    # from https://scikit-image.org/docs/stable/auto_examples/transform/plot_geometric.html#sphx-glr-auto-examples-transform-plot-geometric-py
    tform3 = transform.ProjectiveTransform()
    tform3.estimate(final_coords, initial_coords)
    warped = transform.warp(
        img,
        tform3,
        # output_shape=(300, 300),
    )

    return warped


def warp_squareplate(img, initial_coords, target=(150, 150)):
    """
    Calculate final coords based on small plate dimensions. Incorporate a crop to get rid of the plate edges.
    """
    imdims = (img.shape[1], img.shape[0])

    final_coords = np.array(
        [
            list(target),
            (imdims[0] - target[0], target[1]),
            (target[0], imdims[1] - target[1]),
            (imdims[0] - target[0], imdims[1] - target[1]),
            ((imdims[0] - target[0]) / 2, (imdims[1] - target[1]) / 2),
        ]
    )

    return warp(img, initial_coords, final_coords)


def flatten(img, sigma):
    from skimage.filters import gaussian

    return img - gaussian(img, sigma=sigma)


def draw_colonies(cx: list, cy: list, r: list, image: np.array):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = np.int16(
        np.interp(image, (image.min(), image.max()), (0, 255))
    )  # this rescales the image to be between 0 and 255
    cimage = color.gray2rgb(image)
    for center_y, center_x, rad in zip(cy, cx, r):
        # to make them a bit bolder
        for dr in range(rad, rad + 3):
            circy, circx = circle_perimeter(center_y, center_x, dr, shape=cimage.shape)
            cimage[circy, circx] = (220, 20, 20)

    ax.imshow(cimage)
    plt.show()


def find_colonies(plate_processed, desired_count=192, colony_sizerange=(10, 50)):
    image = plate_processed

    edges = canny(image)

    hough_radii = np.arange(
        colony_sizerange[0], colony_sizerange[1], 1
    )  # the ranges of radii to search for circles
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circles
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res,
        hough_radii,
        min_xdistance=colony_sizerange[1],
        min_ydistance=colony_sizerange[1],
        total_num_peaks=desired_count,
        normalize=True,
    )

    # print(len(accums), len(cx), len(cy), len(radii))

    return accums, cx, cy, radii


def main():

    parser = argparse.ArgumentParser(
        description="""
        Takes image and plate coordinates as input. Outputs locations of colonies relative to the upper left coordinate given.
        """
    )

    parser.add_argument("imagepath", type=str, help="Path to the image file.")
    parser.add_argument("json", type=str, help="Path to the config file.")

    args = vars(parser.parse_args())
    path = pathlib.Path(args["imagepath"])
    with open(pathlib.Path(args["json"])) as f:
        config = json.load(f)

    img = io.imread(path)
    img = np.asarray(img)[:, :, :3]  # for color images TODO: check if color or B&W
    # plate = np.asarray(img)  # for B&W images
    plate = color.rgb2gray(img)  # for color images
    # crop plate to the shortest dimension
    cropped = plate[
        : min(plate.shape[0], plate.shape[1]), : min(plate.shape[0], plate.shape[1])
    ]
    flattened = flatten(
        cropped, 0.6 * cropped.shape[0]
    )  # the size of the gaussian here should work generally.
    warped = warp_squareplate(
        flattened,
        np.array(
            [
                (config["point1"][0], config["point1"][1]),
                (config["point2"][0], config["point2"][1]),
                (config["point3"][0], config["point3"][1]),
                (config["point4"][0], config["point4"][1]),
                (config["point5"][0], config["point5"][1]),
            ]
        ),
        target=(config["crop target"][0], config["crop target"][1]),
    )

    accums, cx, cy, r = find_colonies(
        warped, config["num colonies"], config["colony sizerange"]
    )
    draw_colonies(cx, cy, r, warped)

    df = pd.DataFrame(
        {
            "x coord": cx,
            "y coord": cy,
            "quality": accums,
        }
    )

    # Sort the quality column in descending order and retain the top 50 colonies
    # df = df.sort_values(by="quality", ascending=False).head(50)

    # Filter by quality. Try 0.5 as a cutoff? TODO or should sort by quality and take the top x number?
    df = df[df["quality"] > 0.5]

    mm_conversion_factor = config["target width"] / (
        warped.shape[1] - (2 * config["crop target"][0])
    )

    # find the center of the warped image
    center_x = warped.shape[1] / 2
    center_y = warped.shape[0] / 2

    # calculate the distance from the center in mm
    df["x mm"] = (df["x coord"] - center_x) * mm_conversion_factor
    df["y mm"] = (df["y coord"] - center_y) * mm_conversion_factor * -1

    # df["x mm"] = df["x coord"] * mm_conversion_factor
    # df["y mm"] = df["y coord"] * mm_conversion_factor

    # The OT2 will take positions as a percentage of distance from the center of the labware.
    # Our square plates are 90 mm square (TODO check this). So add a final two columns to calc
    # the percent distance from the center of the plate.

    df["x%"] = df["x mm"] / (config["plate width"] / 2)
    df["y%"] = df["y mm"] / (config["plate width"] / 2)

    print("Found {} colonies.".format(df.shape[0]))
    print("Writing to ", path.with_suffix(".csv"))

    df.to_csv(path.with_suffix(".csv"))


if __name__ == "__main__":
    exit(main())
