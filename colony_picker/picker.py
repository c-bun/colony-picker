import argparse
import pandas as pd
import pathlib
import numpy as np
from skimage import color, io
from skimage.transform import hough_circle, hough_circle_peaks, rotate
from skimage.feature import canny
from skimage.draw import circle_perimeter, disk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import json
from typing import List


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
            # ((imdims[0] - target[0]) / 2, (imdims[1] - target[1]) / 2), # don't use with ansi plate
        ]
    )

    return warp(img, initial_coords, final_coords)


def warp_rectangleplate(img, initial_coords):
    """
    Calculate final coords based on rectangle plate dimensions.
    """

    # these are from the illustration of the plate
    final_coords_mm = np.array(
        [15.1, 10.5],
        [29.9, 10.5],
        [59.6, 10.5],
        [89.2, 10.5],
        [104.0, 10.5],
        [15.1, 20.8],
        [29.9, 20.8],
        [59.6, 20.8],
        [89.2, 20.8],
        [104.0, 20.8],
        [15.1, 41.3],
        [29.9, 41.3],
        [59.6, 41.3],
        [89.2, 41.3],
        [104.0, 41.3],
        [15.1, 61.8],
        [29.9, 61.8],
        [59.6, 61.8],
        [89.2, 61.8],
        [104.0, 61.8],
        [15.1, 72.1],
        [29.9, 72.1],
        [59.6, 72.1],
        [89.2, 72.1],
        [104.0, 72.1],
    )

    # Calulate the distance between the first and fifth corrdinates
    # This is the width of the plate
    width = initial_coords[0][0] - initial_coords[4][0]
    mm_width = final_coords_mm[0][0] - final_coords_mm[4][0]
    scale_factor = width / mm_width

    final_coords = final_coords_mm * scale_factor

    return warp(img, initial_coords, final_coords)


def flatten(img, sigma):
    from skimage.filters import gaussian

    return img - gaussian(img, sigma=sigma)


def draw_colonies(cx: list, cy: list, r: list, image: np.array):

    image = np.int16(
        np.interp(image, (image.min(), image.max()), (0, 255))
    )  # this rescales the image to be between 0 and 255
    cimage = color.gray2rgb(image)
    for center_y, center_x, rad in zip(cy, cx, r):
        # to make them a bit bolder
        for dr in range(rad, rad + 3):
            circy, circx = circle_perimeter(center_y, center_x, dr, shape=cimage.shape)
            cimage[circy, circx] = (220, 20, 20)

    return cimage


# Filter out colonies that are too close to each other
def filter_colonies(accums: list, cx: list, cy: list, radii: list, min_separation: int):
    # Make a copy of the colonies
    accums_copy = accums.copy()
    cx_copy = cx.copy()
    cy_copy = cy.copy()
    radii_copy = radii.copy()
    # Loops through the list of colonies
    for cx1, cy1 in zip(cx, cy):
        # Loops through the list of colonies again
        for i, (accums2, cx2, cy2, radii2) in enumerate(zip(accums, cx, cy, radii)):
            # Calculate the distance between the two colonies
            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            # If the distance is less than the minimum separation, tag the second colony for removal
            if (distance < min_separation) and (distance != 0):
                cx_copy[i] = -1
                cy_copy[i] = -1
                radii_copy[i] = -1
                accums_copy[i] = -1
                # break

    # Remove the removed colonies from the list
    cx_copy = [x for x in cx_copy if x != -1]
    cy_copy = [x for x in cy_copy if x != -1]
    radii_copy = [x for x in radii_copy if x != -1]
    accums_copy = [x for x in accums_copy if x != -1]

    return accums_copy, cx_copy, cy_copy, radii_copy


# Sort the colonies by accums value and return the top N colonies
def get_top_colonies(accums, cx, cy, radii, count):
    # Sort the colonies by accums value
    accums, cx, cy, radii = zip(*sorted(zip(accums, cx, cy, radii), reverse=True))
    # Return the top N colonies
    return list(accums[:count]), list(cx[:count]), list(cy[:count]), list(radii[:count])


def draw_gui(
    image, image_path, default_sizerange=(5, 15), default_count=100, min_separation=10
):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    fig.subplots_adjust(left=0.25, bottom=0.35)
    plt.title(str(image_path.name))

    global accums, cx, cy, radii

    accums, cx, cy, radii = find_colonies(
        image,
        # desired_count=default_count,
        colony_sizerange=default_sizerange,
        # min_separation=min_separation,
    )
    ax_image = ax.imshow(draw_colonies(cx, cy, radii, image))

    def draw_colony_count(ax, count):
        # Add text to display the number of colonies found
        ax.text(
            0.5,
            0.9,
            "Found {} colonies".format(count),
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="white",
        )

    draw_colony_count(ax, len(cx))

    # Add two sliders for tweaking the parameters
    # Define an axes area and draw a slider in it
    min_slider_ax = fig.add_axes(
        [0.25, 0.3, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )
    min_slider = Slider(
        min_slider_ax, "Min Size", 1, 20, valinit=default_sizerange[0], valstep=1
    )
    # Draw another slider
    max_slider_ax = fig.add_axes(
        [0.25, 0.25, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )
    max_slider = Slider(
        max_slider_ax, "Max Size", 1, 20, valinit=default_sizerange[1], valstep=1
    )
    # Draw another slider to define the minimum separation between colonies
    min_separation_ax = fig.add_axes(
        [0.25, 0.2, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )
    min_separation_slider = Slider(
        min_separation_ax,
        "Min Separation",
        5,
        50,
        valinit=min_separation,
        valstep=1,
    )
    # Draw one last slider to define the number of colonies to find
    count_slider_ax = fig.add_axes(
        [0.25, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow"
    )
    count_slider = Slider(
        count_slider_ax, "Count", 10, 500, valinit=default_count, valstep=1
    )

    # Draw a button to close the application
    close_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    close_button = Button(close_ax, "Save and close")

    # Define an action for the button to close the application
    def close_button_on_clicked(event):
        plt.close()

    close_button.on_clicked(close_button_on_clicked)

    # Define an action for modifying the image when any slider's value changes
    def sliders_on_changed(val):
        global accums, cx, cy, radii
        accums, cx, cy, radii = find_colonies(
            image,
            # desired_count=count_slider.val,
            colony_sizerange=(min_slider.val, max_slider.val),
            # min_separation=min_separation_slider.val,
        )
        # Filter out colonies that are too close to each other
        accums, cx, cy, radii = filter_colonies(
            accums, cx, cy, radii, min_separation_slider.val
        )
        # Sort the colonies by accums value and return the top N colonies
        accums, cx, cy, radii = get_top_colonies(
            accums, cx, cy, radii, count_slider.val
        )
        ax_image.set(data=draw_colonies(cx, cy, radii, image))
        # First, remove the old text
        ax.texts.pop()
        draw_colony_count(ax, len(cx))
        fig.canvas.draw_idle()

    min_slider.on_changed(sliders_on_changed)
    max_slider.on_changed(sliders_on_changed)
    count_slider.on_changed(sliders_on_changed)
    min_separation_slider.on_changed(sliders_on_changed)

    plt.show()

    return accums, cx, cy


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
        min_xdistance=colony_sizerange[0],
        min_ydistance=colony_sizerange[0],
        total_num_peaks=500,
        normalize=True,
    )

    # print(len(accums), len(cx), len(cy), len(radii))

    return list(accums), list(cx), list(cy), list(radii)


def find_circle_pixels(cx, cy, radii, image, radius_offset=1) -> list((int, int)):
    """
    Find the pixels that are contained within each circle.
    """
    circle_pixels = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = disk(
            center_y, center_x, radius - radius_offset, shape=image.shape
        )
        circle_pixels.append((circy, circx))
    return circle_pixels


def find_luminescence(image: np.array, cx: list, cy: list, radii: list) -> list:
    """
    Given a list of colony positions and radii, find the luminescence of each colony.
    """
    luminescence = []
    for x, y, r in zip(cx, cy, radii):
        luminescence.append(
            np.mean(image[find_circle_pixels(x, y, r, image)])
        )  # average the luminescence of the colony
    return luminescence


def main():

    parser = argparse.ArgumentParser(
        description="""
        Takes directory of images and plate coordinates as input. Outputs locations of colonies relative to the upper left coordinate given.
        """
    )

    parser.add_argument("dirpath", type=str, help="Path to the dir of image files.")
    parser.add_argument("json", type=str, help="Path to the config file.")

    args = vars(parser.parse_args())
    path = pathlib.Path(args["dirpath"])
    with open(pathlib.Path(args["json"])) as f:
        config = json.load(f)

    dfs = []  # list of dataframes to be concatenated

    # for each image in the directory, find the colonies
    for image_path in path.glob("*.png"):

        img = io.imread(image_path)
        img = np.asarray(img)[:, :, :3]  # for color images
        plate = color.rgb2gray(img)  # for color images

        # rotate the plate image TODO is this necessary?
        # rotated = rotate(plate, config["rotation"])

        # these are from the illustration of the plate
        final_coords_mm = np.array(config["reference coords"])
        # these are from the pixel positions on the plate
        initial_coords = np.array(config["image coords"])

        # Calulate the distance between the first and fifth corrdinates
        # This is the width of the plate
        width = initial_coords[0][0] - initial_coords[4][0]
        mm_width = final_coords_mm[0][0] - final_coords_mm[4][0]
        scale_factor = width / mm_width

        final_coords = final_coords_mm * scale_factor

        warped = warp(plate, initial_coords, final_coords)

        # crop out the edges of the plate
        cropped = warped[
            config["crop target"][1] : int(
                config["plate dimensions"][1] * scale_factor - config["crop target"][1]
            ),
            config["crop target"][0] : int(
                config["plate dimensions"][0] * scale_factor - config["crop target"][0]
            ),
        ]

        flattened = flatten(
            cropped, 0.6 * cropped.shape[0]
        )  # the size of the gaussian here should work generally.

        accums, cx, cy = draw_gui(
            flattened,
            image_path,
            default_sizerange=config["colony sizerange"],
            default_count=config["colony count"],
            min_separation=config["min separation"],
        )

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
        if not config["testing"]:
            df = df[df["quality"] > 0.5]

        # find the center of the cropped image
        center_x = flattened.shape[1] / 2
        center_y = flattened.shape[0] / 2

        # calculate the distance from the center in mm
        # calibration offse is determined through testing on the robot.
        df["x mm"] = ((df["x coord"] - center_x) / scale_factor) - config[
            "calibration offset"
        ][0]
        df["y mm"] = ((df["y coord"] - center_y) / scale_factor * -1) - config[
            "calibration offset"
        ][1]

        if config["testing"]:
            # for testing:
            df["x mm from corner"] = -1 * (
                df["x mm"] - config["plate dimensions"][0] / 2
            )
            df["y mm from corner"] = -1 * (
                df["y mm"] - config["plate dimensions"][1] / 2
            )

            # conduct a regression to find the slope and intercept for x
            x_slope, x_intercept = np.polyfit(
                sorted(df["x mm from corner"]), sorted(final_coords_mm[:, 0]), 1
            )
            # conduct a regression to find the slope and intercept for y
            y_slope, y_intercept = np.polyfit(
                sorted(df["y mm from corner"]), sorted(final_coords_mm[:, 1]), 1
            )

            import matplotlib.pyplot as plt

            # make two subplots
            fig, axs = plt.subplots(1, 2)
            # plot the mm from corner coordinates versus the final_coords_mm
            axs[0].scatter(
                sorted(df["x mm from corner"]), sorted(final_coords_mm[:, 0])
            )
            axs[0].set_xlabel("robot x coord")
            axs[0].set_ylabel("actual x coord")
            # plot the regression line
            axs[0].plot(
                sorted(df["x mm from corner"]),
                x_slope * np.array(sorted(df["x mm from corner"])) + x_intercept,
            )
            # set the title of the plot as the slope and intercept
            axs[0].set_title(f"x slope: {x_slope:.2f}, x intercept: {x_intercept:.2f}")

            axs[1].scatter(
                sorted(df["y mm from corner"]), sorted(final_coords_mm[:, 1])
            )
            axs[1].set_xlabel("robot y coord")
            axs[1].set_ylabel("actual y coord")
            # plot the regression line
            axs[1].plot(
                sorted(df["y mm from corner"]),
                y_slope * np.array(sorted(df["y mm from corner"])) + y_intercept,
            )
            # set the title of the plot as the slope and intercept
            axs[1].set_title(f"y slope: {y_slope:.2f}, y intercept: {y_intercept:.2f}")

            plt.show()
            # TODO make a regression line to calculate calibration offsets for the robot

        # The OT2 will take positions as a percentage of distance from the center of the labware.
        # So add a final two columns to calc the percent distance from the center of the plate.

        df["x%"] = df["x mm"] / (config["plate dimensions"][0] / 2)
        df["y%"] = df["y mm"] / (config["plate dimensions"][1] / 2)

        print("Found {} colonies.".format(df.shape[0]))
        dfs.append((image_path.name, df))
        # print("Writing to ", image_path.with_suffix(".csv"))
        # df.to_csv(image_path.with_suffix(".csv"))

    # concatenate all the dataframes into one with a new column for the image name
    df = pd.concat([df.assign(image_name=image_path) for image_path, df in dfs])
    # add a column for plate number that enumerates the plates
    df["plate"] = df.groupby("image_name").ngroup() + 1
    # write the dataframe to a csv in the same directory as the images
    df.to_csv(image_path.parent / "colony_data.csv", index=False)


if __name__ == "__main__":
    exit(main())
