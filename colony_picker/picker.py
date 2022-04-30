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

def warp(img, initial_coords, final_coords):
    """
    initial and final coords are a list of tuples with x and y values.
    """
    from skimage import transform
    import numpy as np

    for_transform = transform.estimate_transform('projective', np.array(initial_coords), np.array(final_coords))
    transformed = transform.warp(img, for_transform.inverse)

    return transformed

def warp_squareplate(img, initial_coords, target=(150,150)):
    """
    Calculate final coords based on small plate dimensions. Incorporate a crop to get rid of the plate edges.
    """
    imdims = (img.shape[1], img.shape[0])

    final_coords = [
        target,
        (imdims[0]-target[0], target[1]),
        (target[0], imdims[1]-target[1]),
        (imdims[0]-target[0], imdims[1]-target[1])
    ]

    return warp(img, initial_coords, final_coords)

def flatten(img, sigma):
    from skimage.filters import gaussian
    return img - gaussian(img, sigma=sigma)

def find_colonies(plate_processed, desired_count=192):
    image = plate_processed

    edges = canny(image)

    hough_radii = np.arange(10, 50, 1)  # the ranges of radii to search for circles
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res,
        hough_radii,
        min_xdistance=30,
        min_ydistance=30,
        total_num_peaks=desired_count,
        normalize=True,
    )

    #print(len(accums), len(cx), len(cy), len(radii))

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = np.int16(
        np.interp(image, (image.min(), image.max()), (0, 255))
    )  # this rescales the image to be between 0 and 255
    cimage = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=cimage.shape)
        cimage[circy, circx] = (220, 20, 20)

    ax.imshow(cimage)
    plt.show()
    return accums, cx, cy

def main():

    parser = argparse.ArgumentParser(
        description="""
        Takes image and plate coordinates as input. Outputs locations of colonies relative to the upper left coordinate given.
        """
    )

    parser.add_argument(
        "imagepath", type=str, help="Path to the image file."
    )
    parser.add_argument(
        "coord 1x",
        type=int,
        help="Upper left corner coordinate.",
    )
    parser.add_argument(
        "coord 1y",
        type=int,
        help="Upper left corner coordinate.",
    )
    parser.add_argument(
        "coord 2x",
        type=int,
        help="Upper right corner coordinate.",
    )
    parser.add_argument(
        "coord 2y",
        type=int,
        help="Upper right corner coordinate.",
    )
    parser.add_argument(
        "coord 3x",
        type=int,
        help="Lower left corner coordinate.",
    )
    parser.add_argument(
        "coord 3y",
        type=int,
        help="Lower left corner coordinate.",
    )
    parser.add_argument(
        "coord 4x",
        type=int,
        help="Lower right corner coordinate.",
    )
    parser.add_argument(
        "coord 4y",
        type=int,
        help="Lower right corner coordinate.",
    )
    parser.add_argument(
        "num colonies",
        type=int,
        help="Number of colonies I should find.",
    )
    parser.add_argument(
        "--targetx",
        type=int,
        default=150,
        help="Target x location for upper left cross of plate.",
    )
    parser.add_argument(
        "--targety",
        type=int,
        default=150,
        help="Target y location for upper left cross of plate.",
    )
    parser.add_argument(
        "--targetwidth",
        type=float,
        default=40,
        help="Distance between the top two target points in millimeters to allow conversion of units.",
    )
    

    args = vars(parser.parse_args())
    path = pathlib.Path(args['imagepath'])

    img = io.imread(path)
    img = np.asarray(img)[:, :, :3]
    plate = color.rgb2gray(img)
    flattened = flatten(plate, .6*plate.shape[0]) # the size of the gaussian here should work generally.
    warped = warp_squareplate(
        flattened, 
        [
            (args['coord 1x'], args['coord 1y']),
            (args['coord 2x'], args['coord 2y']),
            (args['coord 3x'], args['coord 3y']),
            (args['coord 4x'], args['coord 4y']),
        ],
        target=(args['targetx'], args['targety'])
        )

    accums, cx, cy = find_colonies(
        warped,
        args["num colonies"]
    )

    df = pd.DataFrame(
        {
            'x coord':cx,
            'y coord':cy,
            'quality':accums,
        }
    )
    mm_conversion_factor = args['targetwidth'] / (warped.shape[1] - (2*args['targetx']))
    df['x coord'] = df['x coord'].copy() - args['targetx']
    df['y coord'] = df['y coord'].copy() - args['targety']
    df['x mm'] = df['x coord']*mm_conversion_factor # TODO implement conversion here
    df['y mm'] = df['y coord']*mm_conversion_factor

    print("Writing to ", path.with_suffix('.csv'))

    df.to_csv(path.with_suffix('.csv'))

if __name__ == "__main__":
    exit(main())
