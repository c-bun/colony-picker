from typing import Tuple, List
from opentrons import protocol_api
from opentrons.types import Location
import time
import json
import csv

# metadata
metadata = {
    "protocolName": "Colony Picker into 384 Well Plate",
    "author": "Colin Rathbun <rathbunc@dickinson.edu>",
    "description": "Given a CSV of colony locations, pick the colonies and innoculate them in a 384 well plate.",
    "apiLevel": "2.12",
}

# This should be False when not testing
TESTING = False

# Hardcoded parameters
PLATE_DIMENSIONS = [
    118.63,
    82.13,
]  # TODO: this only works with the current CELLTREAT plate
NUMBER_OF_WELL_PLATES = 1  # This can only be 1 for now because we will run out of tips.
NUMBER_OF_WELLS = 384
FRZ_WELL = "A2"  # Well with furimazine
PBS_WELL = "A1"  # Well with PBS
CONTROL_WELLS = ["A1", "B1", "C1", "D1", "E1", "F1"]  # Don't forget to pick these!
RUNTIME_OFFSET = (  # set to the direction you want the pipette to move in mm
    0,  # x: increase to move right
    0,  # y: increase to move up
)
FILL_VOLUME = 10  # this is the amount that will be initially added to the well plate
FRZ_VOLUME = 5  # this should not be less than 5

colonies_picked = len(CONTROL_WELLS)

# Paste CSV files here. This can only be between 1 and 4 dishes. There MUST be enough colonies in the csv to fill all well plates.
# this is templated in the jinja2 file
if TESTING:  # use some dummy data
    PLATE_CSV = """
x coord,y coord,quality,x mm,y mm,x%,y%,image_name,plate
716,265,1.0,47.40082970839375,-0.8174158208474304,0.7991373127942974,-0.01990541387671814,2023-04-06_12-59-01_8bit.png,1
484,301,1.0,14.172539068793299,-5.973529885613018,0.23893684681435218,-0.14546523525174768,2023-04-06_12-59-01_8bit.png,1
476,164,1.0,13.026735943289834,13.648348638633802,0.21961958936676784,0.3323596405365592,2023-04-06_12-59-01_8bit.png,1
464,212,1.0,11.308031255034638,6.773529885613018,0.19064370319539137,0.16494654536985312,2023-04-06_12-59-01_8bit.png,1
746,154,0.9999999999999996,51.69759142903174,15.080602545513132,0.8715770282227386,0.3672373686962896,2023-04-06_12-59-01_8bit.png,1
698,467,0.9999999999999996,44.822772676010956,-29.748944739809897,0.7556734835372327,-0.7244355227032728,2023-04-06_12-59-01_8bit.png,1
671,457,0.9999999999999996,40.95568712743677,-28.316690832930565,0.6904777396516356,-0.6895577945435424,2023-04-06_12-59-01_8bit.png,1
643,50,0.9999999999999996,36.94537618817464,29.97604317705816,0.6228673385850905,0.729965741557486,2023-04-06_12-59-01_8bit.png,1
578,464,0.9999999999999996,27.635725793459002,-29.319268567746096,0.465914621823468,-0.7139722042553537,2023-04-06_12-59-01_8bit.png,1
491,409,0.9999999999999996,15.175116803608828,-21.441872079909782,0.25583944708098844,-0.5221446993768363,2023-04-06_12-59-01_8bit.png,1"""
else:
    PLATE_CSV = """
    {{colony_locations}}"""


csv_data = PLATE_CSV.splitlines()[1:]  # Discard the blank first line.
colonies = list(csv.DictReader(csv_data))
num_plates = len(set([x for x in [c["plate"] for c in colonies]]))


def populate_deck(
    protocol: protocol_api.ProtocolContext,
    next_open_position=1,
):

    if TESTING:  # use some dummy labware
        petri_dishes = [
            protocol.load_labware(
                "axygen_1_reservoir_90ml",
                next_open_position + i,
            )
            for i in range(num_plates)
        ]
        next_open_position += len(petri_dishes)

        well_plates = [
            protocol.load_labware(
                "corning_384_wellplate_112ul_flat",
                next_open_position + i,
            )
            for i in range(NUMBER_OF_WELL_PLATES)
        ]
        next_open_position += len(well_plates)

    else:
        petri_dishes = [
            protocol.load_labware(
                "celltreat_1_wellplate_48000ul", next_open_position + i
            )
            for i in range(num_plates)
        ]
        next_open_position += len(petri_dishes)

        well_plates = [
            protocol.load_labware(
                "grenierbioone_384_wellplate_138ul", next_open_position + i
            )
            for i in range(NUMBER_OF_WELL_PLATES)
        ]
        next_open_position += len(well_plates)

    # Load in a 12-well reservoir.
    reservoir = protocol.load_labware("nest_12_reservoir_15ml", next_open_position)
    next_open_position += 1

    # Load in the tipracks. Fill the rest of the open positions.
    # TODO: in the future this should only use the number of tips needed.
    tip_racks = [
        protocol.load_labware("opentrons_96_tiprack_20ul", next_open_position + i)
        for i in range(12 - next_open_position)
    ]

    print(
        """
    Loaded labware:
    - 12-well reservoir: {}
    - {} 384-well plates: {}
    - {} petri dishes: {}
    - {} tipracks: {}
    """.format(
            reservoir,
            len(well_plates),
            well_plates,
            len(petri_dishes),
            petri_dishes,
            len(tip_racks),
            tip_racks,
        )
    )

    return tip_racks, reservoir, well_plates, petri_dishes


def pick_colony(
    pipette: protocol_api.InstrumentContext,
    petri_dish: protocol_api.labware.Labware,
    colony_location: Tuple[float, float],  # this is the %x and %y location
    offset: Tuple[float, float] = (0, 0),
):
    """
    Given a plate location and a colony location, pick a colony.
    """
    # Get the colony location in the plate coordinate system.
    x, y = colony_location

    x_offset = offset[0] / (
        PLATE_DIMENSIONS[0] / 2
    )  # TODO: this is hacky to have to convert from mm
    y_offset = offset[1] / (PLATE_DIMENSIONS[1] / 2)

    x += x_offset  # TODO: make sure this is correct
    y += y_offset  # TODO: make sure this is correct

    pipette.pick_up_tip()

    # Move to the colony location. Dip down into the petri dish. Pull back up.
    pipette.move_to(
        Location(petri_dish.wells()[0].from_center_cartesian(x=x, y=y, z=1), petri_dish)
    )
    pipette.move_to(
        Location(
            petri_dish.wells()[0].from_center_cartesian(x=x, y=y, z=-1), petri_dish
        )
    )
    pipette.move_to(
        Location(petri_dish.wells()[0].from_center_cartesian(x=x, y=y, z=1), petri_dish)
    )


def innoculate_colony(
    pipette: protocol_api.InstrumentContext,
    well_plates: List[protocol_api.labware.Labware],
):
    """
    With a picked colony on the tip of the pipette, innoculate the colony in the next availble well.
    """
    global colonies_picked
    plate = well_plates[colonies_picked // NUMBER_OF_WELLS]
    well = plate.wells()[
        colonies_picked % NUMBER_OF_WELLS
    ]  # this accesses the plate down the columns
    pipette.aspirate(10, well)
    pipette.dispense(10, well)
    # pipette.aspirate(10, well)
    # pipette.dispense(10, well)
    pipette.drop_tip()
    colonies_picked += 1


def run(protocol: protocol_api.ProtocolContext):

    tip_racks, reservoir, well_plates, petri_dishes = populate_deck(protocol)

    # pipettes
    left_pipette = protocol.load_instrument(
        "p20_single_gen2", "left", tip_racks=tip_racks
    )
    right_pipette = protocol.load_instrument(
        "p20_multi_gen2", "right", tip_racks=tip_racks
    )

    # Fill the well plates with FILL_VOL from the reservoir.
    for plate in well_plates:
        right_pipette.transfer(
            FILL_VOLUME,
            reservoir.wells_by_name()[PBS_WELL],
            plate.wells(),
            new_tip="once",
        )

    for colony in colonies:
        if colonies_picked < (NUMBER_OF_WELLS * len(well_plates)):
            print(
                "Picking colony {} of {} from {} at x={} and y={}".format(
                    colonies_picked + 1,
                    len(well_plates * NUMBER_OF_WELLS),
                    colony["plate"],
                    colony["x%"],
                    colony["y%"],
                )
            )
            pick_colony(
                left_pipette,
                petri_dishes[int(colony["plate"]) - 1],
                (float(colony["x%"]), float(colony["y%"])),
                offset=RUNTIME_OFFSET,
            )
            if colonies_picked < len(CONTROL_WELLS) + 5:
                # pause to make sure that the tip is in the right spot for the first three colonies
                protocol.pause(
                    "Is the tip in the right spot? If not, adjust the RUNTIME_OFFSET."
                )
            innoculate_colony(left_pipette, well_plates)
        else:
            print("Done with all plates.")
            break

    protocol.pause("Done with colonies, press resume to add furimazine.")

    # Add 5 uL furimazine to the well plates from the reservoir.
    for plate in well_plates:
        right_pipette.transfer(
            FRZ_VOLUME,
            reservoir.wells_by_name()[FRZ_WELL],
            [well.top(-5) for well in plate.wells()],
            new_tip="once",
            touch_tip=True,
        )
