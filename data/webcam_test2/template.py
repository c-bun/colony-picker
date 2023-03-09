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
TESTING = True

# Other hardcoded parameters
NUMBER_OF_WELL_PLATES = 1  # This can only be 1 for now because we will run out of tips.
NUMBER_OF_WELLS = 384
FRZ_WELL = "A2"
PBS_WELL = "A1"
CONTROL_WELLS = ["A1", "B1", "C1", "D1", "E1", "F1"]  # Don't forget to pick these!

colonies_picked = len(CONTROL_WELLS)

# Paste CSV files here. This can only be between 1 and 4 dishes. There MUST be enogh colonies in the csv to fill all well plates.
# this is templated in the jinja2 file
PLATE_CSV = """
{{colony_locations}}"""


csv_data = PLATE_CSV.splitlines()[1:]  # Discard the blank first line.
colonies = list(csv.DictReader(csv_data))
num_plates = len(set([x for x in [c["plate"] for c in colonies]]))


def populate_deck(
    protocol: protocol_api.ProtocolContext,
    next_open_position=1,
):

    if TESTING:
        petri_dishes = [
            protocol.load_labware_from_definition(
                json.load(
                    open(
                        "../labware/celltreat_1_wellplate_48000ul/celltreat_1_wellplate_48000ul.json"
                    )
                ),
                next_open_position + i,
            )
            for i in range(num_plates)
        ]
        next_open_position += len(petri_dishes)

        well_plates = [
            protocol.load_labware_from_definition(
                json.load(
                    open(
                        "../labware/grenierbioone_384_wellplate_138ul/grenierbioone_384_wellplate_138ul.json"
                    )
                ),
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
    colony_location: tuple,
):
    """
    Given a plate location and a colony location, pick a colony.
    """
    # Get the colony location in the plate coordinate system.
    x, y = colony_location

    # TODO I'm guessing on all these parameters. Verify first in the lab.
    pipette.pick_up_tip()

    # Move to the colony location. Dip down into the petri dish. Pull back up.
    pipette.move_to(
        Location(petri_dish.wells()[0].from_center_cartesian(x=x, y=y, z=1), petri_dish)
    )  # TODO make sure z is correct. How do we set this as an offest?
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

    # Fill the well plates with 20 uL from the reservoir.
    for plate in well_plates:
        right_pipette.transfer(
            20,
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
            )
            if colonies_picked < 3:
                # pause to make sure that the tip is in the right spot for the first three colonies
                protocol.pause("Is the tip in the right spot?")
            innoculate_colony(left_pipette, well_plates)
        else:
            print("Done with all plates.")
            break
    print("Done with plate {}".format(plate))

    protocol.pause(
        "Done with colonies, ADD TIPS TO ALL EMPTY POSITIONS and press resume to add furimazine."
    )
    # This assumes that we added tips!!
    right_pipette.reset_tipracks()
    left_pipette.reset_tipracks()

    # Add 5 uL furimazine to the well plates from the reservoir.
    for plate in well_plates:
        right_pipette.transfer(
            5,
            reservoir.wells_by_name()[FRZ_WELL],
            plate.wells(),
            new_tip="always",
            mix_after=(1, 5),
        )
