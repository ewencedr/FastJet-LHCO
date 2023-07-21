import pandas
import matplotlib.pyplot as plt
import numpy as np
import h5py
from pyjet import cluster, DTYPE_PTEPM


def main():
    filepath = "/beegfs/desy/user/ewencedr/data/lhco/events_anomalydetection_v2.h5"
    filepath_save = "/beegfs/desy/user/ewencedr/data/lhco/events_anomalydetection_v2_processed.h5"

    # Option 1: Load everything into memory
    events_combined = pandas.read_hdf(filepath)
    print(f"events_combined shape: {np.shape(events_combined)}")
    print("Memory in GB:", sum(events_combined.memory_usage(deep=True)) / (1024**3))

    # events_combined_perm = np.array(events_combined)[
    #    np.random.permutation(len(np.array(events_combined)))
    # ]
    # events_combined = events_combined_perm[:10000]
    events_combined = np.array(events_combined)

    # Now, let's cluster some jets!
    leadpT = {}
    alljets = {}
    for mytype in ["background", "signal"]:
        leadpT[mytype] = []
        alljets[mytype] = []
        for i in range(np.shape(events_combined)[0]):
            if i % 10000 == 0:
                print(mytype, i)
                pass
            issignal = events_combined[i][2100]
            if mytype == "background" and issignal:
                continue
            elif mytype == "signal" and issignal == 0:
                continue
            pseudojets_input = np.zeros(
                len([x for x in events_combined[i][::3] if x > 0]), dtype=DTYPE_PTEPM
            )
            for j in range(700):
                if events_combined[i][j * 3] > 0:
                    pseudojets_input[j]["pT"] = events_combined[i][j * 3]
                    pseudojets_input[j]["eta"] = events_combined[i][j * 3 + 1]
                    pseudojets_input[j]["phi"] = events_combined[i][j * 3 + 2]
                    pass
                pass
            sequence = cluster(pseudojets_input, R=1.0, p=-1)
            jets = sequence.inclusive_jets(ptmin=20)
            leadpT[mytype] += [jets[0].pt]
            alljets[mytype] += [jets]
            pass

    # sort the jets by mass and take only the first jet
    jets = alljets["background"]
    x_jets = []
    for jet in jets:
        jets_to_sort = jet[:2]  # only sort the first two subjets with highest pT
        sorted_subjet = sorted(jets_to_sort, key=lambda x: x.mass, reverse=True)
        x_jets.append(sorted_subjet[0])
    x_jets = np.array(x_jets)

    rel_constituents = []
    len_constituents = []
    mask = []
    len_padding = 300
    for jet in x_jets:
        const_temp = []
        # transform void struct to array
        for constituent_void in jet.constituents_array():
            array = np.asarray(constituent_void).tolist()
            const_temp.append(array)
        unpadded_consts = np.array(const_temp)
        mask_single_jet = np.ones(len(unpadded_consts))
        # pad constituents and mask
        padded_mask = np.pad(
            mask_single_jet, (0, len_padding - len(mask_single_jet)), "constant", constant_values=0
        )
        padded_consts = np.pad(
            unpadded_consts,
            ((0, len_padding - len(unpadded_consts)), (0, 0)),
            "constant",
            constant_values=0,
        )

        # relative coordinates
        rel_constituents_temp = padded_consts.copy()
        rel_constituents_temp[:, 0] = rel_constituents_temp[:, 0] / jet.pt
        rel_constituents_temp[:, 1] = rel_constituents_temp[:, 1] - jet.eta
        rel_constituents_temp[:, 2] = rel_constituents_temp[:, 2] - jet.phi
        rel_constituents_temp[:, 2] = np.where(
            rel_constituents_temp[:, 2] > np.pi,
            rel_constituents_temp[:, 2] - 2 * np.pi,
            rel_constituents_temp[:, 2],
        )
        rel_constituents_temp[:, 2] = np.where(
            rel_constituents_temp[:, 2] < -np.pi,
            rel_constituents_temp[:, 2] + 2 * np.pi,
            rel_constituents_temp[:, 2],
        )

        rel_constituents.append(rel_constituents_temp)
        mask.append(padded_mask)
        len_constituents.append(len(jet.constituents_array()))
    rel_constituents = np.array(rel_constituents)
    mask = np.array(mask)
    print(f"max constituents: {np.max(len_constituents)}")
    print(f"min constituents: {np.min(len_constituents)}")
    print(f"mask shape: {mask.shape}")
    print(f"rel_constituents shape: {rel_constituents.shape}")

    with h5py.File(filepath_save, "w") as f:
        f.create_dataset("data", data=rel_constituents)
        f.create_dataset("mask", data=mask)
    print(f"Succesfully saved to {filepath_save}")


if __name__ == "__main__":
    main()
