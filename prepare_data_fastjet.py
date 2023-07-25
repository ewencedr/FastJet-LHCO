import pandas
import matplotlib.pyplot as plt
import numpy as np
import h5py
import fastjet as fj


def cluster(data, n_events=1000):
    out = []

    # Loop over events
    for ievt in range(n_events):
        if ievt % 10000 == 0:
            print(f"step: {ievt}")
        # Build a list of all particles
        pjs = []
        for i in range(data.shape[1]):
            pj = fj.PseudoJet()
            pj.reset_PtYPhiM(data[ievt, i, 0], data[ievt, i, 1], data[ievt, i, 2], 0)
            pjs.append(pj)

        # run jet clustering with AntiKt, R=1.0
        R = 1.0
        jet_def = fj.JetDefinition(fj.antikt_algorithm, R)

        # Save the two leading jets
        jets = jet_def(pjs)
        jets = [j for j in jets if j.pt() > 30.0]
        out.append([jets[0], jets[1]])

    return out


def main():
    print("Loading data...")
    filepath = "/beegfs/desy/user/ewencedr/data/lhco/events_anomalydetection_v2.h5"
    filepath_save = (
        "/beegfs/desy/user/ewencedr/data/lhco/events_anomalydetection_v2_processed_fastjet_test"
    )
    filepath_save_raw = (
        "/beegfs/desy/user/ewencedr/data/lhco/events_anomalydetection_v2_raw_fastjet_test.h5"
    )

    # Load everything into memory
    df = pandas.read_hdf(filepath)
    print(df.shape)
    print("Memory in GB:", sum(df.memory_usage(deep=True)) / (1024**3))

    # split into signal and background
    np_array = np.array(df)
    signal = np_array[np_array[:, 2100] == 1]
    background = np_array[np_array[:, 2100] == 0]

    # change the shape of the data into (n_events, n_particles, n_features)
    background_reduced = background[:, :2100]
    signal_reduced = background[:, :2100]
    qcd_data = background_reduced.reshape(-1, 700, 3)
    sig_data = signal_reduced.reshape(-1, 700, 3)

    # Actually cluster the data
    # this is what takes a long time
    print("Clustering...")
    out_qcd = cluster(qcd_data, n_events=1000)  # len(qcd_data))
    out_sig = cluster(sig_data, n_events=1000)  # len(sig_data))

    print("Processing...")
    for c, jets in enumerate([out_qcd, out_sig]):
        signal_type = "signal" if c == 1 else "background"
        print(f"----{signal_type}----")
        # separate the leading and subleading jets
        x_masses = []
        y_masses = []
        x_jets = []
        y_jets = []
        for jets in jets:
            x_jet = jets[0]
            y_jet = jets[1]
            x_jets.append(x_jet)
            y_jets.append(y_jet)
            x_masses.append(x_jet.m())
            y_masses.append(y_jet.m())
        x_jets = np.array(x_jets)
        y_jets = np.array(y_jets)

        for c, xy_jets in enumerate([x_jets, y_jets]):
            print(f"----{'x' if c == 0 else 'y'}----")
            # get padded constituents, relative coordinates and mask in the wanted format
            len_padding = 300
            constituents = []
            constituents_xyze = []
            rel_constituents = []
            mask = []
            jet_mass = []
            jet_pt = []
            jet_eta = []
            jet_phi = []
            jet_px = []
            jet_py = []
            jet_pz = []
            jet_e = []
            for jet in xy_jets:
                # get constituents
                const_pt = np.array(
                    [(jet.constituents()[i].perp()) for i in range(len(jet.constituents()))]
                )
                const_eta = np.array(
                    [
                        (jet.constituents()[i].pseudorapidity())
                        for i in range(len(jet.constituents()))
                    ]
                )
                const_phi = np.array(
                    [(jet.constituents()[i].phi_std()) for i in range(len(jet.constituents()))]
                )
                # xyze
                const_px = np.array(
                    [(jet.constituents()[i].px()) for i in range(len(jet.constituents()))]
                )
                const_py = np.array(
                    [(jet.constituents()[i].py()) for i in range(len(jet.constituents()))]
                )
                const_pz = np.array(
                    [(jet.constituents()[i].pz()) for i in range(len(jet.constituents()))]
                )
                const_e = np.array(
                    [(jet.constituents()[i].e()) for i in range(len(jet.constituents()))]
                )

                # concatenate
                consts = np.concatenate(
                    (const_pt[:, None], const_eta[:, None], const_phi[:, None]), axis=1
                )
                consts_xyze = np.concatenate(
                    (const_px[:, None], const_py[:, None], const_pz[:, None], const_e[:, None]),
                    axis=1,
                )
                # sort constituents by pT from high to low
                consts = consts[np.argsort(consts[:, 0])[::-1]]

                # pad constituents and mask
                padded_consts = np.pad(
                    consts, ((0, len_padding - len(consts)), (0, 0)), "constant", constant_values=0
                )
                padded_consts_xyze = np.pad(
                    consts_xyze,
                    ((0, len_padding - len(consts_xyze)), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                padded_mask = np.pad(
                    np.ones(len(consts)),
                    (0, len_padding - len(consts)),
                    "constant",
                    constant_values=0,
                )

                # relative coordinates
                rel_constituents_temp = padded_consts.copy()
                rel_constituents_temp[:, 0] = rel_constituents_temp[:, 0] / jet.perp()
                rel_constituents_temp[:, 1] = rel_constituents_temp[:, 1] - jet.pseudorapidity()
                rel_constituents_temp[:, 2] = rel_constituents_temp[:, 2] - jet.phi_std()

                # fix phi range
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

                # jet variables & append to list
                jet_mass.append(jet.m())
                jet_pt.append(jet.perp())
                jet_eta.append(jet.pseudorapidity())
                jet_phi.append(jet.phi_std())
                jet_px.append(jet.px())
                jet_py.append(jet.py())
                jet_pz.append(jet.pz())
                jet_e.append(jet.e())
                constituents.append(padded_consts)
                constituents_xyze.append(padded_consts_xyze)
                rel_constituents.append(rel_constituents_temp)
                mask.append(padded_mask)
            mask = np.array(mask)
            constituents = np.array(constituents) * mask[:, :, None]
            constituents_xyze = np.array(constituents_xyze)
            rel_constituents = np.array(rel_constituents) * mask[:, :, None]
            jet_mass = np.array(jet_mass)
            jet_pt = np.array(jet_pt)
            jet_eta = np.array(jet_eta)
            jet_phi = np.array(jet_phi)
            jet_px = np.array(jet_px)
            jet_py = np.array(jet_py)
            jet_pz = np.array(jet_pz)
            jet_e = np.array(jet_e)
            jet_data = np.concatenate(
                (
                    jet_pt[:, None],
                    jet_eta[:, None],
                    jet_phi[:, None],
                    jet_mass[:, None],
                    jet_px[:, None],
                    jet_py[:, None],
                    jet_pz[:, None],
                    jet_e[:, None],
                ),
                axis=1,
            )

            # remove constituents with 0 pT and very high etas
            counter = 0
            for count_i, i in enumerate(rel_constituents):
                for count_j, j in enumerate(i):
                    if j[1] > 1000:
                        counter += 1
                        rel_constituents[count_i, count_j, 1] = 0
                        rel_constituents[count_i, count_j, 2] = 0
                        mask[count_i, count_j] = 0
            print(f"removed {counter} constituents")

            print(f"constituents shape: {constituents.shape}")
            print(f"constituents_xyze shape: {constituents_xyze.shape}")
            print(f"rel_constituents shape: {rel_constituents.shape}")
            print(f"jet_data shape: {jet_data.shape}")
            print(f"mask shape: {mask.shape}")
            print(f"max particle multipilicity: {np.max(np.sum(mask,axis=-1))}")
            print(f"min particle multipilicity: {np.min(np.sum(mask,axis=-1))}")

            if c == 0:
                constituents_x = constituents.copy()
                constituents_xyze_x = constituents_xyze.copy()
                rel_constituents_x = rel_constituents.copy()
                jet_data_x = jet_data.copy()
                mask_x = mask.copy()
            else:
                constituents_y = constituents.copy()
                constituents_xyze_y = constituents_xyze.copy()
                rel_constituents_y = rel_constituents.copy()
                jet_data_y = jet_data.copy()
                mask_y = mask.copy()

        final_path = filepath_save + f"_{signal_type}.h5"
        with h5py.File(final_path, "w") as f:
            f.create_dataset("constituents_x", data=constituents_x)
            f.create_dataset("constituents_xyze_x", data=constituents_xyze_x)
            f.create_dataset("rel_constituents_x", data=rel_constituents_x)
            f.create_dataset("jet_data_x", data=jet_data_x)
            f.create_dataset("mask_x", data=mask_x)
            f.create_dataset("constituents_y", data=constituents_y)
            f.create_dataset("constituents_xyze_y", data=constituents_xyze_y)
            f.create_dataset("rel_constituents_y", data=rel_constituents_y)
            f.create_dataset("jet_data_y", data=jet_data_y)
            f.create_dataset("mask_y", data=mask_y)
        print(f"Succesfully saved to {final_path}")


if __name__ == "__main__":
    print(f"Running {__file__}")
    main()
