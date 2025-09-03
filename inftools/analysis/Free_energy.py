import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
import glob

def extract(trajfile, lm1, lA, lB, xcol, ycol=None):
    # Read and process the file
    traj = np.loadtxt(trajfile)
    first = traj[0, xcol]
    second = traj[1, xcol]
    last = traj[-1, xcol]
    if first >= lA and last >= lA and second < first:
        type = "0-RMR"
    elif first >= lA and last <= lm1:
        type = "0-RML"
    elif first <= lm1 and last <= lm1:
        type = "0-LML"
    elif first <= lm1 and last >= lA:
        type = "0-LMR"
    elif first <= lA and last >= lB:
        type = "0+LMR"
    elif first <= lA and first <= lA and second > first:
        type = "0+LML"
    else:
        print(f"first phasepoint is: {first}")
        print(f"last phasepoint is: {last}")
        raise ValueError("Unexpected type encountered.")
    data = traj[1:-1, xcol] # remove first and last frames
    if ycol is not None:
        data = np.vstack((data, traj[1:-1, ycol]))
    return data, type, len(traj)


def update_histogram(label, data, factor, histogram, Minx, Miny, dx, dy):
    if Miny is not None and dy is not None:
        x = data[0]
        y = data[1]

        ix = ((x - Minx) / dx).astype(int)
        iy = ((y - Miny) / dy).astype(int)

        np.add.at(histogram, (ix, iy), factor)

    else:
        x = data if data.ndim == 1 else data[:,0] # make sure x is one dimensional
        ix = ((x - Minx) / dx).astype(int)
        if np.any(ix >= len(histogram)):
            print(f"{label}: The one before the last phasepoint is on the lB interface due to a rounding issue: {x}")
            ix = np.where(ix >= len(histogram), len(histogram) - 1, ix)
        np.add.at(histogram, ix, factor)

    return histogram


def calculate_free_energy(trajlabels, WFtot, Trajdir, outfolder, histo_stuff, lm1, lA, lB, sym):
    print(f"="*55)
    print(f"Lambda_minus_one is {lm1}!") 
    print(f"="*55)
    type_count = {
        "0-RMR": [0, 0],
        "0-RML": [0, 0],
        "0-LML": [0, 0],
        "0-LMR": [0, 0],
        "0+LMR": [0, 0],
        "0+LML": [0, 0]
    }
    length_count = {
        "L0-": 0,
        "L0+": 0
    }

    Nbinsx, Nbinsy = histo_stuff["nbx"], histo_stuff["nby"]
    Maxx, Minx = histo_stuff["maxx"], histo_stuff["minx"]
    Maxy, Miny = histo_stuff["maxy"], histo_stuff["miny"]
    xcol, ycol = histo_stuff["xcol"], histo_stuff["ycol"]
    
    if any(var is None for var in [Nbinsy, Maxy, Miny, ycol]):
        none_vars = [name for name, var in zip(["nby", "maxy", "miny", "ycol"], [Nbinsy, Maxy, Miny, ycol]) if var is None]
        assert all(var is None for var in [Nbinsy, Maxy, Miny, ycol]), \
            f"The following variables are None and should be set: {', '.join(none_vars)}"
    if Nbinsy is not None:
        histogram = np.zeros((Nbinsx, Nbinsy))
        dy = (Maxy - Miny) / Nbinsy
        yval = [Miny + 0.5 * dy + i * dy for i in range(Nbinsy)]
    else:
        histogram = np.zeros(Nbinsx)
        dy = None
        yval = None
    
    Minx = lm1
    Maxx = lB
    dx = (Maxx - Minx) / Nbinsx
    xval = [Minx + 0.5 * dx + i * dx for i in range(Nbinsx)]
    index_lA = np.where(np.array(xval) > lA)[0][0]

    for label, factor in zip(trajlabels, WFtot):
        trajfile = Trajdir + "/" + str(label) + "/order.txt"
        data, type, L0 = extract(trajfile, lm1, lA, lB, xcol, ycol)
        type_count[type][0] += 1
        type_count[type][1] += factor
        if type[1] == "-":
            length_count["L0-"] += (L0 * factor)
        if type[1] == "+":
            length_count["L0+"] += (L0 * factor)
        histogram = update_histogram(label, data, factor, histogram, Minx, Miny, dx, dy)

    R_END = type_count["0-LMR"][1] + type_count["0-RMR"][1]
    L_END = type_count["0-LML"][1] + type_count["0-RML"][1]
    xi = R_END / (R_END + L_END)
    print(f"-"*35)
    print(f"Path count:")
    print(f"-"*35)
    for key in type_count:
        print(f"{key:7}: {type_count[key][0]}")
    print(f"Total  : {sum(type_count[key][0] for key in type_count)}")
    print(f"-"*35)
    print(f"Weights:")
    print(f"-"*35)
    for key in type_count:
        print(f"{key:7}: {type_count[key][1]}")
    print(f"0-     : {type_count['0-LMR'][1] + type_count['0-LML'][1] + type_count['0-RML'][1] + type_count['0-RMR'][1]}")
    print(f"0+     : {type_count['0+LMR'][1] + type_count['0+LML'][1]}")
    print(f"-"*35)
    print(f"Calculated xi value: {xi:.6f}")
    if Nbinsy is None:
        expected_xi = histogram[index_lA-1] / histogram[index_lA]
        print(f"Expected xi value: {expected_xi:.6f}")
    print(f"L0- (with end-points) is: {length_count['L0-']:.6f}")
    print(f"L0+ (with end-points) is: {length_count['L0+']:.6f}")
    print(f"Flux (without end-points) is: {(1 / (length_count['L0-'] + length_count['L0+'] - 4)):.6f}")
    print(f"Corrected flux (without end-points) is: {(xi / (length_count['L0-'] - 2 + (xi * (length_count['L0+'] - 2)))):.6f}")
    print(f"="*55)

    np.savetxt(os.path.join(outfolder, "histo_xval.txt"), xval)
    np.savetxt(os.path.join(outfolder, "histo_probability.txt"), histogram)

    plt.figure(figsize=(6, 4))
    if Nbinsy is None:
        plt.plot(xval, histogram, marker='o', linestyle='-')
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Probability")
        plt.grid()
    else:
        plt.pcolormesh(xval, yval, histogram.T, shading='auto', cmap='viridis')
        plt.xlabel("X (Å)")
        plt.ylabel("Y (Å)")
        plt.colorbar(label="Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(outfolder, "histogram.png"), dpi=300)
    plt.close()

    if Nbinsy is not None:
        histogram[:index_lA, :] = histogram[:index_lA, :] / xi
    else:
        histogram[:index_lA] = histogram[:index_lA] / xi
    np.savetxt(os.path.join(outfolder, "histo_xi_corrected.txt"), histogram)
    plt.figure(figsize=(6, 4))
    if Nbinsy is None:
        plt.plot(xval, histogram, marker='o', linestyle='-')
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Probability")
        plt.grid()
    else:
        plt.pcolormesh(xval, yval, histogram.T, shading='auto', cmap='viridis')
        plt.xlabel("X (Å)")
        plt.ylabel("Y (Å)")
        plt.colorbar(label="Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(outfolder, "histogram_xi_corrected.png"), dpi=300)
    plt.close()

    max_value = np.max(histogram)
    histogram /= max_value
    conditional_free_energy = -np.log(histogram)
    np.savetxt(os.path.join(outfolder, "free_energy_cond.txt"), conditional_free_energy)
    plt.figure(figsize=(6, 4))
    if Nbinsy is None:
        plt.plot(xval, conditional_free_energy, '-o', label="Conditional A→B")
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Free energy (kBT)")
        plt.grid()
    else:
        plt.pcolormesh(xval, yval, conditional_free_energy.T, shading='auto', cmap='viridis')
        plt.xlabel("X (Å)")
        plt.ylabel("Y (Å)")
        plt.colorbar(label="Free energy (eV)")
    plt.tight_layout()
    plt.savefig(os.path.join(outfolder, "free_energy_cond.png"), dpi=300)
    plt.close()

    if sym:
        # histogram_sym = np.append(histogram, np.zeros(index_lA, dtype=histogram.dtype))
        histogram_sym = histogram[index_lA:]
        histo_sym = histogram_sym + histogram_sym[::-1]
        max_value = np.max(histo_sym)
        histo_sym /= max_value
        free_energy_sym = -np.log(histo_sym)
        np.savetxt(os.path.join(outfolder, "free_energy_sym.txt"), free_energy_sym)
        xval_cond_BA = [Minx + 0.5 * dx + i * dx for i in range(Nbinsx+index_lA)]
        # xval_sym = [Minx + 0.5 * dx + i * dx for i in range(Nbinsx+index_lA)]
        xval_sym = [lA + 0.5 * dx + i * dx for i in range(Nbinsx-index_lA)]
        np.savetxt(os.path.join(outfolder, "xval_sym.txt"), xval_sym)
        plt.figure(figsize=(6, 4))
        plt.plot(xval, conditional_free_energy, '-o', label="Conditional A→B")
        plt.plot(xval_cond_BA[index_lA:], conditional_free_energy[::-1], '-o', label="Conditional B→A")
        # plt.plot(xval_sym, sym_free_energy, '-o', label="Symmetrized")
        plt.plot(xval_sym, free_energy_sym, '-o', label="Symmetrized")
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Free energy (kBT)")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outfolder, "free_energy_sym.png"), dpi=300)
        plt.close()

    if os.path.exists("load1_all"):

        def natural_key(s):
            return [int(text) if text.isdigit() else text.lower() 
                    for text in re.split('([0-9]+)', s)]

        sum_weight = 0
        total_paths = 0
        all_values = []
        all_values_weighted = []
        type_counts = Counter()
        with open("wham/lo_path_Weights.txt", "w") as out_file:
            out_file.write(f"# Weights of the lo paths in load1_all folder, the paths from load1 folder are not included here!\n")
            out_file.write(f"# Path number, weight, MC move\n")
            for folder_root, write_weights in [("load1_all", True), ("load", False)]:
                for folder_name in sorted(os.listdir(folder_root), key=natural_key):
                    folder_path = os.path.join(folder_root, folder_name)
                    if os.path.isdir(folder_path):
                        file_path = os.path.join(folder_path, "order.txt")
                        if os.path.isfile(file_path):
                            with open(file_path, 'r') as f:
                                first_line = f.readline()
                                match_weight = re.search(r'weight\s*=\s*([\d\.Ee+-]+)', first_line)
                                match_type = re.search(r'Cycle:\s*([^,]+)', first_line)

                                if match_weight and match_type:
                                    weight = int(float(match_weight.group(1)))
                                    type_str = match_type.group(1).strip()

                                    sum_weight += weight
                                    total_paths += 1
                                    type_counts[type_str] += 1

                                    # Only write weights when processing "load1_all"
                                    if write_weights:
                                        out_file.write(f"{folder_name}\t{weight}\t{type_str}\n")

                                    # Load second-column data
                                    data = np.loadtxt(file_path, usecols=1, comments='#')
                                    all_values.extend(data)

                                    # Repeat data according to weight
                                    repeated_data = np.tile(data, weight)
                                    all_values_weighted.extend(repeated_data)

        for folder_name in sorted(os.listdir("load1"), key=natural_key):
            folder_path = os.path.join("load1", folder_name)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, "order.txt")
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        first_line = f.readline()
                        match_weight = re.search(r'weight\s*=\s*([\d\.Ee+-]+)', first_line)
                        match_type = re.search(r'Cycle:\s*([^,]+)', first_line)

                        if match_weight and match_type:
                            weight = int(float(match_weight.group(1)))
                            type_str = match_type.group(1).strip()

                            sum_weight += weight
                            total_paths += 1
                            type_counts[type_str] += 1

        # === Print summary ===
        print("lo paths summary:")
        print(f"-"*25)
        for t, count in type_counts.items():
            print(f"{t}: {count}")
        print(f"Total paths: {total_paths}")
        print(f"Sum of weights: {sum_weight}")
        print(f"="*55)

        # === Plot raw histogram (unweighted) ===
        all_values = np.array(all_values)
        hist, _ = np.histogram(all_values, bins=Nbinsx, range=(Minx, Maxx))

        plt.figure(figsize=(6, 4))
        plt.plot(xval, hist, '-o')
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Count")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(outfolder, "histogram_lo_raw.png"), dpi=300)
        plt.close()

        # === Plot weighted histogram ===
        all_values_weighted = np.array(all_values_weighted)
        hist, _ = np.histogram(all_values_weighted, bins=Nbinsx, range=(Minx, Maxx))
        if Nbinsy is not None:
            hist[:index_lA, :] = hist[:index_lA, :] / xi
        else:
            hist[:index_lA] = hist[:index_lA] / xi
        plt.figure(figsize=(6, 4))
        plt.plot(xval, hist, '-o')
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Count")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(outfolder, "histogram_lo_weighted.png"), dpi=300)
        plt.close()

        # === Plot free energy from weighted histogram ===
        histogram = hist.astype(float) / np.max(hist)
        conditional_free_energy = -np.log(histogram)
        conditional_free_energy -= np.min(conditional_free_energy)

        plt.figure(figsize=(6, 4))
        plt.plot(xval, conditional_free_energy, '-o')
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Free Energy (kBT)")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(outfolder, "free_energy_lo_cond.png"), dpi=300)
        plt.close()

        if sym:
            histogram_sym = histogram[index_lA:]
            histo_sym = histogram_sym + histogram_sym[::-1]
            max_value = np.max(histo_sym)
            histo_sym /= max_value
            free_energy_sym = -np.log(histo_sym)
            np.savetxt(os.path.join(outfolder, "free_energy_lo_sym.txt"), free_energy_sym)
            xval_cond_BA = [Minx + 0.5 * dx + i * dx for i in range(Nbinsx+index_lA)]
            xval_sym = [lA + 0.5 * dx + i * dx for i in range(Nbinsx-index_lA)]
            np.savetxt(os.path.join(outfolder, "xval_lo_sym.txt"), xval_sym)
            plt.figure(figsize=(6, 4))
            plt.plot(xval, conditional_free_energy, '-o', label="Conditional A→B")
            plt.plot(xval_cond_BA[index_lA:], conditional_free_energy[::-1], '-o', label="Conditional B→A")
            plt.plot(xval_sym, free_energy_sym, '-o', label="Symmetrized")
            plt.xlabel("Order parameter (Å)")
            plt.ylabel("Free energy (kBT)")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outfolder, "free_energy_lo_sym.png"), dpi=300)
            plt.close()

    log_files = sorted(glob.glob("worker*.log"))
    print(f"Analyzing {log_files}:")
    print(f"-"*45)

    # Initialize total counters
    totals = {
        "total_moves": 0,
        "regular_MC_started": 0,
        "zero_swap_started": 0,
        "engine_swap_started": 0,
        "engine_swap_DDU_rejected": 0,
        "engine_swap_DDU_accepted": 0,
        "engine_swap_DDU_lo_rejected": 0,
        "engine_swap_DDU_hi_rejected": 0,
    }

    for filename in log_files:
        with open(filename, "r") as f:
            content = f.read()
            counts = {
                "total_moves": content.count("Selected engines"),
                "regular_MC_started": content.count("Engine swap rejected"),
                "zero_swap_started": content.count("Zero swap!"),
                "engine_swap_started": content.count("Engine swap started"),
                "engine_swap_DDU_accepted": content.count("DeltaDeltaU Accepted!"),
                "engine_swap_DDU_rejected": content.count("DeltaDeltaU Rejected!"),
                "engine_swap_DDU_lo_rejected": content.count("engine swap (lo path creation) with"),
                "engine_swap_DDU_hi_rejected": content.count("engine swap (hi path creation) with"),
            }

            for key, value in counts.items():
                totals[key] += value

    for key, value in totals.items():
        print(f"{key}: {value}")
    print(f"="*55)