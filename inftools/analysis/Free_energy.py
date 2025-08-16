import os
import numpy as np
import matplotlib.pyplot as plt

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


def update_histogram(data, factor, histogram, Minx, Miny, dx, dy):
    if Miny is not None and dy is not None:
        x = data[0]
        y = data[1]

        ix = ((x - Minx) / dx).astype(int)
        iy = ((y - Miny) / dy).astype(int)

        np.add.at(histogram, (ix, iy), factor)

    else:
        x = data if data.ndim == 1 else data[:,0] # make sure x is one dimensional
        ix = ((x - Minx) / dx).astype(int)
        np.add.at(histogram, ix, factor)

    return histogram


def calculate_free_energy(trajlabels, WFtot, Trajdir, outfolder, histo_stuff, lm1, lA, lB, sym):
    print(f"="*65)
    print(f"We are now going to perform the Landau Free Energy calculations:")
    print(f"Lambda_minus_one is {lm1}.") 
    print(f"==========================")   
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
        histogram = update_histogram(data, factor, histogram, Minx, Miny, dx, dy)

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
        print(f"Expected xi value: {(histogram[index_lA-1] / histogram[index_lA]):.6f}")
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
    np.savetxt(os.path.join(outfolder, "cond_free_energy.txt"), conditional_free_energy)
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
    plt.savefig(os.path.join(outfolder, "Free_Energy.png"), dpi=300)
    plt.close()

    if sym:
        histogram_sym = np.append(histogram, np.zeros(index_lA, dtype=histogram.dtype))
        sym_histo = histogram_sym + histogram_sym[::-1]
        max_value = np.max(sym_histo)
        sym_histo /= max_value
        sym_free_energy = -np.log(sym_histo)
        np.savetxt(os.path.join(outfolder, "sym_free_energy.txt"), sym_free_energy)
        xval_sym = [Minx + 0.5 * dx + i * dx for i in range(Nbinsx+index_lA)]
        np.savetxt(os.path.join(outfolder, "sym_xval.txt"), xval_sym)
        plt.figure(figsize=(6, 4))
        plt.plot(xval, conditional_free_energy, '-o', label="Conditional A→B")
        plt.plot(xval_sym[index_lA:], conditional_free_energy[::-1], '-o', label="Conditional B→A")
        plt.plot(xval_sym, sym_free_energy, '-o', label="Symmetrized")
        plt.xlabel("Order parameter (Å)")
        plt.ylabel("Free energy (kBT)")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outfolder, "Free_Energy_sym.png"), dpi=300)
        plt.close()