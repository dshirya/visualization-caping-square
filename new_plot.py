import os
import re
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
from adjustText import adjust_text
import math

# Global settings
PLOT_FOLDER = "plots"
FILE_EXTENSION = ".png"
DPI = 500
BBOX_INCHES = "tight"
TEXT_SIZE = 20

def save_plot(filename, fig, folder=PLOT_FOLDER):
    """Saves the plotted figure to a file with a unique name."""
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, f"{filename}{FILE_EXTENSION}")
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(folder, f"{filename}_{counter}{FILE_EXTENSION}")
        counter += 1
    fig.savefig(file_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    print(f"Plot saved as {file_path}")

def parse_formula(formula):
    """
    Parses a chemical formula string using regex.
    Returns a list of tuples (element, count). For example:
      "Mg6"         -> [("Mg", 6)]
      "Cu4Na3"      -> [("Cu", 4), ("Na", 3)]
    If no count is provided, defaults to 1.
    """
    pattern = r"([A-Z][a-z]*)([0-9]*\.?[0-9]*)"
    matches = re.findall(pattern, formula)
    parsed = []
    for element, count in matches:
        if count == "":
            parsed.append((element, 1))
        else:
            try:
                num = float(count)
                if num.is_integer():
                    num = int(num)
                parsed.append((element, num))
            except:
                parsed.append((element, 1))
    return parsed

def get_element_coordinates(symbol, coord_df):
    """
    Retrieves the x, y coordinates for an element symbol from coord_df.
    Expects coord_df to have columns 'Symbol', 'x', and 'y'.
    """
    row = coord_df[coord_df["Symbol"] == symbol]
    if not row.empty:
        return row.iloc[0]["x"], row.iloc[0]["y"]
    return None, None

def plot_whole_periodic_table(ax, coord_df):
    """
    Plots a simple periodic table grid using the coordinates provided in coord_df.
    Each element cell is drawn as a rectangle with the element symbol annotated.
    """
    for _, row in coord_df.iterrows():
        element, x, y = row["Symbol"], row["x"], row["y"]
        rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor="black", 
                             facecolor="none", linewidth=1, zorder=1)
        ax.add_patch(rect)
        ax.text(x, y, element, fontsize=TEXT_SIZE, ha="center", va="center", zorder=2)

def calculate_average_points(df, coord_df):
    """
    For each row in df with num_tctp_caps in allowed_caps,
    calculates the following:
      - The overall weighted average (using original element coordinates)
      - The base (site) marker (the unshifted base element coordinate)
      - The tctp group average (weighted average of shifted tctp markers)
      - The sp group average (weighted average of shifted sp markers)
    
    Also returns the lists of individual shifted marker coordinates for tctp and sp,
    which can be used later for drawing dashed lines.
    
    Returns:
      A DataFrame indexed by the original df row index containing:
         site_marker, overall_avg_x, overall_avg_y,
         tctp_group_avg_x, tctp_group_avg_y,
         sp_group_avg_x, sp_group_avg_y,
         tctp_markers, sp_markers.
    """
   

    results = []
    for idx, row in df.iterrows():
        # if row["Structure Type"] not in allowed_caps:
        #    continue

        # Lists for overall weighted average calculation (using original coordinates)
        overall_orig = []
        overall_weights = []

        # Base (site) element:
        site_marker = None

        # For tctp_sites_processed group:
        tctp_markers = []
        tctp_weights = []

        # For sp_square_formula group:
        sp_markers = []
        sp_weights = []

        # 1. Process the base element from the 'site' column.
        site_formula = str(row["tcsp_center"])                         # Changed from site
        base_list = parse_formula(site_formula)
        if not base_list:
            continue
        base_element = base_list[0][0]  # take the first element symbol
        base_coord = get_element_coordinates(base_element, coord_df)
        if base_coord == (None, None):
            continue
        overall_orig.append(base_coord)
        overall_weights.append(1)
        site_marker = base_coord

        # 2. Process the tctp_sites_processed column.
        tctp_formula = str(row["tcsp_caps_formula"])                       # Changed from tctp_sites_processed
        tctp_list = parse_formula(tctp_formula)
        for elem, cnt in tctp_list:
            orig_coord = get_element_coordinates(elem, coord_df)
            if orig_coord == (None, None):
                continue
            overall_orig.append(orig_coord)
            overall_weights.append(cnt)
            # Shift the marker position for tctp elements.
            marker_coord = (orig_coord[0] - 0.25, orig_coord[1] + 0.25)
            tctp_markers.append(marker_coord)
            tctp_weights.append(cnt)

        # 3. Process the sp_square_formula column.
        sp_formula = str(row["tcsp_prism_formula"])              # Changed from sp_square_formula
        sp_list = parse_formula(sp_formula)
        for elem, cnt in sp_list:
            orig_coord = get_element_coordinates(elem, coord_df)
            if orig_coord == (None, None):
                continue
            overall_orig.append(orig_coord)
            overall_weights.append(cnt)
            # Shift the marker position for sp elements.
            marker_coord = (orig_coord[0] + 0.25, orig_coord[1] + 0.25)
            sp_markers.append(marker_coord)
            sp_weights.append(cnt)

        # --- Compute the overall weighted average coordinate (using original coordinates) ---
        overall_arr = np.array(overall_orig)  # shape (N, 2)
        overall_w_arr = np.array(overall_weights)
        if overall_w_arr.sum() == 0:
            continue
        overall_avg = np.average(overall_arr, axis=0, weights=overall_w_arr)

        # --- Compute group averages for tctp_sites_processed and sp_square_formula ---
        tctp_group_avg = None
        if tctp_markers:
            tctp_arr = np.array(tctp_markers)
            tctp_w_arr = np.array(tctp_weights)
            tctp_group_avg = np.average(tctp_arr, axis=0, weights=tctp_w_arr)
        sp_group_avg = None
        if sp_markers:
            sp_arr = np.array(sp_markers)
            sp_w_arr = np.array(sp_weights)
            sp_group_avg = np.average(sp_arr, axis=0, weights=sp_w_arr)

        results.append({
            "index": idx,
            "site_marker": site_marker,
            "overall_avg_x": overall_avg[0],
            "overall_avg_y": overall_avg[1],
            "tctp_group_avg_x": tctp_group_avg[0] if tctp_group_avg is not None else None,
            "tctp_group_avg_y": tctp_group_avg[1] if tctp_group_avg is not None else None,
            "sp_group_avg_x": sp_group_avg[0] if sp_group_avg is not None else None,
            "sp_group_avg_y": sp_group_avg[1] if sp_group_avg is not None else None,
            "tctp_markers": tctp_markers,
            "sp_markers": sp_markers,
        })
    
    avg_df = pd.DataFrame(results).set_index("index")
    return avg_df

def plot_periodic_table_with_structures(df, coord_df, avg_df):
    """
    For each unique Structure Type in df, plots the full periodic table (using coord_df) as a background
    and overlays structure data from df (with precomputed averages from avg_df).
    
    For each structure:
      - The base element (from tcsp_center) is plotted in blue.
      - Elements from tcsp_caps_formula are plotted in red.
      - Elements from tcsp_prism_formula are plotted in green.
      - Connecting lines are drawn between the markers:
           • A solid line from the base (site) marker to the overall weighted average.
           • For caps and square groups: dashed lines from individual markers to the group average,
             then a solid line from the group average to the overall average.
      - A rectangle patch is added for each element (with the color corresponding to its group)
        without duplicating a patch for the same coordinate/color.
    
    The plot title is set using the same formatting as the legend labels,
    and the output file name is derived solely from the formula (the part before "-type").
    """
    import re
    import matplotlib.pyplot as plt

    # --- Helper functions for formatting (for titles, using the same preprocessing as for legend labels) ---
    def format_formula(formula):
        """
        Converts element counts in a formula to subscript.
        E.g., "La11Ru2Al6" becomes "La$_{11}$Ru$_{2}$Al$_{6}$".
        """
        def repl(match):
            element = match.group(1)
            number = match.group(2)
            return f"{element}$_{{{number}}}$"
        return re.sub(r"([A-Z][a-z]*)(\d+)", repl, formula)

    def format_symmetry(sym):
        """
        Italicizes the symmetry group. For each letter immediately followed by digits,
        only the digits after the first digit are converted to subscript.
        
        E.g., "Pbam" becomes "$\mathit{Pbam}$" and "I41/amd" becomes "$\mathit{I4_{1}/amd}$".
        """
        def repl(match):
            letter = match.group(1)
            digits = match.group(2)
            if len(digits) >= 2:
                return f"{letter}{digits[0]}_{{{digits[1:]}}}"
            else:
                return f"{letter}{digits}"
        formatted = re.sub(r"([A-Za-z])(\d+)", repl, sym)
        return r'$\mathit{' + formatted + '}$'

    def format_structure_type_for_title(stype):
        """
        Splits a structure type string of the form "Formula-type Symmetry" into its parts,
        applies formatting to each, and then recombines them.
        
        E.g., "Ru4Al3B2-type P4/mmm" becomes "Ru$_{4}$Al$_{3}$B$_{2}$-type $\mathit{P4/mmm}$".
        """
        parts = re.split(r"-type\s*", stype)
        if len(parts) == 2:
            formula_part, symmetry_part = parts
        else:
            return stype  # if not in the expected format, return original
        formatted_formula = format_formula(formula_part)
        formatted_symmetry = format_symmetry(symmetry_part)
        return f"{formatted_formula}-type {formatted_symmetry}"
    
    # Get the unique structure types.
    unique_struct_types = df["Structure Type"].unique()
    
    # Loop over each structure type.
    for struct_type in unique_struct_types:
        # Subset the data for this structure type.
        subset_df = df[df["Structure Type"] == struct_type]
        subset_avg_df = avg_df.loc[subset_df.index].dropna(how="all")  # ensure we have avg data
        
        # Set figure dimensions based on the periodic table coordinates.
        x_min, x_max = coord_df["x"].min(), coord_df["x"].max()
        y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
        fig_width = (x_max - x_min) * 0.8
        fig_height = (y_max - y_min) * 0.8
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Plot the full periodic table as the background.
        periodic_texts = plot_whole_periodic_table(ax, coord_df)
        
        # Dictionaries to track rectangle patches added per coordinate.
        rectangle_counts = {}
        applied_colors = {}
        
        # List to collect structure text objects (if you wish to adjust them later).
        structure_texts = []
        
        # Process each row (structure) for this structure type.
        for idx, row in subset_df.iterrows():
            if idx not in subset_avg_df.index:
                continue  # Skip rows without computed averages
            
            size = 80
            avg_data = subset_avg_df.loc[idx]
            overall_avg = (avg_data["overall_avg_x"], avg_data["overall_avg_y"])
            site_marker = avg_data["site_marker"]

            # --- Plot markers and connecting lines ---
            # Plot the base (site) marker in blue (with vertical shift) and its connecting line.
            ax.scatter(site_marker[0], site_marker[1] - 0.25, color="#0348a1", s=size, zorder=5,
                       alpha=0.3, edgecolors="none")
            ax.plot([site_marker[0], overall_avg[0]],
                    [site_marker[1] - 0.25, overall_avg[1]],
                    color="#0348a1", linestyle="-", zorder=4, alpha=0.5, linewidth=0.8)
            
            # Process caps group (from tcsp_caps_formula).
            tctp_markers = avg_data["tctp_markers"]
            if tctp_markers:
                tctp_group_avg = (avg_data["tctp_group_avg_x"], avg_data["tctp_group_avg_y"])
                for marker in tctp_markers:
                    ax.plot([marker[0], tctp_group_avg[0]],
                            [marker[1], tctp_group_avg[1]],
                            color="#c3121e", linestyle="--", zorder=4, alpha=0.5, linewidth=0.8)
                ax.scatter(tctp_group_avg[0], tctp_group_avg[1], color="#c3121e", s=size, zorder=7,
                           alpha=0.3, edgecolors="none")
                ax.plot([tctp_group_avg[0], overall_avg[0]],
                        [tctp_group_avg[1], overall_avg[1]],
                        color="#c3121e", linestyle="-", zorder=4, alpha=0.5, linewidth=0.8)
            
            # Process square group (from tcsp_prism_formula).
            sp_markers = avg_data["sp_markers"]
            if sp_markers:
                sp_group_avg = (avg_data["sp_group_avg_x"], avg_data["sp_group_avg_y"])
                for marker in sp_markers:
                    ax.plot([marker[0], sp_group_avg[0]],
                            [marker[1], sp_group_avg[1]],
                            color="#027608", linestyle="--", zorder=4, alpha=0.5, linewidth=0.8)
                ax.scatter(sp_group_avg[0], sp_group_avg[1], color="#027608", s=size, zorder=7,
                           alpha=0.3, edgecolors="none")
                ax.plot([sp_group_avg[0], overall_avg[0]],
                        [sp_group_avg[1], overall_avg[1]],
                        color="#027608", linestyle="-", zorder=4, alpha=0.5, linewidth=0.8)
            
            # Plot the overall weighted average marker.
            ax.scatter(overall_avg[0], overall_avg[1], color="black", s=size, zorder=6)
            
            # Optionally, add a text label for the structure type at the overall average position.
            # (Here a simple subscript conversion is applied, though you could also use the full formatting function.)
            label = re.sub(r'(\d+)', 
                           lambda m: m.group(1).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")), 
                           str(row["Structure Type"]).split(",")[0])
            # Uncomment below to add text labels directly:
            # t = ax.text(overall_avg[0], overall_avg[1], label,
            #             fontsize=14, ha="center", va="center", zorder=8)
            # structure_texts.append(t)
            
            # --- Add rectangle patches for each element ---
            def add_rect(coord, color):
                x, y = coord
                if (x, y) not in rectangle_counts:
                    rectangle_counts[(x, y)] = 0
                if (x, y) not in applied_colors:
                    applied_colors[(x, y)] = set()
                if color in applied_colors[(x, y)]:
                    return  # Skip if a patch with this color was already added here.
                count = rectangle_counts[(x, y)]
                shrink = 0.1 * count
                size_rect = 0.92 - shrink
                offset = 0.46 - shrink / 2
                ax.add_patch(plt.Rectangle((x - offset, y - offset), size_rect, size_rect, fill=False,
                                           edgecolor=color, zorder=4, linewidth=2, alpha=0.8))
                applied_colors[(x, y)].add(color)
                rectangle_counts[(x, y)] += 1
            
            # Add rectangle for the base (site) element.
            site_formula = str(row["tcsp_center"])
            base_list = parse_formula(site_formula)
            if base_list:
                base_element = base_list[0][0]
                base_coord = get_element_coordinates(base_element, coord_df)
                if base_coord != (None, None):
                    add_rect(base_coord, "#0348a1")
            
            # Add rectangles for caps group elements.
            tctp_formula = str(row["tcsp_caps_formula"])
            tctp_list = parse_formula(tctp_formula)
            for elem, cnt in tctp_list:
                orig_coord = get_element_coordinates(elem, coord_df)
                if orig_coord is None or orig_coord == (None, None):
                    continue
                add_rect(orig_coord, "#c3121e")
            
            # Add rectangles for square group elements.
            sp_formula = str(row["tcsp_prism_formula"])
            sp_list = parse_formula(sp_formula)
            for elem, cnt in sp_list:
                orig_coord = get_element_coordinates(elem, coord_df)
                if orig_coord is None or orig_coord == (None, None):
                    continue
                add_rect(orig_coord, "#027608")
        
        # Create dummy markers for legend.
        ax.scatter([], [], color="#0348a1", s=30, label="center")
        ax.scatter([], [], color="#c3121e", s=30, label="tctp site")
        ax.scatter([], [], color="#027608", s=30, label="tcsp site")
        
        # Format the plot.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_aspect("equal")
        ax.legend(loc="lower left", 
                  ncol=1, 
                  fontsize=TEXT_SIZE, 
                  frameon=True, 
                  edgecolor="black", 
                  fancybox=False, 
                  framealpha=1, 
                  borderpad=0.3, 
                  labelspacing=0.3, 
                  handletextpad=0.3, 
                  bbox_to_anchor=(0.035, 0.035))
        
        # --- Set the title using the formatted structure type ---
        formatted_title = format_structure_type_for_title(struct_type)
        ax.set_title(formatted_title, fontsize=20)
        
        # Show the plot.
        plt.show()
        
        # --- Create a safe file name based on the formula (part before '-type') ---
        formula_part = struct_type.split("-type")[0].strip()
        safe_filename = formula_part.replace(" ", "_").replace(",", "_")
        # Save the final plot.
        save_plot(safe_filename, fig)

def plot_periodic_table_all_types(df, coord_df, avg_df):
    """
    Plots the full periodic table (using coord_df) as a background and overlays overall weighted average markers 
    from structures in df (with precomputed averages from avg_df).

    For each structure:
      - Only the overall weighted average marker (based on overall_avg_x and overall_avg_y) is plotted.
      - Structure types belonging to the same group share the same color.
      - If a group contains more than one structure type, different marker shapes (circle, triangle, square)
        are used to distinguish them.
      - Markers are unfilled (only the edge is drawn).
      
    A legend is added in the lower right corner with properly formatted structure type labels.
    """
    # --- Helper functions for formatting legend labels ---
    def format_formula(formula):
        """
        Converts element counts in a formula to subscript.
        E.g., "La11Ru2Al6" becomes "La$_{11}$Ru$_{2}$Al$_{6}$".
        """
        def repl(match):
            element = match.group(1)
            number = match.group(2)
            return f"{element}$_{{{number}}}$"
        return re.sub(r"([A-Z][a-z]*)(\d+)", repl, formula)

    def format_symmetry(sym):
        """
        Italicizes the symmetry group. For each letter immediately followed by digits,
        only the digits after the first digit are converted to subscript.
        
        E.g., "Pbam" becomes "$\mathit{Pbam}$" 
              and "I41/amd" becomes "$\mathit{I4_{1}/amd}$"
              while "P4/mbm" remains "$\mathit{P4/mbm}$".
        """
        def repl(match):
            letter = match.group(1)
            digits = match.group(2)
            if len(digits) >= 2:
                # Keep the first digit normal; subscript the rest.
                return f"{letter}{digits[0]}_{{{digits[1:]}}}"
            else:
                return f"{letter}{digits}"
        formatted = re.sub(r"([A-Za-z])(\d+)", repl, sym)
        return r'$\mathit{' + formatted + '}$'

    def format_structure_type_for_legend(stype):
        """
        Splits a structure type string of the form "Formula-type Symmetry" into two parts,
        applies formatting to each, and then recombines them.
        
        Examples:
          "La11Ru2Al6-type Pbam" -> "La$_{11}$Ru$_{2}$Al$_{6}$-type $\mathit{Pbam}$"
          "MgCeSi2-type I41/amd" -> "MgCeSi$_{2}$-type $\mathit{I4_{1}/amd}$"
        """
        parts = re.split(r"-type\s*", stype)
        if len(parts) == 2:
            formula_part, symmetry_part = parts
        else:
            return stype  # if not in expected format, return original
        formatted_formula = format_formula(formula_part)
        formatted_symmetry = format_symmetry(symmetry_part)
        return f"{formatted_formula}-type {formatted_symmetry}"
    
    # --- Define desired groups and marker assignment ---
    # Each list represents a group of structure types.
    desired_order = [
        ["U3Si2-type P4/mbm", "Sr2Pb3-type P4/mbm", "Mo2FeB2-type P4/mbm"],  # Group 1
        ["Tb3Pd2-type Pbam", "Cs2HgSe2-type Pbam"],                           # Group 2
        ["Mg5Si6-type C2/m", "Yb4Mn2Sn5-type C2/m"],                           # Group 3
        ["Ca7Ni4Sn13-type P4/m", "Yb7Co4InGe12-type P4/m"],                     # Group 4
        ["Mg4CuTb-type Cmmm", "Nd11Pd4In9-type Cmmm"],                          # Group 5
        ["Yb5Fe4Al17Si6-type P4/mmm", "Mg5Dy5Fe4Al12Si6-type P4/mmm"],           # Group 6
        ["Zr9Ni2P4-type P4/mbm"],                                               # Group 7
        ["La11Ru2Al6-type Pbam"],                                               # Group 8
        ["Ru4Al3B2-type P4/mmm"],                                                # Group 9
        ["MgCeSi2-type I41/amd"],                                                # Group 10
        ["Er17Ru6Te3-type C2/m"],                                                # Group 11
        ["Tb4RhInGe4-type C2/m"],                                                # Group 12
        ["Cu4Nb5Si4-type I4/m"]                                                  # Group 13
    ]
    
    # For groups with more than one structure type, assign different markers.
    markers = ["o", "^", "s"]  # circle, triangle, square
    
    # Build a mapping: structure type -> (group_number, marker)
    stype_to_group_marker = {}
    for group_idx, group_list in enumerate(desired_order, start=1):
        if len(group_list) == 1:
            stype_to_group_marker[group_list[0]] = (group_idx, "o")
        else:
            for i, stype in enumerate(group_list):
                marker = markers[i % len(markers)]
                stype_to_group_marker[stype] = (group_idx, marker)
    
    # --- Color assignment ---
    # Use a colormap with as many distinct colors as groups.
    num_groups = len(desired_order)
    cmap = plt.get_cmap("nipy_spectral", num_groups)
    group_colors = {group: cmap(group - 1) for group in range(1, num_groups + 1)}
    
    # --- Create the plot ---
    x_min, x_max = coord_df["x"].min(), coord_df["x"].max()
    y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
    fig_width = (x_max - x_min) * 0.8
    fig_height = (y_max - y_min) * 0.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot the periodic table background.
    plot_whole_periodic_table(ax, coord_df)
    
    size = 50  # Marker size
    # For each structure in the dataframe, plot the overall weighted average marker.
    for idx, row in df.iterrows():
        if idx not in avg_df.index:
            continue
        overall_avg = (avg_df.loc[idx]["overall_avg_x"], avg_df.loc[idx]["overall_avg_y"])
        stype = row["Structure Type"]
        if stype in stype_to_group_marker:
            group, marker = stype_to_group_marker[stype]
            color = group_colors[group]
        else:
            group, marker, color = 999, "o", "black"
        # Plot an unfilled marker (only edge color).
        ax.scatter(overall_avg[0], overall_avg[1], facecolors="none", edgecolors=color, 
                   linewidths=2, marker=marker, s=size, zorder=6)
    
    # --- Create legend handles ---
    legend_handles = []
    legend_labels = []
    for stype, (group, marker) in stype_to_group_marker.items():
        color = group_colors[group]
        formatted_label = format_structure_type_for_legend(stype)
        # Create a proxy scatter for the legend that mimics the plot markers
        handle = ax.scatter([], [], s=size, marker=marker, facecolors="none",
                            edgecolors=color, linewidths=2)
        legend_handles.append(handle)
        legend_labels.append(formatted_label)

    ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=12,
            frameon=True, edgecolor="black", fancybox=False, framealpha=1, borderpad=0.5,
            labelspacing=0.5, handletextpad=0.5, bbox_to_anchor=(0.9999, 0.178))
    # --- Format the plot ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")
    
    plt.show()
    
    # Save the final plot.
    save_plot("all_structures", fig)

def plot_periodic_table_all_types_long(df, coord_df, avg_df):
    """
    Plots the full periodic table (using coord_df) as a background and overlays overall weighted average markers 
    from structures in df (with precomputed averages from avg_df).

    For each structure:
      - Only the overall weighted average marker (based on overall_avg_x and overall_avg_y) is plotted.
      - Structure types belonging to the same group share the same color.
      - If a group contains more than one structure type, different marker shapes (circle, triangle, square)
        are used to distinguish them.
      - Markers are unfilled (only the edge is drawn).
      
    A legend is added in the lower right corner with properly formatted structure type labels.
    """
    # --- Helper functions for formatting legend labels ---
    def format_formula(formula):
        """
        Converts element counts in a formula to subscript.
        E.g., "La11Ru2Al6" becomes "La$_{11}$Ru$_{2}$Al$_{6}$".
        """
        def repl(match):
            element = match.group(1)
            number = match.group(2)
            return f"{element}$_{{{number}}}$"
        return re.sub(r"([A-Z][a-z]*)(\d+)", repl, formula)

    def format_symmetry(sym):
        """
        Italicizes the symmetry group. For each letter immediately followed by digits,
        only the digits after the first digit are converted to subscript.
        
        E.g., "Pbam" becomes "$\mathit{Pbam}$" 
              and "I41/amd" becomes "$\mathit{I4_{1}/amd}$"
              while "P4/mbm" remains "$\mathit{P4/mbm}$".
        """
        def repl(match):
            letter = match.group(1)
            digits = match.group(2)
            if len(digits) >= 2:
                # Keep the first digit normal; subscript the rest.
                return f"{letter}{digits[0]}_{{{digits[1:]}}}"
            else:
                return f"{letter}{digits}"
        formatted = re.sub(r"([A-Za-z])(\d+)", repl, sym)
        return r'$\mathit{' + formatted + '}$'

    def format_structure_type_for_legend(stype):
        """
        Splits a structure type string of the form "Formula-type Symmetry" into two parts,
        applies formatting to each, and then recombines them.
        
        Examples:
          "La11Ru2Al6-type Pbam" -> "La$_{11}$Ru$_{2}$Al$_{6}$-type $\mathit{Pbam}$"
          "MgCeSi2-type I41/amd" -> "MgCeSi$_{2}$-type $\mathit{I4_{1}/amd}$"
        """
        parts = re.split(r"-type\s*", stype)
        if len(parts) == 2:
            formula_part, symmetry_part = parts
        else:
            return stype  # if not in expected format, return original
        formatted_formula = format_formula(formula_part)
        formatted_symmetry = format_symmetry(symmetry_part)
        return f"{formatted_formula}-type {formatted_symmetry}"
    
    # --- Define desired groups and marker assignment ---
    # Each list represents a group of structure types.
    desired_order = [
        ["U3Si2-type P4/mbm", "Sr2Pb3-type P4/mbm", "Mo2FeB2-type P4/mbm"],  # Group 1
        ["Tb3Pd2-type Pbam", "Cs2HgSe2-type Pbam"],                           # Group 2
        ["Mg5Si6-type C2/m", "Yb4Mn2Sn5-type C2/m"],                           # Group 3
        ["Ca7Ni4Sn13-type P4/m", "Yb7Co4InGe12-type P4/m"],                     # Group 4
        ["Mg4CuTb-type Cmmm", "Nd11Pd4In9-type Cmmm"],                          # Group 5
        ["Yb5Fe4Al17Si6-type P4/mmm", "Mg5Dy5Fe4Al12Si6-type P4/mmm"],           # Group 6
        ["Zr9Ni2P4-type P4/mbm"],                                               # Group 7
        ["La11Ru2Al6-type Pbam"],                                               # Group 8
        ["Ru4Al3B2-type P4/mmm"],                                                # Group 9
        ["MgCeSi2-type I41/amd"],                                                # Group 10
        ["Er17Ru6Te3-type C2/m"],                                                # Group 11
        ["Tb4RhInGe4-type C2/m"],                                                # Group 12
        ["Cu4Nb5Si4-type I4/m"]                                                  # Group 13
    ]
    
    # For groups with more than one structure type, assign different markers.
    markers = ["o", "^", "s"]  # circle, triangle, square
    
    # Build a mapping: structure type -> (group_number, marker)
    stype_to_group_marker = {}
    for group_idx, group_list in enumerate(desired_order, start=1):
        if len(group_list) == 1:
            stype_to_group_marker[group_list[0]] = (group_idx, "o")
        else:
            for i, stype in enumerate(group_list):
                marker = markers[i % len(markers)]
                stype_to_group_marker[stype] = (group_idx, marker)
    
    # --- Color assignment ---
    # Use a colormap with as many distinct colors as groups.
    num_groups = len(desired_order)
    cmap = plt.get_cmap("nipy_spectral", num_groups)
    group_colors = {group: cmap(group - 1) for group in range(1, num_groups + 1)}
    
    # --- Create the plot ---
    x_min, x_max = coord_df["x"].min(), coord_df["x"].max()
    y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
    fig_width = (x_max - x_min) * 0.8
    fig_height = (y_max - y_min) * 0.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot the periodic table background.
    plot_whole_periodic_table(ax, coord_df)
    
    size = 50  # Marker size
    # For each structure in the dataframe, plot the overall weighted average marker.
    for idx, row in df.iterrows():
        if idx not in avg_df.index:
            continue
        overall_avg = (avg_df.loc[idx]["overall_avg_x"], avg_df.loc[idx]["overall_avg_y"])
        stype = row["Structure Type"]
        if stype in stype_to_group_marker:
            group, marker = stype_to_group_marker[stype]
            color = group_colors[group]
        else:
            group, marker, color = 999, "o", "black"
        # Plot an unfilled marker (only edge color).
        ax.scatter(overall_avg[0], overall_avg[1], facecolors="none", edgecolors=color, 
                   linewidths=2, marker=marker, s=size, zorder=6)
    
    # --- Create legend handles ---
    legend_handles = []
    legend_labels = []
    for stype, (group, marker) in stype_to_group_marker.items():
        color = group_colors[group]
        formatted_label = format_structure_type_for_legend(stype)
        # Create a proxy scatter for the legend that mimics the plot markers
        handle = ax.scatter([], [], s=size, marker=marker, facecolors="none",
                            edgecolors=color, linewidths=2)
        legend_handles.append(handle)
        legend_labels.append(formatted_label)

    
    ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=12,
              frameon=True, edgecolor="black", fancybox=False, framealpha=1, borderpad=0.5,
              labelspacing=0.5, handletextpad=0.5, bbox_to_anchor=(1.2, 0.01))
    
    # --- Format the plot ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")
    
    plt.show()
    
    # Save the final plot.
    save_plot("all_structures", fig)

def plot_periodic_table_all_types_kde(df, coord_df, avg_df):
    """
    Plots the full periodic table (using coord_df) as a background and overlays 
    filled KDE density fields for overall weighted average points from structures in df 
    (with precomputed averages from avg_df) using seaborn.kdeplot.
    
    For each group of structure types:
      - Overall weighted average coordinates are grouped and a KDE density field 
        is computed using seaborn.kdeplot.
      - Structure types belonging to the same group share the same color.
      - For groups with only one point, a fallback marker is plotted.
      
    A vertical legend is added in the lower right corner with one structure type per line.
    """
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.patches as mpatches

    # --- Helper functions for formatting legend labels ---
    def format_formula(formula):
        def repl(match):
            element = match.group(1)
            number = match.group(2)
            return f"{element}$_{{{number}}}$"
        return re.sub(r"([A-Z][a-z]*)(\d+)", repl, formula)

    def format_symmetry(sym):
        def repl(match):
            letter = match.group(1)
            digits = match.group(2)
            if len(digits) >= 2:
                return f"{letter}{digits[0]}_{{{digits[1:]}}}"
            else:
                return f"{letter}{digits}"
        formatted = re.sub(r"([A-Za-z])(\d+)", repl, sym)
        return r'$\mathit{' + formatted + '}$'

    def format_structure_type_for_legend(stype):
        parts = re.split(r"-type\s*", stype)
        if len(parts) == 2:
            formula_part, symmetry_part = parts
        else:
            return stype
        formatted_formula = format_formula(formula_part)
        formatted_symmetry = format_symmetry(symmetry_part)
        return f"{formatted_formula}-type {formatted_symmetry}"
    
    # --- Define desired groups and mapping: structure type -> group number ---
    desired_order = [
        ["U3Si2-type P4/mbm", "Sr2Pb3-type P4/mbm", "Mo2FeB2-type P4/mbm"],  # Group 1
        ["Tb3Pd2-type Pbam", "Cs2HgSe2-type Pbam"],                           # Group 2
        ["Mg5Si6-type C2/m", "Yb4Mn2Sn5-type C2/m"],                           # Group 3
        ["Ca7Ni4Sn13-type P4/m", "Yb7Co4InGe12-type P4/m"],                     # Group 4
        ["Mg4CuTb-type Cmmm", "Nd11Pd4In9-type Cmmm"],                          # Group 5
        ["Yb5Fe4Al17Si6-type P4/mmm", "Mg5Dy5Fe4Al12Si6-type P4/mmm"],           # Group 6
        ["Zr9Ni2P4-type P4/mbm"],                                               # Group 7
        ["La11Ru2Al6-type Pbam"],                                               # Group 8
        ["Ru4Al3B2-type P4/mmm"],                                                # Group 9
        ["MgCeSi2-type I41/amd"],                                                # Group 10
        ["Er17Ru6Te3-type C2/m"],                                                # Group 11
        ["Tb4RhInGe4-type C2/m"],                                                # Group 12
        ["Cu4Nb5Si4-type I4/m"]                                                  # Group 13
    ]
    stype_to_group = {}
    for group_idx, group_list in enumerate(desired_order, start=1):
        for stype in group_list:
            stype_to_group[stype] = group_idx

    # --- Color assignment ---
    num_groups = len(desired_order)
    cmap = plt.get_cmap("nipy_spectral", num_groups)
    group_colors = {group: cmap(group - 1) for group in range(1, num_groups + 1)}
    
    # --- Determine plotting limits from the periodic table coordinates ---
    x_min, x_max = coord_df["x"].min(), coord_df["x"].max()
    y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
    fig_width = (x_max - x_min) * 0.8
    fig_height = (y_max - y_min) * 0.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot the periodic table background.
    plot_whole_periodic_table(ax, coord_df)
    
    # --- Group overall average points by group ---
    group_points = {group: [] for group in range(1, num_groups + 1)}
    for idx, row in df.iterrows():
        if idx not in avg_df.index:
            continue
        overall_avg_x = avg_df.loc[idx]["overall_avg_x"]
        overall_avg_y = avg_df.loc[idx]["overall_avg_y"]
        stype = row["Structure Type"]
        if stype in stype_to_group:
            group = stype_to_group[stype]
            group_points[group].append((overall_avg_x, overall_avg_y))
    
    # --- Plot KDE for each group using seaborn.kdeplot ---
    for group, points in group_points.items():
        color = group_colors[group]
        if len(points) >= 2:
            points_arr = np.array(points)
            xs = points_arr[:, 0]
            ys = points_arr[:, 1]
            sns.kdeplot(
                x=xs, y=ys, ax=ax, fill=True, alpha=0.5, linewidth=2, color=color
            )
        elif len(points) == 1:
            # Fallback: plot a single point if only one exists.
            x_val, y_val = points[0]
            ax.scatter(x_val, y_val, facecolors="none", edgecolors=color,
                       linewidths=2, marker="o", s=50, zorder=6)
    
    # --- Create vertical legend handles: one entry per structure type in desired order ---
    legend_handles = []
    for group_list in desired_order:
        for stype in group_list:
            group = stype_to_group[stype]
            color = group_colors[group]
            formatted_label = format_structure_type_for_legend(stype)
            handle = mpatches.Patch(color=color, label=formatted_label)
            legend_handles.append(handle)
    
    ax.legend(handles=legend_handles, loc="lower right", fontsize=12,
              frameon=True, edgecolor="black", fancybox=False, framealpha=1, borderpad=0.5,
              labelspacing=0.5, handletextpad=0.5, ncol=1)
    
    # --- Format the plot ---
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")
    
    plt.show()
    
    # Save the final plot.
    save_plot("all_structures_kde", fig)