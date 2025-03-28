import os
import re
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
from adjustText import adjust_text

# Global settings
PLOT_FOLDER = "plots"
FILE_EXTENSION = ".png"
DPI = 500
BBOX_INCHES = "tight"
TEXT_SIZE = 16

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

def calculate_average_points(df, coord_df, allowed_caps=[1, 2, 3, 4]):
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
        if row["Structure Type"] not in allowed_caps:
           continue

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

def plot_periodic_table_with_structures(df, coord_df, avg_df, allowed_caps=[1, 2, 3, 4]):
    """
    Plots the full periodic table (using coord_df) as a background and overlays structure data from df.
    
    For each row in df (with num_tctp_caps in allowed_caps), the function uses pre-calculated averages 
    from avg_df to:
      - Plot the base element (site) in blue.
      - Plot tctp elements (caps) in red and their group average.
      - Plot sp_square_formula elements (square) in green and their group average.
      - Draw lines connecting markers:
           • A solid line from the site marker to the overall weighted average.
           • For tctp and sp groups: dashed lines from individual markers to the group average,
             then a solid line from the group average to the overall average.
             
    Additionally, for each element (from site, tctp, and sp columns), a rectangle patch is added.
    The rectangle size shrinks if that element already has a patch in the same color.
    """
    # Fixed rectangle properties (no global props used)
    initial_rect_size = 0.92      # initial rectangle size
    shrink_factor_rect = 0.1      # shrink factor for each additional patch on same coordinate
    initial_rect_offset = 0.46    # initial offset for rectangle placement

    # Set figure dimensions based on the periodic table coordinates.
    x_min, x_max = coord_df["x"].min(), coord_df["x"].max()
    y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
    fig_width = (x_max - x_min) * 0.8
    fig_height = (y_max - y_min) * 0.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Plot the full periodic table as the background and collect its text objects.
    periodic_texts = plot_whole_periodic_table(ax, coord_df)
    
    # Dictionaries to track how many patches have been added per element coordinate,
    # and which colors have been applied.
    rectangle_counts = {}
    applied_colors = {}
    
    # List to collect all structure text objects.
    structure_texts = []

    # Process each structure in df.
    for idx, row in df.iterrows():
        if row["Structure Type"] not in allowed_caps:
            continue
        if idx not in avg_df.index:
            continue  # Skip rows without computed averages               #commented out to plot all structures
        
        # Retrieve precomputed averages and markers.
        avg_data = avg_df.loc[idx]
        overall_avg = (avg_data["overall_avg_x"], avg_data["overall_avg_y"])
        site_marker = avg_data["site_marker"]

        # --- Plot markers and connecting lines ---
        # Plot the base (site) marker in blue (with vertical shift) and its connecting line.
        ax.scatter(site_marker[0], site_marker[1] - 0.25, color="blue", s=20, zorder=5,
                   alpha=0.3, edgecolors="none")
        ax.plot([site_marker[0], overall_avg[0]],
                [site_marker[1] - 0.25, overall_avg[1]],
                color="blue", linestyle="-", zorder=4, alpha=0.5, linewidth=0.8)
        
        # Process tctp_sites_processed group.
        tctp_markers = avg_data["tctp_markers"]
        if tctp_markers:
            tctp_group_avg = (avg_data["tctp_group_avg_x"], avg_data["tctp_group_avg_y"])
            for marker in tctp_markers:
                ax.plot([marker[0], tctp_group_avg[0]],
                        [marker[1], tctp_group_avg[1]],
                        color="red", linestyle="--", zorder=4, alpha=0.5, linewidth=0.8)
            ax.scatter(tctp_group_avg[0], tctp_group_avg[1], color="red", s=20, zorder=7,
                       alpha=0.3, edgecolors="none")
            ax.plot([tctp_group_avg[0], overall_avg[0]],
                    [tctp_group_avg[1], overall_avg[1]],
                    color="red", linestyle="-", zorder=4, alpha=0.5, linewidth=0.8)
        
        # Process sp_square_formula group.
        sp_markers = avg_data["sp_markers"]
        if sp_markers:
            sp_group_avg = (avg_data["sp_group_avg_x"], avg_data["sp_group_avg_y"])
            for marker in sp_markers:
                ax.plot([marker[0], sp_group_avg[0]],
                        [marker[1], sp_group_avg[1]],
                        color="green", linestyle="--", zorder=4, alpha=0.5, linewidth=0.8)
            ax.scatter(sp_group_avg[0], sp_group_avg[1], color="green", s=20, zorder=7,
                       alpha=0.3, edgecolors="none")
            ax.plot([sp_group_avg[0], overall_avg[0]],
                    [sp_group_avg[1], overall_avg[1]],
                    color="green", linestyle="-", zorder=4, alpha=0.5, linewidth=0.8)
        
        # Plot the overall weighted average marker.
        ax.scatter(overall_avg[0], overall_avg[1], color="black", s=20, zorder=6)
        structure_type = str(row["Structure Type"]).split(",")[0]
        
        # Helper to convert digits to subscript (using Unicode subscripts).
        def to_subscript(text):
            normal = "0123456789"
            subscript = "₀₁₂₃₄₅₆₇₈₉"
            trans = str.maketrans(''.join(normal), ''.join(subscript))
            return text.translate(trans)
        
        # Convert structure_type digits to subscripts.
        structure_type = re.sub(r'(\d+)', lambda m: to_subscript(m.group(1)), structure_type)
        
        # # Instead of adjusting right away, store the text object.
        # t = ax.text(overall_avg[0], overall_avg[1], structure_type,
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
            shrink = shrink_factor_rect * count
            size = initial_rect_size - shrink
            offset = initial_rect_offset - shrink / 2
            ax.add_patch(plt.Rectangle((x - offset, y - offset), size, size, fill=False,
                                       edgecolor=color, zorder=4, linewidth=2, alpha=0.5))
            applied_colors[(x, y)].add(color)
            rectangle_counts[(x, y)] += 1

        # Add rectangle for the base (site) element.
        site_formula = str(row["tcsp_center"])                         # changed from site
        base_list = parse_formula(site_formula)
        if base_list:
            base_element = base_list[0][0]
            base_coord = get_element_coordinates(base_element, coord_df)
            if base_coord != (None, None):
                add_rect(base_coord, "blue")
        
        # Add rectangles for tctp_sites_processed elements.
        tctp_formula = str(row["tcsp_caps_formula"])                           # changed from tctp_sites_processed 
        tctp_list = parse_formula(tctp_formula)
        for elem, cnt in tctp_list:
            orig_coord = get_element_coordinates(elem, coord_df)
            if orig_coord == (None, None):
                continue
            add_rect(orig_coord, "red")
        
        # Add rectangles for sp_square_formula elements.
        sp_formula = str(row["tcsp_prism_formula"])         #  changed from sp_square_formula
        sp_list = parse_formula(sp_formula)
        for elem, cnt in sp_list:
            orig_coord = get_element_coordinates(elem, coord_df)
            if orig_coord == (None, None):
                continue
            add_rect(orig_coord, "green")
    
    # Create dummy markers for legend.
    ax.scatter([], [], color="blue", s=30, label="Site")
    ax.scatter([], [], color="red", s=30, label="Caps")
    ax.scatter([], [], color="green", s=30, label="Square")
    
    # --- Adjust structure text labels as a group ---
    # Draw the canvas to update text extents.
    plt.draw()
    # adjust_text(
    #     structure_texts,
    #     ax=ax,
    #     objects=periodic_texts,  # Prevent overlap with periodic table labels.
    #     arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
    #     autoalign=True,
    #     only_move={'text': 'y+', 'points': 'y+'},
    #     expand=(1.5, 1.5),
    #     force_text=0.8,
    #     force_static=0.5,
    #     force_pull=0.1,
    #     iter_lim=500
    # )
    
    # Format the plot.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("4 caps", fontsize=20)
    plt.show()
    
    # Save the final plot.
    save_plot("periodic_table_structures", fig)