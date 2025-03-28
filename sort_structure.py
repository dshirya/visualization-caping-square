def sort_dataframe_by_structure_type(df, col="Structure Type"):
    """
    Sorts the dataframe so that rows with structure types appear in a custom order
    according to the new structure type list, and creates a new column with the group number.
    
    The custom order is defined as:
    
      Group 1:
         U3Si2-type P4/mbm
         Sr2Pb3-type P4/mbm
         Mo2FeB2-type P4/mbm
      Group 2:
         Tb3Pd2-type Pbam
         Cs2HgSe2-type Pbam
      Group 3:
         Mg5Si6-type C2/m
         Yb4Mn2Sn5-type C2/m
      Group 4:
         Ca7Ni4Sn13-type P4/m
         Yb7Co4InGe12-type P4/m
      Group 5:
         Mg4CuTb-type Cmmm
         Nd11Pd4In9-type Cmmm
      Group 6:
         Yb5Fe4Al17Si6-type P4/mmm
         Mg5Dy5Fe4Al12Si6-type P4/mmm
      Group 7:
         Zr9Ni2P4-type P4/mbm
      Group 8:
         La11Ru2Al6-type Pbam
      Group 9:
         Ru4Al3B2-type P4/mmm
      Group 10:
         MgCeSi2-type I41/amd
      Group 11:
         Er17Ru6Te3-type C2/m
      Group 12:
         Tb4RhInGe4-type C2/m
      Group 13:
         Cu4Nb5Si4-type I4/m
         
    For any structure type not found in the above list, it is assigned a group number of 999
    and sorted alphabetically at the end.
    
    Parameters:
      df (pd.DataFrame): The input dataframe.
      col (str): The column name that holds the structure type strings.
      
    Returns:
      pd.DataFrame: The sorted dataframe with a new column "Group Number".
    """
    
    # Define the desired order groups.
    desired_order = [
        ["U3Si2-type P4/mbm", "Sr2Pb3-type P4/mbm", "Mo2FeB2-type P4/mbm"],
        ["Tb3Pd2-type Pbam", "Cs2HgSe2-type Pbam"],
        ["Mg5Si6-type C2/m", "Yb4Mn2Sn5-type C2/m"],
        ["Ca7Ni4Sn13-type P4/m", "Yb7Co4InGe12-type P4/m"],
        ["Mg4CuTb-type Cmmm", "Nd11Pd4In9-type Cmmm"],
        ["Yb5Fe4Al17Si6-type P4/mmm", "Mg5Dy5Fe4Al12Si6-type P4/mmm"],
        ["Zr9Ni2P4-type P4/mbm"],
        ["La11Ru2Al6-type Pbam"],
        ["Ru4Al3B2-type P4/mmm"],
        ["MgCeSi2-type I41/amd"],
        ["Er17Ru6Te3-type C2/m"],
        ["Tb4RhInGe4-type C2/m"],
        ["Cu4Nb5Si4-type I4/m"]
    ]
    
    # Build a mapping: normalized structure type -> (group, order within group)
    order_mapping = {}
    for group_idx, group in enumerate(desired_order, start=1):
        for order_idx, stype in enumerate(group):
            order_mapping[stype] = (group_idx, order_idx)
    
    def normalize_structure_type(s):
        """
        Normalizes the structure type string.
        For example, it converts:
          "Mg4CuTb,oS48,65" to "Mg4CuTb-type P4/mbm" if needed.
        Here we assume that any commas are replaced by a dash.
        """
        return s.strip().replace(",", "-")
    
    # Work on a copy of the dataframe.
    df = df.copy()
    # Create a new column with normalized structure types.
    df["Normalized Structure Type"] = df[col].apply(normalize_structure_type)
    
    # Create a new column for group number.
    def get_group(s):
        return order_mapping[s][0] if s in order_mapping else 999
    df["Group Number"] = df["Normalized Structure Type"].apply(get_group)
    
    def sort_key(s):
        """
        Returns a sort key tuple for each normalized structure type.
        If found in the mapping, returns (group, order); otherwise (999, s) so it sorts alphabetically at the end.
        """
        return order_mapping[s] if s in order_mapping else (999, s)
    
    # Sort the dataframe by the custom sort key.
    df_sorted = df.sort_values(by="Normalized Structure Type", key=lambda col: col.map(sort_key))
    return df_sorted