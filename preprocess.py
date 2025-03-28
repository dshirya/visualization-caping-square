import re
import ast

def process_tctp_sites(tctp_sites):
    """
    Processes the tctp_sites data and returns a string with summed counts for each element.
    
    Parameters:
    -----------
    tctp_sites : list or str
        A list of lists (or tuple) where each inner list contains an element (with an index)
        and its corresponding count. For example: [['Ge1', 2], ['Ge2', 2]].
        If provided as a string, it will be converted using ast.literal_eval.
    
    Returns:
    --------
    result : str
        A string where each unique element (with its trailing numeric index removed)
        is concatenated with the sum of its counts. For the example above, the output is 'Ge4'.
    
    Raises:
    -------
    ValueError
        If the input cannot be processed correctly.
    """
    # Convert string representation to list if necessary
    if isinstance(tctp_sites, str):
        try:
            tctp_sites = ast.literal_eval(tctp_sites)
        except Exception as e:
            raise ValueError(f"Error converting tctp_sites string to list: {e}")
    
    totals = {}
    order = []  # To preserve the order of first occurrence
    
    for item in tctp_sites:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            element_with_index, count = item
            # Use regex to extract the alphabetic part (element symbol) and drop the index
            match = re.match(r"([A-Za-z]+)", element_with_index)
            if match:
                element = match.group(1)
                if element not in totals:
                    totals[element] = 0
                    order.append(element)
                totals[element] += count
            else:
                raise ValueError(f"Could not extract element from {element_with_index}")
        else:
            raise ValueError("Each item in tctp_sites should be a list or tuple of length 2.")
    
    # Build the resulting string preserving the original order
    result = ""
    for element in order:
        result += f"{element}{totals[element]}"
    
    return result
