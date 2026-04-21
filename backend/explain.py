from from_list_to_shapy_values import get_shaped_values
from utils_2 import from_group_to_shape

def run_explanations(df, groups, selected_attributes, k_min):
    # Helper flow the Dexer repo already uses for Shapley-based explanations.
    return from_group_to_shape(groups, df, selected_attributes, k_min)