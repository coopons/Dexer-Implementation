from algorithms import iter_td_prop_bounds as prop_mod
from algorithms import iter_td_global_bounds as global_mod


def run_detection(df, selected_attributes, threshold, alpha, k_min, k_max, mode="prop"):
    if mode == "prop":
        detector = getattr(prop_mod, "GraphTraverseProportional", None) or getattr(prop_mod, "GraphTraverse", None)
    else:
        detector = getattr(global_mod, "GraphTraverseGlobalBounds", None) or getattr(global_mod, "GraphTraverse", None)

    if detector is None:
        raise ImportError("Could not find the detection function in the algorithm module")

    if mode == "prop":
        groups, visited, elapsed = detector(
            df,
            selected_attributes,
            threshold,
            alpha,
            k_min,
            k_max,
            60 * 10,  # time limit
        )
    else:
        lower_bound_value = int(alpha)
        lowerbounds = [lower_bound_value] * max(1, int(k_max) - int(k_min))
        groups, visited, elapsed = detector(
            df,
            selected_attributes,
            threshold,
            lowerbounds,
            k_min,
            k_max,
            60 * 10,  # time limit
        )

    return {
        "groups": groups,
        "visited": visited,
        "elapsed": elapsed,
    }