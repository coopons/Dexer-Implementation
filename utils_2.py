# This file is adapted from the Dexer project.
# Original source: Detection of Groups with Biased Representation in Ranking
# https://github.com/talni/Detection-of-Groups-with-Biased-Representation-in-Ranking

import json
import pandas as pd
from flask import Flask, request, render_template, redirect, flash
from matplotlib import pyplot as plt
from numba import typeof
from werkzeug.utils import secure_filename
from from_list_to_shapy_values import string2num, shapley_values_att_value_seperated, get_shaped_values, \
    plot_average_shap_value_of_group

from algorithms.iter_td_global_bounds\
    import GraphTraverse as GraphTraverseNonProportional
from algorithms.iter_td_prop_bounds\
    import GraphTraverse as GraphTraverseProportional


def from_group_to_shape(pattern_treated_unfairly_lowerbound, ranked_data, selected_attributes, k_min=0):

    ans = {}
    count = k_min
    shaped_values = get_shaped_values(ranked_data, selected_attributes)
    for item in pattern_treated_unfairly_lowerbound:
        print("******************", count)
        group = string2num(item)
        fig, axis = plt.subplots(1, 1,figsize=(14, 7))
        shaped_values_per_group = plot_average_shap_value_of_group(
            ranked_data, group, selected_attributes, selected_attributes, shaped_values, axis)
        ans_temp = shaped_values_per_group.to_dict(orient='records')
        print("_______: ",ans_temp)
        ans[count] = ans_temp
        count = count + 1
    return ans;
