# This file is adapted from the Dexer project.
# Original source: Detection of Groups with Biased Representation in Ranking
# https://github.com/talni/Detection-of-Groups-with-Biased-Representation-in-Ranking

"""
Naive algorithm for group detection in ranking
fairness definition: the number of a group members in top-k should be proportional to the group size, k_min <= k
We don't include k_max here !

Expected output: most general patterns treated unfairly, w.r.t. lower bound
Difference from definition 1: in definition 1, we find most specific patterns for upper bound,
most general for lower bound, and all patterns have same bounds.

But here, patterns have different bounds depending on their sizes. We only do lower bound.

"""

"""
naive alg:
for each k, we iterate the whole process again, go top down.
lowerbound = alpha * whole_cardinality * k / data_size

difference from 4:
use str to denote a pattern
"""



import time
import pandas as pd
import numpy as np

from algorithms import pattern_count


def DFSattributes(cur, last, comb, pattern, all_p, mcdes, attributes):
    # print("DFS", attributes)
    if cur == last:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            all_p.append(s)
        return
    else:
        # print("comb[{}] = {}".format(cur, comb[cur]))
        # print("{} {}".format(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max'])))
        for a in range(int(mcdes[attributes[comb[cur]]]['min']), int(mcdes[attributes[comb[cur]]]['max']) + 1):
            s = pattern.copy()
            s[comb[cur]] = a
            DFSattributes(cur + 1, last, comb, s, all_p, mcdes, attributes)


def AllPatternsInComb(comb, NumAttribute, mcdes, attributes):  # comb = [1,4]
    # print("All", attributes)
    all_p = []
    pattern = [-1] * NumAttribute
    DFSattributes(0, len(comb) - 1, comb, pattern, all_p, mcdes, attributes)
    return all_p


def num2string(pattern):
    st = ''
    for i in pattern:
        if i != -1:
            st += str(i)
        st += '|'
    st = st[:-1]
    return st


def P1DominatedByP2(P1, P2):
    length = len(P1)
    for i in range(length):
        if P1[i] == -1:
            if P2[i] != -1:
                return False
        if P1[i] != -1:
            if P2[i] != P1[i] and P2[i] != -1:
                return False
    return True

def PatternEqual(m, P):
    length = len(m)
    if len(P) != length:
        return False
    for i in range(length):
        if m[i] != P[i]:
            return False
    return True


# coverage of P among dataset D
def cov(P, D):
    cnt = 0
    for d in D:
        if P1DominatedByP2(d, P):
            cnt += 1
    return cnt


# whether a pattern P is dominated by MUP M
# except from P itself
def PDominatedByM(P, M):
    for m in M:
        if PatternEqual(m, P):
            continue
        if P1DominatedByP2(P, m):
            # print(P, "domintated by", m)
            return True, m
    return False, None

def PDominatesM(P, M):
    for m in M:
        if PatternEqual(m, P):
            continue
        if P1DominatedByP2(m, P):
            return True, m
    return False, None


def P1DominatedByP2ForStr(str1, str2, num_att):
    if str1 == str2:
        return True
    num_separator = num_att - 1
    start_pos1 = 0
    start_pos2 = 0
    for i in range(num_separator):
        p1 = str1.find("|", start_pos1)
        p2 = str2.find("|", start_pos2)
        s1 = str1[start_pos1:p1]
        s2 = str2[start_pos2:p2]
        if s1 != s2 and s2 != '':
            return False
        start_pos1 = p1 + 1
        start_pos2 = p2 + 1
    s1 = str1[start_pos1:]
    s2 = str2[start_pos2:]
    if s1 != s2 and s2 != '':
        return False
    return True


def GenerateChildren(P, whole_data_frame, ranked_data, attributes):
    children = []
    length = len(P)
    i = 0
    for i in range(length-1, -1, -1):
        if P[i] != -1:
            break
    if P[i] == -1:
        i -= 1
    for j in range(i+1, length, 1):
        for a in range(int(whole_data_frame[attributes[j]]['min']), int(whole_data_frame[attributes[j]]['max'])+1):
            s = P.copy()
            s[j] = a
            # print(ranked_data.loc[3, attributes[j]], type(ranked_data.loc[3, attributes[j]]))
            if not isinstance(ranked_data.loc[3, attributes[j]], (int, np.integer)):
            #if type(whole_data_frame[attributes[j]]['min']) is not int:
                s[j] = float(a)
            children.append(s)
    return children

#
# def CheckDominationAndAdd(pattern, pattern_treated_unfairly):
#     to_remove = []
#     for p in pattern_treated_unfairly:
#         if PatternEqual(p, pattern):
#             return
#         if P1DominatedByP2(pattern, p):
#             return
#         elif P1DominatedByP2(p, pattern):
#             to_remove.append(p)
#     for p in to_remove:
#         pattern_treated_unfairly.remove(p)
#     pattern_treated_unfairly.append(pattern)
#

# patterns are string
def CheckDominationAndAdd(pattern_st, pattern_treated_unfairly, num_att):
    to_remove = []
    for st in pattern_treated_unfairly:
        if st == pattern_st:
            return
        if P1DominatedByP2ForStr(pattern_st, st, num_att):
            return
        elif P1DominatedByP2ForStr(st, pattern_st, num_att):
            to_remove.append(st)
    for st in to_remove:
        pattern_treated_unfairly.remove(st)
    pattern_treated_unfairly.add(pattern_st)



def string2num(st):
    p = list()
    idx = 0
    item = ''
    i = ''
    for i in st:
        if i == '|':
            if item == '':
                p.append(-1)
            else:
                p.append(int(item))
                item = ''
            idx += 1
        else:
            item += i
    if i != '|':
        p.append(int(item))
    else:
        p.append(-1)
    return p


def PatternInSet(p, set):
    if isinstance(p, str):
        p = string2num(p)
    for q in set:
        if PatternEqual(p, q):
            return True
    return False


"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: delta fairness value 
Thc: size threshold
"""

def NaiveAlg(ranked_data, attributes, Thc, alpha, k_min, k_max, time_limit):
    time0 = time.time()
    data_size = len(ranked_data)
    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()
    whole_data_frame = ranked_data.describe(include='all')
    num_patterns_visited = 0
    pattern_treated_unfairly = []
    overtime_flag = False
    num_att = len(attributes)
    root_str = '|' * (num_att - 1)
    root = [-1] * (len(attributes))
    S = GenerateChildren(root, whole_data_frame, ranked_data, attributes)
    store_children = {root_str: S}
    for k in range(k_min, k_max):
        if overtime_flag:
            print("naive overtime, exiting the loop of k")
            break
        result_set = set()
        patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_kmin.parse_data()
        S = store_children[root_str].copy()
        # lower bound
        while len(S) > 0:
            if time.time() - time0 > time_limit:
                overtime_flag = True
                print("naive overtime")
                break
            P = S.pop(0)
            st = num2string(P)
            num_patterns_visited += 1
            whole_cardinality = pc_whole_data.pattern_count(st)
            if whole_cardinality < Thc:
                continue
            num_top_k = patterns_top_kmin.pattern_count(st)
            lowerbound = alpha * whole_cardinality * k / data_size  # (whole_cardinality / data_size - alpha) * k
            if num_top_k < lowerbound:
                CheckDominationAndAdd(st, result_set, num_att)
            else:
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                    store_children[st] = children
                S = children + S
                continue
        pattern_treated_unfairly.append(result_set)
    time1 = time.time()
    return pattern_treated_unfairly, num_patterns_visited, time1 - time0


#
# all_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                   'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C', 'guardian_C', 'traveltime_C', 'studytime_C',
#                   'failures_C', 'schoolsup_C', 'famsup_C', 'paid_C', 'activities_C', 'nursery_C', 'higher_C',
#                   'internet_C', 'romantic_C', 'famrel_C', 'freetime_C', 'goout_C', 'Dalc_C', 'Walc_C',
#                   'health_C', 'absences_C', 'G1_C', 'G2_C', 'G3_C']
#
# selected_attributes = ['school_C', 'sex_C', 'age_C', 'address_C', 'famsize_C', 'Pstatus_C', 'Medu_C',
#                        'Fedu_C', 'Mjob_C', 'Fjob_C', 'reason_C']
#
# original_data_file = r"../../InputData/StudentDataset/ForRanking_1/student-mat_cat_ranked.csv"
#
# ranked_data = pd.read_csv(original_data_file)
# ranked_data = ranked_data[selected_attributes]
#
# time_limit = 5 * 60
# k_min = 10
# k_max = 16
# Thc = 30
#
# List_k = list(range(k_min, k_max))
#
# alpha = 0.1
#
#
#
# pattern_treated_unfairly, num_patterns_visited, running_time = \
#     NaiveAlg(ranked_data, selected_attributes, Thc,
#                      alpha,
#                      k_min, k_max, time_limit)
#
# print("num_patterns_visited = {}".format(num_patterns_visited))
# print("time = {} s, num of pattern_treated_unfairly = {}".format(running_time,
#         len(pattern_treated_unfairly)), "\n", "patterns:\n",
#       "lower bound ", pattern_treated_unfairly)
#
# print("dominated by pattern_treated_unfairly:")
# for p in pattern_treated_unfairly:
#     if PDominatedByM(p, pattern_treated_unfairly)[0]:
#         print(p)
#
#
