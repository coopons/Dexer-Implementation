# This file is adapted from the Dexer project.
# Original source: Detection of Groups with Biased Representation in Ranking
# https://github.com/talni/Detection-of-Groups-with-Biased-Representation-in-Ranking

"""
naive alg for ranking
"""

from itertools import combinations
import pandas as pd
from algorithms import pattern_count
import time
import numpy as np



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


def CheckDominationAndAddForLowerBound(pattern_st, pattern_treated_unfairly, num_att):
    to_remove = set()
    for st in pattern_treated_unfairly:
        if st == pattern_st:
            return
        if P1DominatedByP2ForStr(pattern_st, st, num_att):
            return
        elif P1DominatedByP2ForStr(st, pattern_st, num_att):
            to_remove.add(st)
    for st in to_remove:
        pattern_treated_unfairly.remove(st)
    pattern_treated_unfairly.add(pattern_st)



def CheckDominationAndAddForUpperbound(pattern, pattern_treated_unfairly):
    to_remove = []
    for p in pattern_treated_unfairly:
        if PatternEqual(p, pattern):
            return
        if P1DominatedByP2(pattern, p):
            to_remove.append(p)
        elif P1DominatedByP2(p, pattern):
            return
    for p in to_remove:
        pattern_treated_unfairly.remove(p)
    pattern_treated_unfairly.append(pattern)



"""
whole_data: the original data file 
mis_class_data: file containing mis-classified tuples
Tha: delta fairness value 
Thc: size threshold
"""

def NaiveAlg(ranked_data, attributes, Thc, Lowerbounds, k_min, k_max, time_limit):
    time0 = time.time()
    pc_whole_data = pattern_count.PatternCounter(ranked_data, encoded=False)
    pc_whole_data.parse_data()
    whole_data_frame = ranked_data.describe(include='all')
    num_patterns_visited = 0
    pattern_treated_unfairly_lowerbound = []
    overtime_flag = False
    root = [-1] * (len(attributes))
    num_att = len(attributes)
    S = GenerateChildren(root, whole_data_frame, ranked_data, attributes)
    root_str = '|' * (num_att - 1)
    store_children = {root_str: S}
    for k in range(k_min, k_max):
        if overtime_flag:
            print("naive overtime, exiting the loop of k")
            break
        num_patterns_visited_k = 0
        result_set_lowerbound = set()
        S = store_children[root_str].copy()
        patterns_top_kmin = pattern_count.PatternCounter(ranked_data[:k], encoded=False)
        patterns_top_kmin.parse_data()
        # lower bound
        while len(S) > 0:
            if time.time() - time0 > time_limit:
                overtime_flag = True
                print("naive overtime")
                break
            P = S.pop(0)
            st = num2string(P)
            num_patterns_visited_k += 1
            whole_cardinality = pc_whole_data.pattern_count(st)
            # print("P={}, whole size={}".format(P, whole_cardinality))
            if whole_cardinality < Thc:
                continue
            num_top_k = patterns_top_kmin.pattern_count(st)
            if num_top_k < Lowerbounds[k - k_min]:
                CheckDominationAndAddForLowerBound(st, result_set_lowerbound, num_att)
            else:
                if st in store_children:
                    children = store_children[st]
                else:
                    children = GenerateChildren(P, whole_data_frame, ranked_data, attributes)
                    store_children[st] = children
                S = S + children
                continue
        pattern_treated_unfairly_lowerbound.append(result_set_lowerbound)
        num_patterns_visited += num_patterns_visited_k
    time1 = time.time()
    return pattern_treated_unfairly_lowerbound, num_patterns_visited, time1 - time0


# selected_attributes = ["sex_binary", "age_binary", "race_C", "age_bucketized"]
#
# original_file = r"../../InputData/CompasData/ForRanking/SmallDataset/CompasData_ranked_5att_1000.csv"
# ranked_data = pd.read_csv(original_file)
# ranked_data = ranked_data.drop('rank', axis=1)
#
# # def GraphTraverse(ranked_data, Thc, Lowerbounds, Upperbounds, k_min, k_max, time_limit):
#
#
# time_limit = 20 * 60
# k_min = 40
# k_max = 50
# Thc = 10
# Lowerbounds = [1, 1, 2, 2, 2, 3, 3, 3, 3, 4]
# Upperbounds = [5,5,6,7,8, 9,10,11,11, 12, 12]
#
# print(ranked_data[:k_max])
#
# pattern_treated_unfairly_lowerbound, pattern_treated_unfairly_upperbound, num_patterns_visited, running_time = NaiveAlg(ranked_data, selected_attributes, Thc,
#                                                                      Lowerbounds, Upperbounds,
#                                                                      k_min, k_max, time_limit)
#
# print("num_patterns_visited = {}".format(num_patterns_visited))
# print("time = {} s, num of pattern_treated_unfairly_lowerbound = {}, num of pattern_treated_unfairly_upperbound = {} ".format(running_time,
#         len(pattern_treated_unfairly_lowerbound), len(pattern_treated_unfairly_upperbound)), "\n", "patterns:\n",
#       pattern_treated_unfairly_lowerbound, "\n", pattern_treated_unfairly_upperbound)
#
# print("dominates pattern_treated_unfairly_upperbound:")
# for p in pattern_treated_unfairly_upperbound:
#     t, m = PDominatesM(p, pattern_treated_unfairly_upperbound)
#     if t:
#         print(p, m)
#
