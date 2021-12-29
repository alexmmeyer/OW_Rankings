import pandas

event_points = pandas.read_csv("event_points.csv")
DIRECTORY = "results"
RANKING_FILE_NAME = "PageRanking.csv"

# Must be in MM/DD/YYYY format:
RANKING_AS_OF = "08/12/2018"

# Depreciation Period: time period in days over which a depreciation is applied to the initial weight of a result.
DEPRECIATION_PERIOD = 365 * 1.2

# Drives age_weight_exp() exponential decay function. The more negative, the quicker the decline in age_weight.
LAMBDA = -2
