# from biogeme.expressions import *
# from biogeme.loglikelihood import *
# from biogeme.biogeme import *
# from biogeme.tests.headers import *
# from loglikelihood import *
# from statistics import *

### List of parameters. to.. be estimated
from biogeme.models import lognested, Beta
from biogeme.tests.headers import Choice

ASC_CAR = Beta('ASC_CAR', 0, -10000, 10000, 0)
ASC_MT = Beta('ASC_MT', 0, -10000, 10000, 0)
ASC_BK = Beta('ASC_BK', 0, -10000, 10000, 0)
ASC_SM = Beta('ASC_SM', 0, -10000, 10000, 0)
BETA_TIME_PT = Beta('BETA_TIME_PT', 0, -10000, 10000, 0)
BETA_TIME_CR = Beta('BETA_TIME_CR', 0, -10000, 10000, 0)
BETA_TIME_TX = Beta('BETA_TIME_TX', 0, -10000, 10000, 0)
BETA_TIME_MT = Beta('BETA_TIME_MT', 0, -10000, 10000, 0)
BETA_TIME_OVT = Beta('BETA_TIME_OVT', 0, -10000, 10000, 0)
BETA_COST = Beta('BETA_COST', 0, -10000, 10000, 0)
BETA_COST_TX = Beta('BETA_COST_TX', 0, -10000, 10000, 0)
BETA_BK = Beta('BETA_BK', 0, -10000, 10000, 0)

V_PT = BETA_TIME_PT * tr_VT + BETA_TIME_OVT * tr_OVT
V_CAR = ASC_CAR + BETA_TIME_CR * cr_VT + BETA_COST * cr_CO
V_TX = BETA_TIME_TX * cr_VT + BETA_COST_TX * tx_CO
V_MT = ASC_MT + BETA_TIME_MT * mt_VT
V_BK = ASC_BK + BETA_BK * bk_time
V_SM = ASC_SM + BETA_SM * mp_time

V = {
    9: V_PT,
    8: V_CAR,
    5: V_MT,
    6: V_TX,
    2: V_BK,
    4: V_SM
}

av = {
    9: 1,
    8: 1,
    5: 1,
    6: 1,
    2: 1,
    4: 1
}
NEST_CAR = Beta('NEST_CAR', 1, 0.0, 10, 0)
NEST_NM = Beta('NEST_NM', 1, 0.0, 10, 0)
PT = 1.0,
CAR = NEST_CAR, [8, 5, 6]
NM = NEST_NM, [2, 4]
nests = PT, CAR, NM

BIOGEME_OBJECT.EXCLUDE = ((Choice == 0) + (Choice == 1)) > 0
# The choice model is a nested logit, with availability conditions
logprob = lognested(V, av, nests, Choice)
# Defines an itertor on the data
rowIterator('obsIter')
# Statistics
nullLoglikelihood(av, ' obsIter')
choiceSet = [2, 4, 5, 6, 8, 9]
cteLoglikelihood(choiceSet, Choice, ' obsIter')
availabilityStatistics(av, 'obsIter')
# Define the likelihood function for the estimation .
BIOGEME_OBJECT.ESTIMATE = Sum(logprob,'obsIter')
BIOGEME_OBJECT.PARAMETERS['optimi zationAlgorithm'] = "CFSQP"
