

import numpy as np
import math
from matplotlib import pyplot as plt

from scipy.stats import binom

def main():
    # wiki stats
    # sensitivity = 0.67  # qqp: 0.39
    # specificity = 0.62 # qqp: 0.78

    # qqp stats
    sensitivity = 0.39
    specificity = 0.78

    confidence = 0.99

    trials = math.ceil(math.log((1.0 - confidence), (1.0-sensitivity)))

    print(trials)
    print(math.pow((1.0-sensitivity),trials))

    # NEW: How many validated paraphrases need to be created to guarantee at least 1 correct paraphrase?
    # Need to test different paraphrase efficiencies...

    # 50% has the highest variance, though lets test out for the less-likely cases.
    # This is an unknown value, of how often our paraphraser makes valid paraphrases.
    paraphrase_efficiency = 0.5

    true_success_rate = (paraphrase_efficiency * sensitivity) / ((paraphrase_efficiency * sensitivity) + (1.0 - paraphrase_efficiency)*(1.0-specificity))

    print("Estimated true success rate: " + str(true_success_rate))
    trials_success = math.ceil(math.log((1.0 - confidence), (1.0 - true_success_rate)))

    # Lower confidence? Can tune this...
    trials_success_90 = math.ceil(math.log((1.0 - 0.9), (1.0 - true_success_rate)))

    # Independent of paraphrase efficiency. But low efficiency will reduce performance.
    print("Estimated number of validated paraphrases to guarantee one, at 99% confidence: " + str(trials_success))
    print("Estimated number of validated paraphrases to guarantee one, at 90% confidence: " + str(trials_success_90))


    # How many trials to guarantee 6 or 11 paraphrases? Test different efficiencies
    # Try 99% confidence

    # Binomial distribution: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
    # Interval is around the mean. We want the left half to be < 0.01 (right tailed interval)
    test_range = range(1000)
    guaranteed_trials = []

    for x in test_range:
        the_interval = binom.interval(confidence=0.98, n=(x+1), p=paraphrase_efficiency)
        guaranteed_trials.append(the_interval[0])

    plt.plot(guaranteed_trials)
    plt.axhline(trials_success, color='red')
    plt.axhline(trials_success_90, color='blue')
    plt.show()

    # For QQP stats of best one-char prompt.
    # Total runs - efficiency - number of screened total responses.
    # 15 for 90%, 18 for 99% - for 50% paraphrase efficiency - need 3/5 to pass!
    # 222 for 90%, 385 for 99% - for 10% paraphrase efficiency - need 13/26 to pass!
    # 2 for 90%, 3 for 99% - for 90% paraphrase efficiency - need 1/2 to pass!

    # TODO: Plot trial curve over range of paraphrase efficiencies - how bad does it get at extremely low efficiency?
    # Key thing here: Bad paraphraser will kill performance. Evaluator can help
    # Evaluator affects # of qs the end user has to see, paraphraser affects overall performance

    print("Done")

if __name__ == "__main__":
    main()