

import sys, os
import pandas as pd

# Compile openai mmlu resultls

def main():
    result_dir = "/home/jeremy/Documents/WPI_MS/Q_Inoculate/Q_Inoculate/MMLU_Repo/test_output/results_gpt-3.5-turbo-0125"

    fnames = os.listdir(result_dir)
    fpaths = [os.path.join(result_dir, x) for x in fnames]

    total_qs = 0
    total_correct = 0

    for f in fpaths:
        df = pd.read_csv(f)
        # https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe

        for index, row in df.iterrows():
            if row['gpt-3.5-turbo-0125_correct']:
                total_correct += 1
            total_qs += 1

    print("Total accuracy: " + str(total_correct/total_qs))

if __name__ == "__main__":
    main()