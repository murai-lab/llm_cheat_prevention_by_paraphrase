# JL - quick script to combine all of the test batches.
import pickle
import sys, os

def main():
    fpath_list = ['/home/jlim/MS_Project/Paraphrase_Attack/paraphrase_batches/test/para_batch_0.pickle', 
                  '/home/jlim/MS_Project/Paraphrase_Attack/paraphrase_batches/test/para_batch_1.pickle',
                  '/home/jlim/MS_Project/Paraphrase_Attack/paraphrase_batches/test/para_batch_2.pickle',
                  '/home/jlim/MS_Project/Paraphrase_Attack/paraphrase_batches/test/para_batch_3.pickle',
                  '/home/jlim/MS_Project/Paraphrase_Attack/paraphrase_batches/test/para_batch_4.pickle']

    record_list = []
    for fpath in fpath_list:
        with open(fpath, 'rb') as f:
            batch = pickle.load(f)

        record_list = record_list + batch

    print("Combined number of records: " + str(len(record_list)))

    with open('/home/jlim/MS_Project/Paraphrase_Attack/paraphrase_batches/test/para_batches_combined.pickle', 'wb') as f:
        pickle.dump(record_list, f)
    
if __name__ == "__main__":
    main()