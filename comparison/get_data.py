import os
import sys
import logging
import argparse

# Set logging object.
if not os.path.isdir('./logs'):
    os.mkdir('./logs')
logging.basicConfig(level=logging.DEBUG, filename='logs/get_data.log',
    filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Constants.
viol = 1
n_viol = 0
test_share = 0.2
violation_symb = '!'
state_list_char = '[' # Refer to the first character in obs file.
output_folder = './data'
keywords = ['vsignal', 'obs']


def get_files(folder_path):
    # Read folder_path and return files.
    files = os.listdir(folder_path)
    v_files = [] # List of verified files.

    for f in files:
        if keywords[0] in f and keywords[1] in f:
            v_files.append(os.path.join(folder_path, f))

    return v_files


def get_states(lines):
    # Read lines and select those referring to a list of states.
    s_lines = [] # Define a list of lines describing a list of states.
    for line in lines:
        if line.startswith(state_list_char):
            s_lines.append(eval(line))

    return s_lines


def encode_states(s_lists):
    # List of states and encode them into a binary representation.
    sts_list = [] # List of states.

    for s_list in s_lists:
        # Run over the list of list of states.
        for state in s_list:
            if state not in sts_list and state != violation_symb:
                # If starts with a '[', add to a list.
                sts_list.append(state)

    s_set = list(set(sts_list)) # Get a set of all states.

    len_set = len(s_set)
    b_max = bin(len_set)[2:]
    len_b = len(b_max)

    enc_states = dict()

    for i in range(len_set):
        state = s_set[i]
        b_state = bin(i)[2:]
        enc_states[state] = '0'*(len_b - len(b_state)) + b_state
        
    return enc_states


def add_class(s_list, e_states):
    # Add class to each state according to violation occurrence.
    f_struct = dict()

    for ind, states_list in enumerate(s_list):
        f_struct[ind] = []
        for ind2, st in enumerate(states_list):
            if st == violation_symb:
                continue
            clss = n_viol
            if ind2 + 1 < len(states_list):
                if states_list[ind2 + 1] == violation_symb:
                    clss = viol
            if clss:
                print "VIOLATION!"
            f_struct[ind].append((e_states[st], clss))

    return f_struct


def process_files(files):
    # Read files and return structured data.
    data = dict()
    for ind, f in enumerate(files):
        logging.debug("Processing: %s" % f)
        # Read file.
        f_lines = open(f, 'r').readlines()
        states_list = get_states(f_lines)
        logging.debug("States list: {}".format(states_list[:2]))
        enc_states = encode_states(states_list)
        logging.debug("Encoded states: {}".format(enc_states.keys()[:2]))
        data[ind] = add_class(states_list, enc_states)
        logging.debug("Added class to data: {}".format(data[ind].keys()[:2]))

    return data


def save_data(data, test_share):
    # Read data and save to files.
    # Create a folder if necessary.
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    for file_key in data:
        # Divide into train and test.
        file_struct = data[file_key]
        len_file = len(file_struct)
        test_size = float(len_file*test_share)
        if not test_size:
            # Ensure that test_size is never zero.
            logging.debug("Setting test_size to 1 since it was 0.")
            test_size = 1
        train_size = len_file - test_size
        logging.debug("Test size: %d\nTrain size: %d" % (
            test_size, train_size))

        w_train = open(os.path.join(output_folder,
            str(file_key)) + "_train.txt", 'w')
        w_train.write("sample class\n")
        w_test = open(os.path.join(output_folder,
            str(file_key)) + "_test.txt", 'w')
        w_test.write("sample class\n")

        for ind, file_id in enumerate(file_struct):
            for tup in file_struct[file_id]:
                sample, clss = tup
                if ind < test_size:
                    # Add states to test file.
                    w_test.write(sample + ' ' + str(clss) + "\n")
                else:
                    w_train.write(sample + ' ' + str(clss) + "\n")
    return True

def main(folder_path, test_share):
    # Reads folder path and create data files for training.
    logging.debug("Scanning folder: %s" % folder_path)
    files = get_files(folder_path)
    logging.debug("Got %d valid files." % len(files))
    data = process_files(files)
    logging.debug("Processed data.")
    save = save_data(data, test_share)

    if save:
        logging.debug("Successfully saved data.")
    else:
        logging.debug("Error while saving data.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Folder path to read files.')
    parser.add_argument('folder_path', type=str,
        help='Path to a folder containing observations.')
    parser.add_argument('--test', type=float,
        help='Percentage of data that must be used for test.')

    args = parser.parse_args()
    if args.test:
        test_share = args.test
    main(args.folder_path, test_share)