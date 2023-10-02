import numpy as np

def read_input_file(filename):
    instructions = []
    with open(filename, "r") as reader:
        for line in reader.readlines():
            instructions.append(line.strip())
    return instructions


def get_register_name(reg):
    reg_name = 'R' + str(int(reg, 2))
    return reg_name


def complement_to_int(complement):
    res = 0
    if complement[0] == '0':
        res = int(complement, base=2)
    else:
        com_len = len(complement)
        res += -1 * 2**(com_len - 1)
        for i in range(1, len(complement)):
            res += int(complement[i]) * 2 ** (com_len - 1 - i)
    return res


def int_to_complement(num, width=32):
    return np.binary_repr(num, width=width)


def sign_extend(complement, width=32):
    res = complement[0] * (width - len(complement)) + complement
    return res


def zero_extend(complement, width=32):
    res = '0' * (width - len(complement)) + complement
    return res

def instruction_split_str(instruction):
    return instruction.op_code + ' ' + instruction.rs + ' ' + instruction.rt + ' ' +\
           instruction.rd + ' ' + instruction.shift_amount + ' ' + instruction.func
