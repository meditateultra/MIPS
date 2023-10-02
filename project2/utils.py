import numpy as np

code_map_func = {
    '100000': 'ADD',
    '100010': 'SUB',
    '100100': 'AND',
    '100111': 'NOR',
    '101010': 'SLT',
    '001000': 'JR',
    '001101': 'BREAK',
    '000000': 'SLL',
    '000010': 'SRL',
    '000011': 'SRA',
}

code_map_op = {
    '110000': 'ADD',
    '110001': 'SUB',
    '100001': 'MUL',
    '110010': 'AND',
    '110011': 'NOR',
    '110101': 'SLT',
    '000010': 'J',
    '000100': 'BEQ',
    '000001': 'BLTZ',
    '000111': 'BGTZ',
    '101011': 'SW',
    '100011': 'LW',
    '011100': 'MUL'
}

name_map_function = {
    'J': 'simulation_J',
    'JR': 'simulation_JR',
    'BEQ': 'simulation_BEQ',
    'BLTZ': 'simulation_BLTZ',
    'BGTZ': 'simulation_BGTZ',
    'BREAK': 'simulation_BREAK',
    'SW': 'simulation_SW',
    'LW': 'simulation_LW',
    'SLL': 'simulation_SLL',
    'SRL': 'simulation_SRL',
    'SRA': 'simulation_SRA',
    'NOP': 'simulation_NOP',
    'ADD': 'simulation_ADD',
    'SUB': 'simulation_SUB',
    'MUL': 'simulation_MUL',
    'AND': 'simulation_AND',
    'NOR': 'simulation_NOR',
    'SLT': 'simulation_SLT',
}

easy_two_ins = ['ADD', 'SUB', 'MUL', 'AND', 'NOR', 'SLT']
cond_jump_ins = ['BLTZ', 'BGTZ']
easy_load_ins = ['SW', 'LW']
easy_move_ins = ['SLL', 'SRL', 'SRA']

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


def get_instruction_name(instruction):
    if instruction.op_code == '000000':
        ins_name_str = code_map_func[instruction.func]
    else:
        ins_name_str = code_map_op[instruction.op_code]
    return ins_name_str

def generate_instruction_str(ins_name_str, instruction):
    ins_args_str = ''
    if ins_name_str in easy_two_ins:
        if instruction.category == 0:
            rd = get_register_name(instruction.rd)
            rs = get_register_name(instruction.rs)
            rt = get_register_name(instruction.rt)
            ins_args_str = rd + ', ' + rs + ', ' + rt
        else:
            rs = get_register_name(instruction.rs)
            rt = get_register_name(instruction.rt)
            immediate = instruction.rd + instruction.shift_amount + instruction.func
            ins_args_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))
    elif ins_name_str == 'J':
        immediate = instruction.rs + instruction.rt + instruction.rd + \
                    instruction.shift_amount + instruction.func + '00'
        ins_args_str = '#' + str(complement_to_int(immediate))
    elif ins_name_str == 'JR':
        ins_args_str = get_register_name(instruction.rs)
    elif ins_name_str in cond_jump_ins:
        rs = get_register_name(instruction.rs)
        offset = instruction.rt + instruction.rd + \
                 instruction.shift_amount + instruction.func + '00'
        offset = sign_extend(offset, width=32)
        ins_args_str = rs + ', ' + '#' + str(complement_to_int(offset))
    elif ins_name_str == 'BEQ':
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
        offset = sign_extend(offset, width=32)
        ins_args_str = rs + ', ' + rt + ', ' + '#' + str(complement_to_int(offset))
    elif ins_name_str in easy_load_ins:
        base = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        offset = instruction.rd + instruction.shift_amount + instruction.func
        ins_args_str = rt + ', ' + str(complement_to_int(offset)) + '(' + base + ')'
    elif ins_name_str in easy_move_ins:
        rd = get_register_name(instruction.rd)
        rt = get_register_name(instruction.rt)
        ins_args_str = rd + ', ' + rt + ', ' + '#' + \
                       str(complement_to_int(instruction.shift_amount))
    return ins_args_str