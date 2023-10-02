import numpy as np


class instruction:

    def __init__(self, instruction_string):
        self.category = int(instruction_string[0:1])
        self.op_code = instruction_string[0:6]
        self.rs = instruction_string[6:11]
        self.rt = instruction_string[11:16]
        self.rd = instruction_string[16:21]
        self.shift_amount = instruction_string[21:26]
        self.func = instruction_string[26:]
        self.ins_str = instruction_string

    def __str__(self):
        return self.ins_str


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


def disassembly_J(instruction):
    immediate = instruction.rs + instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    ins_str = '#' + str(complement_to_int(immediate))
    return ins_str


def simulation_J(instruction, PC, data_address, regsters, memory):
    immediate = instruction.rs + instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    PC[0] = complement_to_int(immediate)


def disassembly_JR(instruction):
    ins_str = get_register_name(instruction.rs)
    return ins_str


def simulation_JR(instruction, PC, data_address, regsters, memory):
    PC[0] = registers[complement_to_int(instruction.rs)]


def disassembly_BEQ(instruction):
    rs = get_register_name(instruction.rs)
    rt = get_register_name(instruction.rt)
    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    ins_str = rs + ', ' + rt + ', ' + '#' + str(complement_to_int(offset))
    return ins_str


def simulation_BEQ(instruction, PC, data_address, regsters, memory):
    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    if registers[int(instruction.rs, 2)] == registers[int(instruction.rt, 2)]:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4


def disassembly_BLTZ(instruction):
    rs = get_register_name(instruction.rs)
    offset = instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    ins_str = rs + ', ' + '#' + str(complement_to_int(offset))
    return ins_str


def simulation_BLTZ(instruction, PC, data_address, regsters, memory):
    offset = instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    if registers[int(instruction.rs, 2)] < 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4


def disassembly_BGTZ(instruction):
    rs = get_register_name(instruction.rs)
    offset = instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    ins_str = rs + ', ' + '#' + str(complement_to_int(offset))
    return ins_str


def simulation_BGTZ(instruction, PC, data_address, regsters, memory):
    offset = instruction.rt + instruction.rd + \
        instruction.shift_amount + instruction.func + '00'
    offset = sign_extend(offset, width=32)
    if registers[int(instruction.rs, 2)] > 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4


def disassembly_BREAK(instruction):
    return ""


def simulation_BREAK(instruction, PC, data_address, regsters, memory):
    PC[0] += 4


def disassembly_SW(instruction):
    base = get_register_name(instruction.rs)
    rt = get_register_name(instruction.rt)
    offset = instruction.rd + instruction.shift_amount + instruction.func
    ins_str = rt + ', ' + str(complement_to_int(offset)) + '(' + base + ')'
    return ins_str


def simulation_SW(instruction, PC, data_address, regsters, memory):
    base = int(instruction.rs, 2)
    offset = instruction.rd + instruction.shift_amount + instruction.func
    valid_address = int(
        (registers[base] +
         complement_to_int(offset) -
         data_address) /
        4)
    memory[int(valid_address)] = registers[int(instruction.rt, 2)]
    PC[0] += 4


def disassembly_LW(instruction):
    base = get_register_name(instruction.rs)
    rt = get_register_name(instruction.rt)
    offset = instruction.rd + instruction.shift_amount + instruction.func
    ins_str = rt + ', ' + str(complement_to_int(offset)) + '(' + base + ')'
    return ins_str


def simulation_LW(instruction, PC, data_address, regsters, memory):
    base = int(instruction.rs, 2)
    offset = instruction.rd + instruction.shift_amount + instruction.func
    valid_address = int(
        (registers[base] +
         complement_to_int(offset) -
         data_address) / 4)
    registers[int(instruction.rt, 2)] = memory[int(valid_address)]
    PC[0] += 4


def disassembly_SLL(instruction):
    rd = get_register_name(instruction.rd)
    rt = get_register_name(instruction.rt)
    ins_str = rd + ', ' + rt + ', ' + '#' + \
        str(complement_to_int(instruction.shift_amount))
    return ins_str


def simulation_SLL(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = (registers[int(instruction.rt, 2)] << int(
        instruction.shift_amount, 2)) & (2 ** 32 - 1)
    PC[0] += 4


def disassembly_SRL(instruction):
    rd = get_register_name(instruction.rd)
    rt = get_register_name(instruction.rt)
    ins_str = rd + ', ' + rt + ', ' + '#' + \
        str(complement_to_int(instruction.shift_amount))
    return ins_str


def simulation_SRL(instruction, PC, data_address, regsters, memory):
    sa = complement_to_int(instruction.shift_amount)
    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]
    res = '0' * sa + res
    registers[int(instruction.rd, 2)] = complement_to_int(res)
    PC[0] += 4


def disassembly_SRA(instruction):
    rd = get_register_name(instruction.rd)
    rt = get_register_name(instruction.rt)
    ins_str = rd + ', ' + rt + ', ' + '#' + \
        str(complement_to_int(instruction.shift_amount))
    return ins_str


def simulation_SRA(instruction, PC, data_address, regsters, memory):
    sa = complement_to_int(instruction.shift_amount)
    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]
    res = res + res[0] * sa
    registers[int(instruction.rd, 2)] = complement_to_int(res)
    PC[0] += 4


def disassembly_NOP(instruction):
    return ""


def simulation_NOP(instruction, PC, data_address, regsters, memory):
    PC[0] += 4


def disassembly_ADD(instruction):
    if instruction.category == 0:
        rd = get_register_name(instruction.rd)
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)

        ins_str = rd + ', ' + rs + ', ' + rt
        return ins_str
    else:
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        ins_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))
        return ins_str


def simulation_ADD(instruction, PC, data_address, regsters, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] + registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] + complement_to_int(immediate)
        PC[0] += 4


def disassembly_SUB(instruction):
    if instruction.category == 0:
        rd = get_register_name(instruction.rd)
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        ins_str = rd + ', ' + rs + ', ' + rt
        return ins_str
    else:
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        ins_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))
        return ins_str


def simulation_SUB(instruction, PC, data_address, regsters, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] - registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] - complement_to_int(immediate)
        PC[0] += 4


def disassembly_MUL(instruction):
    if instruction.category == 0:
        rd = get_register_name(instruction.rd)
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        ins_str = rd + ', ' + rs + ', ' + rt
        return ins_str
    else:
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        ins_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))
        return ins_str


def simulation_MUL(instruction, PC, data_address, regsters, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] * registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] * complement_to_int(immediate)
        PC[0] += 4


def disassembly_AND(instruction):
    if instruction.category == 0:
        rd = get_register_name(instruction.rd)
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        ins_str = rd + ', ' + rs + ', ' + rt
        return ins_str
    else:
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        ins_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))
        return ins_str


def simulation_AND(instruction, PC, data_address, regsters, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = registers[int(
            instruction.rs, 2)] & registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = registers[int(
            instruction.rs, 2)] & complement_to_int(immediate)
        PC[0] += 4


def disassembly_NOR(instruction):
    if instruction.category == 0:
        rd = get_register_name(instruction.rd)
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        ins_str = rd + ', ' + rs + ', ' + rt
        return ins_str
    else:
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        ins_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))
        return ins_str


def simulation_NOR(instruction, PC, data_address, regsters, memory):
    if instruction.category == 0:
        registers[int(instruction.rd, 2)] = ~(
            registers[int(instruction.rs, 2)]) | registers[int(instruction.rt, 2)]
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        registers[int(instruction.rt, 2)] = ~(
            registers[int(instruction.rs, 2)] | complement_to_int(immediate))
        PC[0] += 4


def disassembly_SLT(instruction):
    if instruction.category == 0:
        rd = get_register_name(instruction.rd)
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        ins_str = rd + ', ' + rs + ', ' + rt
        return ins_str
    else:
        rs = get_register_name(instruction.rs)
        rt = get_register_name(instruction.rt)
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        ins_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))
        return ins_str


def simulation_SLT(instruction, PC, data_address, regsters, memory):
    if instruction.category == 0:
        if registers[int(instruction.rs, 2)] < registers[int(instruction.rt, 2)]:
            registers[int(instruction.rd, 2)] = 1
        else:
            registers[int(instruction.rd, 2)] = 0
        PC[0] += 4
    else:
        immediate = instruction.rd + instruction.shift_amount + instruction.func
        if registers[int(instruction.rs, 2)] < complement_to_int(immediate):
            registers[int(instruction.rt, 2)] = 1
        else:
            registers[int(instruction.rt, 2)] = 0
        PC[0] += 4


def instruction_split_str(instruction):
    return instruction.op_code + ' ' + instruction.rs + ' ' + instruction.rt + ' ' +\
           instruction.rd + ' ' + instruction.shift_amount + ' ' + instruction.func


def generate_disassembly(instruction, PC, disassembly_file):
    if instruction.op_code == '000000':
        ins_name_str = code_map_func[instruction.func]
    else:
        ins_name_str = code_map_op[instruction.op_code]
    ins_args_str = eval(name_map_function[ins_name_str][0])(instruction)
    if ins_name_str == 'BREAK':
        disassembly_file.write(instruction_split_str(instruction) + '\t' +
                               str(PC[0]) + '\t' + ins_name_str + '\n')
        return True
    disassembly_file.write(instruction_split_str(instruction) +
                           '\t' +
                           str(PC[0]) +
                           '\t' +
                           ins_name_str +
                           ' ' +
                           ins_args_str +
                           '\n')
    return False


def generate_simulation(
        instruction,
        PC,
        data_address,
        registers,
        memory,
        simulation_file,
        count,
        cnt):
    if instruction.op_code == '000000':
        ins_name_str = code_map_func[instruction.func]
    else:
        ins_name_str = code_map_op[instruction.op_code]
    ins_args_str = eval(name_map_function[ins_name_str][0])(instruction)

    simulation_file.write("--------------------\n")

    flag = False

    if ins_name_str != 'BREAK':
        simulation_file.write("Cycle:" +
                              str(cnt) +
                              '\t' +
                              str(PC[0]) +
                              '\t' +
                              ins_name_str +
                              '\t' +
                              ins_args_str +
                              '\n')
    else:
        simulation_file.write("Cycle:" + str(cnt) + '\t' +
                              str(PC[0]) + '\t' + ins_name_str + '\n')
        flag = True

    simulation_file.write('\n')

    eval(
        name_map_function[ins_name_str][1])(
        instruction,
        PC,
        data_address,
        registers,
        memory)

    simulation_file.write('Registers\n')
    sim_str = 'R00:'
    for i in range(16):
        sim_str += '\t' + str(registers[i])
    sim_str += '\n'
    simulation_file.write(sim_str)
    sim_str = 'R16:'
    for i in range(16, 32):
        sim_str += '\t' + str(registers[i])
    sim_str += '\n'
    simulation_file.write(sim_str)
    simulation_file.write('\n')
    simulation_file.write('Data\n')

    for i in range(0, count, 8):
        simulation_file.write(str(data_address) +
                              ':\t' +
                              str(memory[i]) +
                              '\t' +
                              str(memory[i +
                                         1]) +
                              '\t' +
                              str(memory[i +
                                         2]) +
                              '\t' +
                              str(memory[i +
                                         3]) +
                              '\t' +
                              str(memory[i +
                                         4]) +
                              '\t' +
                              str(memory[i +
                                         5]) +
                              '\t' +
                              str(memory[i +
                                         6]) +
                              '\t' +
                              str(memory[i +
                                         7]) +
                              '\n')
        data_address += 8 * 4

    simulation_file.write('\n')
    return flag

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
    'J': ['disassembly_J', 'simulation_J'],
    'JR': ['disassembly_JR', 'simulation_JR'],
    'BEQ': ['disassembly_BEQ', 'simulation_BEQ'],
    'BLTZ': ['disassembly_BLTZ', 'simulation_BLTZ'],
    'BGTZ': ['disassembly_BGTZ', 'simulation_BGTZ'],
    'BREAK': ['disassembly_BREAK', 'simulation_BREAK'],
    'SW': ['disassembly_SW', 'simulation_SW'],
    'LW': ['disassembly_LW', 'simulation_LW'],
    'SLL': ['disassembly_SLL', 'simulation_SLL'],
    'SRL': ['disassembly_SRL', 'simulation_SRL'],
    'SRA': ['disassembly_SRA', 'simulation_SRA'],
    'NOP': ['disassembly_NOP', 'simulation_NOP'],
    'ADD': ['disassembly_ADD', 'simulation_ADD'],
    'SUB': ['disassembly_SUB', 'simulation_SUB'],
    'MUL': ['disassembly_MUL', 'simulation_MUL'],
    'AND': ['disassembly_AND', 'simulation_AND'],
    'NOR': ['disassembly_NOR', 'simulation_NOR'],
    'SLT': ['disassembly_SLT', 'simulation_SLT'],
}

input_path = 'sample.txt'

if __name__ == '__main__':
    registers = [0] * 32
    memory = [0] * 64
    ins_str = read_input_file(input_path)
    instructions = []
    for ins in ins_str:
        instructions.append(instruction(ins))
    start_address = 64
    PC = [start_address]
    disassembly_file = open('project1/disassembly.txt', 'w')
    data_index = 0
    for ins in instructions:
        print(ins)
        flag = generate_disassembly(ins, PC, disassembly_file)
        PC[0] += 4
        data_index += 1
        if flag:
            break
    data_ins = instructions[data_index:]
    count = 0
    for data in data_ins:
        current_data_index = int((PC[0] - start_address - 4 * data_index) / 4)
        memory[current_data_index] = complement_to_int(str(data))
        disassembly_file.write(str(data) +
                               '\t' +
                               str(PC[0]) +
                               '\t' +
                               str(complement_to_int(str(data))) +
                               '\n')
        PC[0] += 4
        count += 1
    disassembly_file.close()
    simulation_file = open('project1/simulation.txt', 'w')
    cnt = 1
    PC[0] = start_address
    data_address = start_address + 4 * data_index
    flag = False
    while not flag:
        index = int((PC[0] - start_address) / 4)
        flag = generate_simulation(
            instructions[index],
            PC,
            data_address,
            registers,
            memory,
            simulation_file,
            count,
            cnt)
        cnt += 1
    simulation_file.close()
