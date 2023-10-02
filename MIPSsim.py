'''
On my honor, I have neither given nor received unauthorized aid on this assignment
'''


import numpy as np
import sys

# 指令类


class instruction:
    category = 0
    op_code = ''
    rs = ''
    rt = ''
    rd = ''
    shift_amount = ''
    func = ''
    ins_str = ''

    def __init__(self, instruction_string):
        category_str = instruction_string[0:2]
        if category_str == '01':
            self.category = 0
        else:
            self.category = 1
        self.op_code = instruction_string[2:6]
        self.rs = instruction_string[6:11]
        self.rt = instruction_string[11:16]
        self.rd = instruction_string[16:21]
        self.shift_amount = instruction_string[21:26]
        self.func = instruction_string[26:]
        self.ins_str = instruction_string

    def __str__(self):
        return self.ins_str


# 读取二进制文件
def read_input_file(filename):
    instructions = []
    with open(filename, "r") as reader:
        for line in reader.readlines():
            instructions.append(line.strip())

    return instructions

# 对寄存器进行格式化输出


def generate_register_name(reg):
    reg_name = 'R' + str(int(reg, 2))
    return reg_name

# 补码转换为整数


def complement_to_int(complement):
    res = 0
    if complement[0] == '0':
        res = int(complement, base=2)
    else:
        com_len = len(complement)
        res += -1 * 2**(com_len - 1)
        for i in range(1, len(complement)):
            res += int(complement[i]) * 2**(com_len - 1 - i)
    return res


def int_to_complement(num, width=32):
    return np.binary_repr(num, width=width)

# offset为有符号整数，offset需要左移并且扩充为32位


def sign_extend(complement, width=32):
    res = complement[0] * (width - len(complement)) + complement
    return res


def zero_extend(complement, width=32):
    res = '0' * (width - len(complement)) + complement
    return res


def disassemble_J(instruction):
    immediate = instruction.rs + instruction.rt + \
        instruction.shift_amount + instruction.func + '00'
    ins_arg_str = '#' + str(complement_to_int(immediate))
    return ins_arg_str


def simulate_J(instruction, PC, data_address, regsters, memory):
    immediate = instruction.rs + instruction.rt + \
        instruction.shift_amount + instruction.func + '00'
    PC[0] = complement_to_int(immediate)


def disassemble_JR(instruction):
    ins_arg_str = generate_register_name(instruction.rs)
    return ins_arg_str


def simulate_JR(instruction, PC, data_address, regsters, memory):
    PC[0] = registers[complement_to_int(instruction.rs)]


def disassemble_BEQ(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    # 右移2位数
    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    ins_arg_str = rs + ', ' + rt + ', ' + '#' + str(complement_to_int(offset))

    return ins_arg_str


def simulate_BEQ(instruction, PC, data_address, regsters, memory):
    # 右移2位数
    offset = instruction.rd + instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    if registers[int(instruction.rs, 2)] == registers[int(instruction.rt, 2)]:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4

# BLTZ rs, offset


def disassemble_BLTZ(instruction):
    rs = generate_register_name(instruction.rs)
    offset = instruction.rd + instruction.rt + \
        instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    ins_arg_str = rs + ', ' + '#' + str(complement_to_int(offset))

    return ins_arg_str


def simulate_BLTZ(instruction, PC, data_address, regsters, memory):
    offset = instruction.rd + instruction.rt + \
        instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    if registers[int(instruction.rs, 2)] < 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4

# BGTZ rs, offset


def disassemble_BGTZ(instruction):
    rs = generate_register_name(instruction.rs)
    offset = instruction.rd + instruction.rt + \
        instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    ins_arg_str = rs + ', ' + '#' + str(complement_to_int(offset))

    return ins_arg_str


def simulate_BGTZ(instruction, PC, data_address, regsters, memory):
    offset = instruction.rd + instruction.rt + \
        instruction.shift_amount + instruction.func + '00'
    # 符号扩充为32位
    offset = sign_extend(offset, width=32)

    if registers[int(instruction.rs, 2)] > 0:
        PC[0] += complement_to_int(offset) + 4
    else:
        PC[0] += 4

# BREAK指令无输出


def disassemble_BREAK(instruction):
    return ""


def simulate_BREAK(instruction, PC, data_address, regsters, memory):
    PC[0] += 4

# SW rt, offset(base)


def disassemble_SW(instruction):
    base = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    offset = instruction.rd + instruction.shift_amount + instruction.func
    ins_arg_str = rt + ', ' + str(complement_to_int(offset)) + '(' + base + ')'
    return ins_arg_str


def simulate_SW(instruction, PC, data_address, regsters, memory):
    base = int(instruction.rs, 2)

    offset = instruction.rd + instruction.shift_amount + instruction.func

    valid_address = int(
        (registers[base] +
         complement_to_int(offset) -
         data_address) /
        4)
    memory[int(valid_address)] = registers[int(instruction.rt, 2)]

    PC[0] += 4


# LW rt, offset(base)
def disassemble_LW(instruction):
    base = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    offset = instruction.rd + instruction.shift_amount + instruction.func
    ins_arg_str = rt + ', ' + str(complement_to_int(offset)) + '(' + base + ')'
    return ins_arg_str


def simulate_LW(instruction, PC, data_address, regsters, memory):
    base = int(instruction.rs, 2)

    offset = instruction.rd + instruction.shift_amount + instruction.func

    valid_address = int(
        (registers[base] +
         complement_to_int(offset) -
         data_address) /
        4)
    registers[int(instruction.rt, 2)] = memory[int(valid_address)]

    PC[0] += 4


# SLL rd, rt, sa
def disassemble_SLL(instruction):
    rd = generate_register_name(instruction.rd)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rt + ', ' + '#' + \
        str(complement_to_int(instruction.shift_amount))
    return ins_arg_str


def simulate_SLL(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = (registers[int(instruction.rt, 2)] << int(
        instruction.shift_amount, 2)) & (2 ** 32 - 1)

    PC[0] += 4

# SRL rd, rt, sa


def disassemble_SRL(instruction):
    rd = generate_register_name(instruction.rd)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rt + ', ' + '#' + \
        str(complement_to_int(instruction.shift_amount))
    return ins_arg_str


def simulate_SRL(instruction, PC, data_address, regsters, memory):
    sa = complement_to_int(instruction.shift_amount)

    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]

    res = '0' * sa + res

    registers[int(instruction.rd, 2)] = complement_to_int(res)

    PC[0] += 4

# SRA rd, rt, sa


def disassemble_SRA(instruction):
    rd = generate_register_name(instruction.rd)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rt + ', ' + '#' + \
        str(complement_to_int(instruction.shift_amount))
    return ins_arg_str


def simulate_SRA(instruction, PC, data_address, regsters, memory):
    sa = complement_to_int(instruction.shift_amount)

    s = 32 - sa
    res = int_to_complement(registers[int(instruction.rt, 2)])[:s]

    res = res[0] * sa + res

    registers[int(instruction.rd, 2)] = complement_to_int(res)
    PC[0] += 4

# NOP


def disassemble_NOP(instruction):
    return ""


def simulate_NOP(instruction, PC, data_address, regsters, memory):
    PC[0] += 4

# ADD rd, rs, rt


def disassemble_ADD(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_ADD(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = registers[int(
        instruction.rs, 2)] + registers[int(instruction.rt, 2)]
    PC[0] += 4

# SUB rd, rs, rt


def disassemble_SUB(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_SUB(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = registers[int(
        instruction.rs, 2)] - registers[int(instruction.rt, 2)]
    PC[0] += 4


def disassemble_MUL(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_MUL(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = registers[int(
        instruction.rs, 2)] * registers[int(instruction.rt, 2)]
    PC[0] += 4


def disassemble_AND(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_AND(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = registers[int(
        instruction.rs, 2)] & registers[int(instruction.rt, 2)]
    PC[0] += 4


def disassemble_OR(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_OR(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = registers[int(
        instruction.rs, 2)] | registers[int(instruction.rt, 2)]
    PC[0] += 4


def disassemble_XOR(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_XOR(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = registers[int(
        instruction.rs, 2)] ^ registers[int(instruction.rt, 2)]
    PC[0] += 4


def disassemble_NOR(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_NOR(instruction, PC, data_address, regsters, memory):
    registers[int(instruction.rd, 2)] = ~(
        registers[int(instruction.rs, 2)]) | registers[int(instruction.rt, 2)]
    PC[0] += 4


def disassemble_SLT(instruction):
    rd = generate_register_name(instruction.rd)
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)

    ins_arg_str = rd + ', ' + rs + ', ' + rt
    return ins_arg_str


def simulate_SLT(instruction, PC, data_address, regsters, memory):
    if registers[int(instruction.rs, 2)] < registers[int(instruction.rt, 2)]:
        registers[int(instruction.rd, 2)] = 1
    else:
        registers[int(instruction.rd, 2)] = 0

    PC[0] += 4

# ADDI rt, rs, immediate


def disassemble_ADDI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = sign_extend(immediate)
    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str


def simulate_ADDI(instruction, PC, data_address, regsters, memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = sign_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(
        instruction.rs, 2)] + complement_to_int(immediate)
    PC[0] += 4


def disassemble_ANDI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str


def simulate_ANDI(instruction, PC, data_address, regsters, memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = zero_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(
        instruction.rs, 2)] & complement_to_int(immediate)
    PC[0] += 4


def disassemble_ORI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func

    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str


def simulate_ORI(instruction, PC, data_address, regsters, memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = zero_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(
        instruction.rs, 2)] | complement_to_int(immediate)
    PC[0] += 4


def disassemble_XORI(instruction):
    rs = generate_register_name(instruction.rs)
    rt = generate_register_name(instruction.rt)
    immediate = instruction.rd + instruction.shift_amount + instruction.func

    ins_arg_str = rt + ', ' + rs + ', #' + str(complement_to_int(immediate))

    return ins_arg_str


def simulate_XORI(instruction, PC, data_address, regsters, memory):
    immediate = instruction.rd + instruction.shift_amount + instruction.func
    immediate = zero_extend(immediate)
    registers[int(instruction.rt, 2)] = registers[int(
        instruction.rs, 2)] ^ complement_to_int(immediate)
    PC[0] += 4


op_code_dict = {
    '0000': ['J', 'ADD'],
    '0001': ['JR', 'SUB'],
    '0010': ['BEQ', 'MUL'],
    '0011': ['BLTZ', 'AND'],
    '0100': ['BGTZ', 'OR'],
    '0101': ['BREAK', 'XOR'],
    '0110': ['SW', 'NOR'],
    '0111': ['LW', 'SLT'],
    '1000': ['SLL', 'ADDI'],
    '1001': ['SRL', 'ANDI'],
    '1010': ['SRA', 'ORI'],
    '1011': ['NOP', 'XORI']
}

function_dict = {
    'J': ['disassemble_J', 'simulate_J'],
    'JR': ['disassemble_JR', 'simulate_JR'],
    'BEQ': ['disassemble_BEQ', 'simulate_BEQ'],
    'BLTZ': ['disassemble_BLTZ', 'simulate_BLTZ'],
    'BGTZ': ['disassemble_BGTZ', 'simulate_BGTZ'],
    'BREAK': ['disassemble_BREAK', 'simulate_BREAK'],
    'SW': ['disassemble_SW', 'simulate_SW'],
    'LW': ['disassemble_LW', 'simulate_LW'],
    'SLL': ['disassemble_SLL', 'simulate_SLL'],
    'SRL': ['disassemble_SRL', 'simulate_SRL'],
    'SRA': ['disassemble_SRA', 'simulate_SRA'],
    'NOP': ['disassemble_NOP', 'simulate_NOP'],
    'ADD': ['disassemble_ADD', 'simulate_ADD'],
    'SUB': ['disassemble_SUB', 'simulate_SUB'],
    'MUL': ['disassemble_MUL', 'simulate_MUL'],
    'AND': ['disassemble_AND', 'simulate_AND'],
    'OR': ['disassemble_OR', 'simulate_OR'],
    'XOR': ['disassemble_XOR', 'simulate_XOR'],
    'NOR': ['disassemble_NOR', 'simulate_NOR'],
    'SLT': ['disassemble_SLT', 'simulate_SLT'],
    'ADDI': ['disassemble_ADDI', 'simulate_ADDI'],
    'ANDI': ['disassemble_ANDI', 'simulate_ANDI'],
    'ORI': ['disassemble_ORI', 'simulate_ORI'],
    'XORI': ['disassemble_XORI', 'simulate_XORI'],
}


def generate_disassembly(instruction, PC, disassemble_file):
    ins_name_str = op_code_dict[instruction.op_code][instruction.category]
    ins_args_str = eval(function_dict[ins_name_str][0])(instruction)

    if ins_name_str != 'BREAK':
        disassemble_file.write(str(instruction) +
                               '\t' +
                               str(PC[0]) +
                               '\t' +
                               ins_name_str +
                               ' ' +
                               ins_args_str +
                               '\n')
    else:
        disassemble_file.write(str(instruction) + '\t' +
                               str(PC[0]) + '\t' + ins_name_str + '\n')
        return True

    return False


# registers
registers = [0] * 32
# memory
memory = [0] * 64


def generate_simulation(
        instruction,
        PC,
        data_address,
        registers,
        memory,
        simulation_file,
        count,
        cnt):
    ins_name_str = op_code_dict[instruction.op_code][instruction.category]
    ins_args_str = eval(function_dict[ins_name_str][0])(instruction)

    simulation_file.write("--------------------\n")

    flag = False

    if ins_name_str != 'BREAK':
        simulation_file.write("Cycle:" +
                              str(cnt) +
                              '\t' +
                              str(PC[0]) +
                              '\t' +
                              ins_name_str +
                              ' ' +
                              ins_args_str +
                              '\n')
    else:
        simulation_file.write("Cycle:" + str(cnt) + '\t' +
                              str(PC[0]) + '\t' + ins_name_str + '\n')
        flag = True

    simulation_file.write('\n')

    eval(
        function_dict[ins_name_str][1])(
        instruction,
        PC,
        data_address,
        registers,
        memory)

    simulation_file.write('Registers\n')
    simulation_file.write('R00:' +
                          '\t' +
                          str(registers[0]) +
                          '\t' +
                          str(registers[1]) +
                          '\t' +
                          str(registers[2]) +
                          '\t' +
                          str(registers[3]) +
                          '\t' +
                          str(registers[4]) +
                          '\t' +
                          str(registers[5]) +
                          '\t' +
                          str(registers[6]) +
                          '\t' +
                          str(registers[7]) +
                          '\n')
    simulation_file.write('R08:' +
                          '\t' +
                          str(registers[8]) +
                          '\t' +
                          str(registers[9]) +
                          '\t' +
                          str(registers[10]) +
                          '\t' +
                          str(registers[11]) +
                          '\t' +
                          str(registers[12]) +
                          '\t' +
                          str(registers[13]) +
                          '\t' +
                          str(registers[14]) +
                          '\t' +
                          str(registers[15]) +
                          '\n')
    simulation_file.write('R16:' +
                          '\t' +
                          str(registers[16]) +
                          '\t' +
                          str(registers[17]) +
                          '\t' +
                          str(registers[18]) +
                          '\t' +
                          str(registers[19]) +
                          '\t' +
                          str(registers[20]) +
                          '\t' +
                          str(registers[21]) +
                          '\t' +
                          str(registers[22]) +
                          '\t' +
                          str(registers[23]) +
                          '\n')
    simulation_file.write('R24:' +
                          '\t' +
                          str(registers[24]) +
                          '\t' +
                          str(registers[25]) +
                          '\t' +
                          str(registers[26]) +
                          '\t' +
                          str(registers[27]) +
                          '\t' +
                          str(registers[28]) +
                          '\t' +
                          str(registers[29]) +
                          '\t' +
                          str(registers[30]) +
                          '\t' +
                          str(registers[31]) +
                          '\n')
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


if __name__ == '__main__':

    # 以命令行方式读取
    # input_filename = sys.argv[1]
    input_filename = 'sample.txt'

    ins_strs = read_input_file(input_filename)
    instructions = []
    for ins in ins_strs:
        instructions.append(instruction(ins))

    PC = [256]
    disassemble_file = open('disassembly.txt', 'w')
    data_index = 0

    for ins in instructions:
        flag = generate_disassembly(ins, PC, disassemble_file)
        PC[0] += 4
        data_index += 1
        if flag:
            break

    # data段
    data_ins = instructions[data_index:]

    count = 0
    for data in data_ins:
        current_data_index = int((PC[0] - 256 - 4 * data_index) / 4)
        memory[current_data_index] = complement_to_int(str(data))
        disassemble_file.write(str(data) +
                               '\t' +
                               str(PC[0]) +
                               '\t' +
                               str(complement_to_int(str(data))) +
                               '\n')
        PC[0] += 4
        count += 1

    disassemble_file.close()

    simulation_file = open('simulation.txt', 'w')
    cnt = 1
    PC[0] = 256

    data_address = data_index * 4 + 256
    flag = False
    while not flag:
        index = int((PC[0] - 256) / 4)
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
        if flag:
            break

    simulation_file.close()
