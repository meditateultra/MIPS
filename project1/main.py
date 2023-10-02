import numpy as np
from utils import *
from instruction import *
from simulation import *


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


def generate_disassembly(instruction, PC, disassembly_file):
    if instruction.op_code == '000000':
        ins_name_str = code_map_func[instruction.func]
    else:
        ins_name_str = code_map_op[instruction.op_code]
    ins_args_str = generate_instruction_str(ins_name_str, instruction)

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

    ins_args_str = generate_instruction_str(ins_name_str, instruction)
    # ins_args_str = eval(name_map_function[ins_name_str][0])(instruction)

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
        name_map_function[ins_name_str])(
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

easy_two_ins = ['ADD', 'SUB', 'MUL', 'AND', 'NOR', 'SLT']
cond_jump_ins = ['BLTZ', 'BGTZ']
easy_load_ins = ['SW', 'LW']
easy_move_ins = ['SLL', 'SRL', 'SRA']

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

input_path = './sample.txt'

if __name__ == '__main__':
    registers = [0] * 32
    memory = [0] * 64
    ins_str = read_input_file(input_path)
    instructions = []
    for ins in ins_str:
        instructions.append(instruction(ins))
    start_address = 64
    PC = [start_address]
    disassembly_file = open('disassembly.txt', 'w')
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
    simulation_file = open('simulation.txt', 'w')
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
